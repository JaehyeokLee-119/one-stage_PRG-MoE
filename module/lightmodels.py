import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup
from module.evaluation import log_metrics, FocalLoss
from sklearn.metrics import classification_report
from transformers import AutoModel, AutoModelForSequenceClassification, BertModel, AutoTokenizer
import numpy as np

class OneStageModel(pl.LightningModule):
    def __init__(self, encoder_name, freeze_ratio=0.0, guiding_lambda=0.6, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        self.guiding_lambda = guiding_lambda
        
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        self.pair_hidden_size = 256
        
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(encoder_name, output_hidden_states=True, num_labels=n_emotion)
        
        # 모델에 speaker 토큰 추가
        tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)
        tokens = ["[Speaker A]", "[Speaker B]", "[/Speaker A]", "[/Speaker B]"]
        tokenizer_.add_tokens(tokens, special_tokens=True)
        self.model.resize_token_embeddings(len(tokenizer_)) 
        
        pair_embedding_size = 2 * (self.model.config.hidden_size + 1) #: 2 * (768 + 1) = 1536
        
        self.model = AutoModel.from_pretrained(encoder_name, output_hidden_states=True, num_labels=n_emotion)
        self.pair_classification_layer1 = nn.Linear(pair_embedding_size, self.pair_hidden_size)
        self.pair_classification_layer2 = nn.Linear(self.pair_hidden_size, n_emotion*n_cause)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):
        outputs = self.model(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        pooled_output = outputs[0][:,0,:]
        # pooled_output = outputs[1][-1][:,0,:]
        pair_embedding = self.get_pair_embedding(pooled_output, input_ids, attention_mask, token_type_ids, speaker_ids)
        pair_classification_result = self.pair_classification_layer2(self.pair_classification_layer1(pair_embedding))
        
        return pair_classification_result # size: [n_emotion * n_cause]

    def get_pair_embedding(self, pooled_output, input_ids, attention_mask, token_type_ids, speaker_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        
        utterance_representation = self.dropout(pooled_output)

        concatenated_embedding = torch.cat((speaker_ids.view(-1).unsqueeze(1), utterance_representation), dim=1)
        
        pair_embedding = list()
        for batch in concatenated_embedding.view(batch_size, max_doc_len, -1):
            pair_per_batch = list()
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    # backward 시, cycle이 생겨 문제가 생길 경우, batch[end_t].detach() 시도.
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t])))
            pair_embedding.append(torch.stack(pair_per_batch))

        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)

        return pair_embedding

class TotalModel_exp12(pl.LightningModule):
    def __init__(self, encoder_name, freeze_ratio=0.0, guiding_lambda=0.6, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        self.guiding_lambda = guiding_lambda
        self.n_expert = 14
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(encoder_name, output_hidden_states=True, num_labels=n_emotion)
        if freeze_ratio > 0:
            for name, param in self.model.named_parameters():
                for i in range(int(self.model.config.num_hidden_layers * freeze_ratio)):
                    if str(i) in name:
                        param.requires_grad = False
            
        
        pair_embedding_size = 2 * (self.model.config.hidden_size + n_emotion + 1)
        
        self.fc_cau = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.gating_network = nn.Linear(pair_embedding_size, self.n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(self.n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(pair_embedding_size, 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):
        
        outputs = self.model(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        pooled_output = outputs[1][-1][:,0,:]
        emotion_prediction = outputs[0]
        
        # encoder(AutoModelForClassification)에서 분류 결과와 pooled_output을 받아와야 함
        
        pair_embedding = self.get_pair_embedding(pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
            # -> 2030개의 pair_embedding 생성됨 ([5, 406, 1552])
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach()) # [5, 406, 1552] -> [2030, 1552]
            # pair마다 gating probability를 계산 [2030, 4]

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return emotion_prediction, cause_pred

    def get_pair_embedding(self, pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        
        utterance_representation = self.dropout(pooled_output)

        concatenated_embedding = torch.cat((utterance_representation, emotion_prediction, 
                                            speaker_ids.view(-1).unsqueeze(1)), dim=1)

                                    # utterance_representation [140, 768], emotion_prediction [140, 7], speaker_ids [5,28] -> [140, 1]
                                    # concatenated_embedding = [140, 776]
                                    
        pair_embedding = list()
        for batch in concatenated_embedding.view(batch_size, max_doc_len, -1): # [140,776] -> [5,28,776]
            pair_per_batch = list() # batch: [28,776]
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    # backward 시, cycle이 생겨 문제가 생길 경우, batch[end_t].detach() 시도.
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t])))
            pair_embedding.append(torch.stack(pair_per_batch))

        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)

        return pair_embedding # [5, 406, 1552]
    
    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        pair_info = []
        for speaker_batch, emotion_batch in zip(speaker_ids.view(batch_size, max_doc_len, -1), emotion_prediction.view(batch_size, max_doc_len, -1)):
            info_pair_per_batch = []
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    speaker_condition = speaker_batch[t] == speaker_batch[end_t]
                    
                    # 00,   10,     20,     30,     40,     50,     01,     11,     21,     31,     41,     51
                    # n_expert = 12
                    # 0     1       2       3       4       5       6       7       8       9       10      11
                    one_hot_index = int(speaker_condition*(self.n_expert/2) + torch.argmax(emotion_batch[end_t]))
                    emotion_condition = torch.argmax(
                        emotion_batch[t]) == torch.argmax(emotion_batch[end_t])
                    one_hot_position = torch.Tensor([0] * self.n_expert)
                    one_hot_position[one_hot_index] = 1
                    info_pair_per_batch.append(one_hot_position)
            pair_info.append(torch.stack(info_pair_per_batch))

        pair_info = torch.stack(pair_info).to(input_ids.device)

        return pair_info
