import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup
from module.evaluation import FocalLoss #log_metrics, 
from sklearn.metrics import classification_report, precision_score , recall_score , confusion_matrix
from transformers import AutoModel
import numpy as np
from module.lightmodels import OneStageModel

class LitPRGMoE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # 모델 셋팅 파라미터
        self.encoder_name = kwargs['encoder_name']
        # Model
        
        self.use_original = kwargs['use_original']
        self.use_exp12 = kwargs['use_exp12']
        
        self.freeze_ratio = kwargs['freeze_ratio']
        self.model = OneStageModel(self.encoder_name, freeze_ratio=self.freeze_ratio) # output: (emotion prediction, cause prediction)

        # 하이퍼파라미터 설정
        self.training_iter = kwargs['training_iter']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.window_size = kwargs['window_size']
        self.n_expert = 4
        self.n_emotion = 7
        self.guiding_lambda = kwargs['guiding_lambda']
        self.loss_lambda = kwargs['loss_lambda'] # loss 중 Emotion loss의 비율
        # 학습 방법 설정
        self.n_cause = kwargs['n_cause']
        self.ckpt_type = kwargs['ckpt_type']
        self.multiclass_avg_type = kwargs['multiclass_avg_type']
        
        # 모델 내 학습 중 변수 설정
        self.test = False # True when testing(on_test_epoch_start ~ on_test_epoch_end)
            # test에서 joint_accuracy를 계산
        self.train_type = 'total'
        
        if 'bert-base' in self.encoder_name:
            self.is_bert_like = True
        else:
            self.is_bert_like = False
                        
        # Dictionaries for logging
        types = ['train', 'valid', 'test']
        self.pair_emotion_prediction = {}
        self.pair_emotion_label = {}
        self.loss_sum = {}
        self.batch_count = {}
        
        for i in types:
            self.pair_emotion_prediction[i] = []
            self.pair_emotion_label[i] = []
            self.loss_sum[i] = 0.0
            self.batch_count[i] = 0
        
        self.best_performance_emo = {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'epoch': 0,
            'loss': 0.0,
        }
        self.best_performance_cau = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'epoch': 0,
            'loss': 0.0,
        }
        self.best_performance_emo_cau = {
            'epoch': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=5,
                                                    num_training_steps=self.training_iter,
                                                    )
        return [optimizer], [scheduler]
                    
    def forward(self, batch):
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        
        batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
        
        input_ids = utterance_input_ids_batch
        attention_mask = utterance_attention_mask_batch
        token_type_ids = utterance_token_type_ids_batch
        speaker_ids = speaker_batch
        
        # Forward
        pair_classification_result = self.model(input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len)
        
        return pair_classification_result
    
    def output_processing(self, utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, pair_classification_result):
        # 모델의 forward 결과(emotion-cause pair - emotion 분류 결과) 로부터 loss 계산과 로깅을 위한 input 리턴
        # 일단은 emotion-cause 분류 결과랑 라벨만 바로 1:1 비교 가능하도록 프로세싱하기
        
        batch_size, _, _ = utterance_input_ids_batch.shape
        check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=1000, )

        # emotion label을 pair candidate만큼 늘림
        emotion_label_pair_list = [] 
        for doc_emotion in emotion_label_batch:
            end_t = 0
            for emotion in doc_emotion:
                for _ in range(end_t+1):
                    emotion_label_pair_list.append(emotion)
                end_t += 1
        emotion_pair_true_expanded = torch.stack(emotion_label_pair_list).view(batch_size, -1)
        
        pair_emotion_label_batch_all = emotion_pair_true_expanded[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        # Emotion label에 10을 곱함 (0~6 -> 0~60). Cause 여부는 0,1만 있기 때문에 10을 넘길 수 없어서 이래야 충돌 없음
        # pair_emotion_cause_prediction_all = torch.argmax(pair_classification_result, dim=-1)
        
        # pair_binary_cause_label_batch_all: 1 = pair가 맞음, 0 = pair가 아님
        pair_emotion_cause_prediction_all = pair_classification_result[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        pair_emotion_cause_label_batch_all = pair_emotion_label_batch_all + (pair_binary_cause_label_batch_all==0)*self.n_emotion
                # pair가 아니면 7을 더하고, 거기다가 emotion 값을 더함. 아니면 그냥 emotion 값을 가짐
                # 0~6: pair맞음 & angry~neutral, 7~13: pair아님 & angry~neutral
        emotion_cause_ = (pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all)
        return emotion_cause_
    
    def loss_calculation(self, pair_emo_cause_prediction, pair_emo_cause_label):
        criterion = FocalLoss(gamma=2)
        loss = criterion(pair_emo_cause_prediction, pair_emo_cause_label)
        return loss
    
    def training_step(self, batch, batch_idx):
        types = 'train'
        # utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        pair_classification_result = self.forward(batch)
        # Output processing
        emotion_cause_ = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, pair_classification_result)
        (pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all) = emotion_cause_
        
        # Logging
        self.pair_emotion_prediction[types].append(torch.argmax(pair_emotion_cause_prediction_all, dim=-1))
        self.pair_emotion_label[types].append(pair_emotion_cause_label_batch_all)
        # Loss Calculation
        loss = self.loss_calculation(pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all)
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        types = 'valid'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        pair_classification_result = self.forward(batch)
        
        # Output processing
        emotion_cause_ = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, pair_classification_result)
        (pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all) = emotion_cause_
        
        # Loss Calculation
        loss = self.loss_calculation(pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all)
            
        self.log("valid_loss: ", loss, sync_dist=True)
        # Logging
        self.pair_emotion_prediction[types].append(torch.argmax(pair_emotion_cause_prediction_all, dim=-1))
        self.pair_emotion_label[types].append(pair_emotion_cause_label_batch_all)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
    def test_step(self, batch, batch_idx):
        types = 'test'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        pair_classification_result = self.forward(batch)
        
        # Output processing
        emotion_cause_ = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, pair_classification_result)
        (pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all) = emotion_cause_
        
        # Loss Calculation
        loss = self.loss_calculation(pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_all)
            
        self.log("test_loss: ", loss, sync_dist=True)
        # Logging
        self.pair_emotion_prediction[types].append(torch.argmax(pair_emotion_cause_prediction_all, dim=-1))
        self.pair_emotion_label[types].append(pair_emotion_cause_label_batch_all)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1

    def on_train_epoch_start(self):
        self.make_test_setting(types='train')
        print('Train type: ', self.train_type)
        
    def on_train_epoch_end(self):
        self.log_test_result(types='train')
    
    def on_validation_epoch_start(self):
        self.test = True
        self.make_test_setting(types='valid')

    def on_validation_epoch_end(self):
        self.test = False
        self.log_test_result(types='valid')
    
    def on_test_epoch_start(self):
        self.test = True
        self.make_test_setting(types='test')
        
    def on_test_epoch_end(self):
        self.test = False
        self.log_test_result(types='test')
        
    def make_test_setting(self, types='train'):
        self.pair_emotion_prediction[types] = []
        self.pair_emotion_label[types] = []
        self.loss_sum[types] = 0.0
        self.batch_count[types] = 0
        
    def log_test_result(self, types='train'):
        logger = logging.getLogger(types)
        
        loss_avg = self.loss_sum[types] / self.batch_count[types]
        
        emo_dict, emo_metrics, p_cau, r_cau, f1_cau, p_emo_cau, r_emo_cau, f1_emo_cau = log_metrics(self.pair_emotion_prediction[types], self.pair_emotion_label[types], 
                                                 loss_avg, self.multiclass_avg_type)
        
        self.log('total-precision', p_emo_cau, sync_dist=True)
        self.log('total-recall', r_emo_cau, sync_dist=True)
        self.log('total-f1', f1_emo_cau, sync_dist=True)
        
        logging_texts = f'''\n[Epoch {self.current_epoch}] / <Emotion Prediction> of {types}
Train type: {self.train_type}\n
{emo_dict}\n
<Emotion-Cause Prediction>
\tprecision:\t{p_cau}
\trecall:   \t{r_cau}
\tf1-score: \t{f1_cau}
<Pair-Emotion Prediction>
\tprecision:\t{p_emo_cau}
\trecall:   \t{r_emo_cau}
\tf1-score: \t{f1_emo_cau}
\n'''
                        
        if (types == 'valid'):
            if (self.best_performance_emo['weighted_f1'] < emo_metrics[2]):
                self.best_performance_emo['weighted_f1'] = emo_metrics[2]
                self.best_performance_emo['accuracy'] = emo_metrics[0]
                self.best_performance_emo['macro_f1'] = emo_metrics[1]
                self.best_performance_emo['epoch'] = self.current_epoch
                self.best_performance_emo['loss'] = loss_avg
            if (self.best_performance_cau['f1'] < f1_cau):
                self.best_performance_cau['f1'] = f1_cau
                self.best_performance_cau['precision'] = p_cau
                self.best_performance_cau['recall'] = r_cau
                self.best_performance_cau['epoch'] = self.current_epoch
                self.best_performance_cau['loss'] = loss_avg
            if (self.best_performance_emo_cau['f1'] < f1_emo_cau):
                self.best_performance_emo_cau['precision'] = p_emo_cau
                self.best_performance_emo_cau['recall'] = r_emo_cau
                self.best_performance_emo_cau['f1'] = f1_emo_cau
                self.best_performance_emo_cau['epoch'] = self.current_epoch
            
            appended_log_valid = f'''\nCurrent Best Performance: loss: {self.best_performance_cau["loss"]}
\t<Emotion Prediction: [Epoch: {self.best_performance_emo["epoch"]}]>
\t\taccuracy: \t{self.best_performance_emo["accuracy"]}
\t\tmacro_f1: \t{self.best_performance_emo["macro_f1"]}
\t\tweighted_f1: \t{self.best_performance_emo["weighted_f1"]}
\t<Emotion-Cause Prediction: [Epoch: {self.best_performance_cau["epoch"]}]>
\t\taccuracy: \t{self.best_performance_cau["accuracy"]}
\t\tprecision: \t{self.best_performance_cau["precision"]}
\t\trecall: \t{self.best_performance_cau["recall"]}
\t\tf1:\t\t{self.best_performance_cau["f1"]}
\t<Pair-Emotion Prediction: [Epoch: {self.best_performance_emo_cau["epoch"]}]>
\t\tprecision: \t{self.best_performance_emo_cau["precision"]}
\t\trecall: \t{self.best_performance_emo_cau["recall"]}
\t\tf1:\t\t{self.best_performance_emo_cau["f1"]}\n
'''
            
        if (types == 'valid'):
            logging_texts += appended_log_valid
            
        # if (types == 'test'):
        # label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        # label_ = np.array([1, 11, 21, 31, 41, 51, 61, 0, 10, 20, 30, 40, 50, 60])
        
        label_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        confusion_all = confusion_matrix(torch.cat(self.pair_emotion_label[types]).to('cpu'), torch.cat(self.pair_emotion_prediction[types]).to('cpu'), labels=label_)
        label_name = np.array(['P-angry', 'P-disgu', 'P-fear', 'P-happy', 'P-sad', 'P-surpr', 'P-neutr', 'N-angry', 'N-disgu', 'N-fear', 'N-happy', 'N-sad', 'N-surpr', 'N-neutr'])
    
        confusion_log = ""
        confusion_log+="[Confusion_all matrix]\n"
        confusion_log+='\t'
        for label in label_name:
            confusion_log+=f'{label}\t'
        confusion_log+="\n"
        for row, label in zip(confusion_all, label_name):
            confusion_log+=f'{label}\t'
            for col in row:
                confusion_log+=f'{col}\t'
            confusion_log+="\n"
        logger.info(confusion_log+logging_texts)
            
def log_metrics(pair_emotion_prediction, pair_emotion_label, loss_avg, multiclass_avg_type):
    # label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    label_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # label_ = np.array(['P-angry', 'P-disgust', 'P-fear', 'P-happy', 'P-sad', 'P-surprise', 'P-neutral', 'N-angry', 'N-disgust', 'N-fear', 'N-happy', 'N-sad', 'N-surprise', 'N-neutral'])
    confusion_all = confusion_matrix(torch.cat(pair_emotion_label).to('cpu'), torch.cat(pair_emotion_prediction).to('cpu'), labels=label_)
    
    # 가공해서 Emotion만 빼내기
    emotion_prediction = torch.cat(pair_emotion_prediction).to('cpu') % 7
    emotion_label = torch.cat(pair_emotion_label).to('cpu') % 7
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    emo_report_dict = metrics_report(emotion_prediction, emotion_label, label=label_, get_dict=True)
    emo_report_str = metrics_report(emotion_prediction, emotion_label, label=label_, get_dict=False)
    acc_emo, macro_f1, weighted_f1 = emo_report_dict['accuracy'], emo_report_dict['macro avg']['f1-score'], emo_report_dict['weighted avg']['f1-score']
    emo_metrics = (acc_emo, macro_f1, weighted_f1)
    
    # 가공해서 Cause만 빼내기
    cause_prediction = torch.cat(pair_emotion_prediction).to('cpu') // 7
    cause_label = torch.cat(pair_emotion_label).to('cpu') // 7
    label_ = np.array(['Cause', 'No cause'])
    cause_report_dict = metrics_report(cause_prediction, cause_label, label=label_, get_dict=True)
    if 'Cause' in cause_report_dict.keys():   #추가된 부분
        acc_cau, p_cau, r_cau, f1_cau = cause_report_dict['accuracy'], cause_report_dict['Cause']['precision'], cause_report_dict['Cause']['recall'], cause_report_dict['Cause']['f1-score']
    else:   #추가된 부분
        acc_cau, p_cau, r_cau, f1_cau = 0, 0, 0, 0   #추가된 부분
    
    
    
    # 1, 11, 21, 31, 41, 51 순서로
    idx_list = [0, 1, 2, 3, 4, 5]
    pred_num_acc_list_all = [0, 0, 0, 0, 0, 0]
    pred_num_recall_denominator_dict = [0, 0, 0, 0, 0, 0]
    pred_num_precision_denominator_dict = [0, 0, 0, 0, 0, 0]
    
    for i in idx_list:
        pred_num_acc_list_all[i] = confusion_all[i][i]
        pred_num_precision_denominator_dict[i] = sum(confusion_all[:,i])
        pred_num_recall_denominator_dict[i] = sum(confusion_all[i,:])
        
    # support_ratio = [0, 0, 0, 0, 0, 0]
    # for i in range(len(support_ratio)):
    #     support_ratio[i] = pred_num_recall_denominator_dict[i]/sum(pred_num_recall_denominator_dict)
        
    '''# Confusion matrix 시각화
    print('\t', end="")
    for label in label_:
        print(label, '\t', end="")
    print("")
    for row, label in zip(confusion_pred, label_):
        print(label, '\t', end="")
        for col in row:
            print(col, '\t', end="")
        print("")
        '''
    '''
    <Pair-emotion F1 알고리즘 정리>
    - 각 Pair-Emotion 예측 여부에 따라 7*2 크기의 confusion matrix를 만듦
    - 클래스 1, 11, 21, 31, 41, 51 [(pair,angry), (pair,disgust), (pair,fear), ..., (pair,surprise)]에 대해 각각의 precision, recall 구한다.
    - 구한 각 클래스의 precision, recall을 평균내어 전체 precision, recall 구한다 (macro average)
    - 전체 precision, recall을 통해 Pair-Emotion F1 구한다.
    '''
    # 불균형이 심하므로 micro
    micro_precision = sum(pred_num_acc_list_all) / sum(pred_num_precision_denominator_dict)
    micro_recall = sum(pred_num_acc_list_all) / sum(pred_num_recall_denominator_dict)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0
    
    p_emo_cau = micro_precision
    r_emo_cau = micro_recall
    f1_emo_cau = micro_f1
    
    # p_emo_cau /= len(idx_list)
    # r_emo_cau /= len(idx_list)
    # f1_emo_cau = 2 * p_emo_cau * r_emo_cau / (p_emo_cau + r_emo_cau) if p_emo_cau + r_emo_cau != 0 else 0
    
    return emo_report_str, emo_metrics, p_cau, r_cau, f1_cau, p_emo_cau, r_emo_cau, f1_emo_cau

def argmax_prediction(pred_y, true_y):
    # pred_argmax = torch.argmax(pred_y, dim=1).cpu()
    pred_y = pred_y.cpu()
    true_y = true_y.cpu()
    # return pred_argmax, true_y
    return pred_y, true_y

def threshold_prediction(pred_y, true_y):
    pred_y = pred_y > 0.5
    return pred_y, true_y

def metrics_report(pred_y, true_y, label, get_dict=False, multilabel=False):
    true_y = true_y.view(-1)
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = list(label[available_label])
    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)
