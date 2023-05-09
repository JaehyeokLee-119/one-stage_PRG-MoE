import json
import torch
from transformers import BertTokenizer, AutoTokenizer

def get_data(data_file, device, max_seq_len, encoder_name, contain_context=False):
    '''
    encoder_name: bert-base-uncased, roberta-large, etc.
    '''
    f = open(data_file)
    data = json.load(f)
    f.close()

    emotion_label_policy = {'angry': 0, 'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
        'sad': 4, 'sadness': 4, 'frustrated': 4,
        'surprise': 5, 'surprised': 5, 
        'neutral': 6}

    cause_label_policy = {'no-context':0, 'inter-personal':1, 'self-contagion':2, 'latent':3}

    if contain_context:
        preprocessed_utterance, max_doc_len, max_seq_len = load_utterance_with_context(data_file, device, max_seq_len, encoder_name)
    else:
        preprocessed_utterance, max_doc_len, max_seq_len = load_utterance(data_file, device, max_seq_len, encoder_name)


    doc_speaker, doc_emotion_label, doc_pair_cause_label, doc_pair_binary_cause_label = [list() for _ in range(4)]

    for doc_id, content in data.items():
        speaker, emotion_label, corresponding_cause_turn, corresponding_cause_span, corresponding_cause_label = [list() for _ in range(5)]
        content = content[0]

        pair_cause_label = torch.zeros((int(max_doc_len * (max_doc_len + 1) / 2), 4), dtype=torch.long) 
        pair_binary_cause_label = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2), dtype=torch.long)

        for turn_data in content:
            speaker.append(0 if turn_data["speaker"] == "A" else 1)

            emotion_label.append(emotion_label_policy[turn_data["emotion"]])

            corresponding_cause_label_by_turn = list()
            if "expanded emotion cause evidence" in turn_data.keys():

                corresponding_cause_per_turn = [_ - 1 if type(_) != str else -1 for _ in turn_data["expanded emotion cause evidence"]]
                corresponding_cause_turn.append(corresponding_cause_per_turn)

                for _ in corresponding_cause_per_turn: 
                    if _ == -1:
                        corresponding_cause_label_by_turn.append(cause_label_policy["latent"])
                    elif _ + 1 == turn_data["turn"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["no-context"])
                    elif content[_]["speaker"] == turn_data["speaker"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["self-contagion"])
                    elif content[_]["speaker"] != turn_data["speaker"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["inter-personal"])
                        
            corresponding_cause_label.append(corresponding_cause_label_by_turn)

        for idx, corresponding_cause_per_turn in enumerate(corresponding_cause_label):
            pair_idx = int(idx * (idx + 1) / 2)

            if corresponding_cause_per_turn:
                for cause_turn, cause in zip(content[idx]["expanded emotion cause evidence"], corresponding_cause_per_turn):
                    if type(cause_turn) == str:
                        continue
                    
                    cause_idx = int(cause_turn) - 1
                    pair_cause_label[pair_idx + cause_idx][cause] = 1
                    pair_binary_cause_label[pair_idx + cause_idx] = 1
        
        pair_cause_label[(torch.sum(pair_cause_label, dim=1) == False).nonzero(as_tuple=True)[0], 3] = 1

        doc_speaker.append(speaker)
        doc_emotion_label.append(emotion_label)
        doc_pair_cause_label.append(pair_cause_label)
        doc_pair_binary_cause_label.append(pair_binary_cause_label)
        
    out_speaker, out_emotion_label = [list() for _ in range(2)]
    out_pair_cause_label, out_pair_binary_cause_label = torch.stack(doc_pair_cause_label), torch.stack(doc_pair_binary_cause_label)

    for speaker, emotion_label in zip(doc_speaker, doc_emotion_label):
        speaker_t = torch.zeros(max_doc_len, dtype=torch.long)
        speaker_t[:len(speaker)] = torch.tensor(speaker)

        emotion_label_t = torch.zeros(max_doc_len, dtype=torch.long)
        emotion_label_t[:len(speaker)] = torch.tensor(emotion_label)

        out_speaker.append(speaker_t); out_emotion_label.append(emotion_label_t)

    out_speaker, out_emotion_label = torch.stack(out_speaker).type(torch.FloatTensor), torch.stack(out_emotion_label)

    # return preprocessed_utterance, out_speaker.to(device), out_emotion_label.to(device), out_pair_cause_label.to(device), out_pair_binary_cause_label.to(device)
    return preprocessed_utterance, out_speaker, out_emotion_label, out_pair_cause_label, out_pair_binary_cause_label

def load_utterance(data_file, device, max_seq_len, encoder_name):
    f = open(data_file)
    data = json.load(f)
    f.close()

    tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)

    max_seq_len = max_seq_len
    max_doc_len = 0

    doc_utterance = list()
    for doc_id, content in data.items():
        utterance = list()
        content = content[0]
        max_doc_len = max(len(content), max_doc_len)
            
        for turn_data in content:
            utterance.append(tokenizer_(turn_data["utterance"], padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))
            
        doc_utterance.append(utterance)
        
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]

    for utterance_t in doc_utterance:
        padding_sequence = tokenizer_('', padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt")

        padding_sequence_t = [padding_sequence for _ in range(max_doc_len - len(utterance_t))]

        utterance_t = utterance_t + padding_sequence_t # shape: (max_doc_len, max_seq_len)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
        
        for _ in utterance_t:
            if ('token_type_ids' in _.keys()):
                utterance_input_ids_t.append(_['input_ids'])
                utterance_attention_mask_t.append(_['attention_mask'])
                utterance_token_type_ids_t.append(_['token_type_ids'])
            else: #RoBERTa 류는 token_type_ids 가 없음
                utterance_input_ids_t.append(_['input_ids'])
                utterance_attention_mask_t.append(_['attention_mask'])
                utterance_token_type_ids_t.append(torch.zeros(_['input_ids'].shape).to(torch.int)) # 그 자리에 크기만큼 0으로 채운 텐서 넣음

        utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
        utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
        utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)

        out_utterance_input_ids.append(utterance_input_ids_t)
        out_utterance_attention_mask.append(utterance_attention_mask_t)
        out_utterance_token_type_ids.append(utterance_token_type_ids_t)


    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    return (out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids), max_doc_len, max_seq_len

def load_utterance_with_context(data_file, device, max_seq_len, encoder_name):
    def make_context(utterance_list, speaker_list, start_t, end_t, max_seq_len):
        # context = "[SEP]".join(utterance_list[start_t:end_t])
        # Original: context = " ".join(utterance_list[start_t:end_t])
        # 1st(context를 [SEP]로 분리): context = "[SEP]".join(utterance_list[start_t:end_t])
        # 2nd(context 순서를 뒤집어 최신이 앞에 오게): context = "[SEP]".join(utterance_list[start_t:end_t][::-1])
        # 3rd(화자 정보를 추가: [CLS] A said [SEP] 내용 [SEP] B said [SEP] 이전 발화 ...): 
        # 4th(화자 정보를 추가: [CLS] [Speaker A] 내용 [/Speaker A] [SEP] [Speaker B] 내용 [/Speaker B] 이전 발화 ...):
        # 5th(화자 정보를 추가: [CLS] [Speaker A] 내용 [SEP] [Speaker B] 내용 [SEP] [Speaker A] 내용 [SEP] ...)
        
        context = " ".join(utterance_list[start_t:end_t])
        # # 아래 3줄은 3rd, 4th, 5th를 위한 코드
        # context = ""
        # for utterance, speaker in zip(utterance_list[start_t:end_t], speaker_list[start_t:end_t]):
        #     context += f'[Speaker {speaker}] {utterance} [SEP]'
            
        if start_t > end_t:
            return ""

        if len(context.split()) + len(utterance_list[end_t].split()) > max_seq_len:
            context = make_context(utterance_list=utterance_list, speaker_list=speaker_list, start_t=start_t+1, end_t=end_t, max_seq_len=max_seq_len)
        else:
            return context
        
        return context
    
    f = open(data_file)
    data = json.load(f)
    f.close()

    tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)
    
    tokens = ["[Speaker A]", "[Speaker B]", "[/Speaker A]", "[/Speaker B]"]
    tokenizer_.add_tokens(tokens, special_tokens=True)

    max_seq_len = max_seq_len
    max_doc_len = 0

    doc_utterance = list()

    for doc_id, content in data.items():
        single_utterances = list()
        single_utterances_speaker = list()
        utterance = list()
        content = content[0]
        max_doc_len = max(len(content), max_doc_len)

        for turn_data in content:
            single_utterances.append(turn_data["utterance"])
            single_utterances_speaker.append(turn_data["speaker"])

        for end_t in range(len(single_utterances)):
            context = make_context(utterance_list=single_utterances, speaker_list=single_utterances_speaker, start_t=0, end_t=end_t, max_seq_len=max_seq_len)
            
            # Original
            utterance.append(tokenizer_(single_utterances[end_t], context, padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))
            
            # # Speaker 정보 추가
            # spk = single_utterances_speaker[end_t]
            # speaker_plus_utterance = f'[Speaker {spk}] {single_utterances[end_t]}' # 감싸거나 말거나
            # utterance.append(tokenizer_(speaker_plus_utterance, context, padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))
        
        doc_utterance.append(utterance)
        
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]

    for utterance_t in doc_utterance:
        padding_sequence = tokenizer_('', padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt")

        padding_sequence_t = [padding_sequence for _ in range(max_doc_len - len(utterance_t))]

        utterance_t = utterance_t + padding_sequence_t
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
        
        for _ in utterance_t:
            if ('token_type_ids' in _.keys()):
                utterance_input_ids_t.append(_['input_ids'])
                utterance_attention_mask_t.append(_['attention_mask'])
                utterance_token_type_ids_t.append(_['token_type_ids'])
            else: #RoBERTa 류는 token_type_ids 가 없음
                utterance_input_ids_t.append(_['input_ids'])
                utterance_attention_mask_t.append(_['attention_mask'])
                utterance_token_type_ids_t.append(torch.zeros(_['input_ids'].shape).to(torch.int)) # 그 자리에 크기만큼 0으로 채운 텐서 넣음

        utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
        utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
        utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)

        out_utterance_input_ids.append(utterance_input_ids_t)
        out_utterance_attention_mask.append(utterance_attention_mask_t)
        out_utterance_token_type_ids.append(utterance_token_type_ids_t)

    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    # return (out_utterance_input_ids.to(device), out_utterance_attention_mask.to(device), out_utterance_token_type_ids.to(device)), max_doc_len, max_seq_len
    
    # device로 보내는 옵션을 해제 ( CUDA error: initialization error 때문에 )
    return (out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids), max_doc_len, max_seq_len

def tokenize_conversation(conversation, device, max_seq_len, encoder_name):
    tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)

    max_seq_len = max_seq_len
    max_doc_len = 0

    doc_utterance, speaker = [list() for _ in range(2)]
    for doc_id, content in conversation.items():
        utterance = list()
        content = content[0]
            
        for turn_data in content:
            utterance.append(tokenizer_(turn_data["utterance"], padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))

            speaker.append(0 if turn_data["speaker"] == "A" else 1)
            
        doc_utterance.append(utterance)
        
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]

    for utterance_t in doc_utterance:
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
        for _ in utterance_t:
            utterance_input_ids_t.append(_['input_ids'])
            utterance_attention_mask_t.append(_['attention_mask'])
            utterance_token_type_ids_t.append(_['token_type_ids'])

        utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
        utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
        utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)

        out_utterance_input_ids.append(utterance_input_ids_t)
        out_utterance_attention_mask.append(utterance_attention_mask_t)
        out_utterance_token_type_ids.append(utterance_token_type_ids_t)

    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    return out_utterance_input_ids.to(device), out_utterance_attention_mask.to(device), out_utterance_token_type_ids.to(device), torch.Tensor(speaker).to(device)


def get_pad_idx(utterance_input_ids_batch, encoder_name):
    # 하나의 batch (dialog) 속에서, 각 token들이 pad인지 아닌지를 담은 idx를 구한다 (유효한 토큰은 NOT_ZERO, pad는 0)
    # 입력: batch 개의 dialog [5, 28, 75]
    # 출력: batch 개의 dialog 속 유효성을 모두 이어서 담은 idx [140] (5*28)
    batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
    
    if 'bert-base' in encoder_name:
        # BERT 류인 경우
        check_pad_idx = torch.sum(
            utterance_input_ids_batch.view(-1, max_seq_len)[:, 2:], dim=1).cpu()
    else:
        # RoBERTa 류인 경우
        tmp = utterance_input_ids_batch.view(-1, max_seq_len)[:, 2:]-1
        check_pad_idx = torch.sum(tmp, dim=1).cpu()

    return check_pad_idx

def get_pair_pad_idx(utterance_input_ids_batch, encoder_name, window_constraint=3, emotion_pred=None):
    # input utterance batch에 대한 pad idx를 구한다 (유효한 pair는 1, pad는 0)
    batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
    
    check_pad_idx = get_pad_idx(utterance_input_ids_batch, encoder_name) # dialog batch에서 padding은 0으로, 유효한 토큰은 그냥 숫자가 있는 idx [140]

    if emotion_pred is not None:
        emotion_pred = torch.argmax(emotion_pred, dim=1) # 7차원 분류결과([140, 7])를 1차원으로 줄임(가장 큰 값의 index로, [140])
        
    check_pair_window_idx = list()
    
    if (emotion_pred is not None):
        for batch, emo_pred in zip(check_pad_idx.view(-1, max_doc_len), emotion_pred.view(batch_size,-1)): # check_pad_idx [140] -> [5,28]해서, batch=[28]
            pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2)) # pair window를 넣을 공간을 만든다 (28*29/2=406)
            for end_t in range(1, len(batch.nonzero()) + 1): # 각 dialog속의 '진짜 문장 길이'만큼 반복
                if emotion_pred is not None and emo_pred[end_t - 1] == 6: # emotion이 6(중립)인 경우는 제외
                    continue
                
                # non-neutral인 경우, window_constraint만큼의 window를 만들어서 1로 채운다
                pair_window_idx[max(0, int((end_t-1)*end_t/2), int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1 
                
            check_pair_window_idx.append(pair_window_idx)
    else:
        for batch in check_pad_idx.view(-1, max_doc_len): # check_pad_idx [140] -> [5,28]해서, batch=[28]
            pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2)) # pair window를 넣을 공간을 만든다 (28*29/2=406)
            for end_t in range(1, len(batch.nonzero()) + 1): # 각 dialog속의 '진짜 문장 길이'만큼 반복
                # non-neutral인 경우, window_constraint만큼의 window를 만들어서 1로 채운다
                pair_window_idx[max(0, int((end_t-1)*end_t/2), int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1 
                
            check_pair_window_idx.append(pair_window_idx)
            
    return torch.stack(check_pair_window_idx)