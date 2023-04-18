import pandas as pd
import random

import numpy as np

## Torch Modules
import torch
from torch.utils.data import Dataset


## PyTorch Transformer
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def prepare_features(question, answer, max_qa_seq_length=40, padding=True):
    # Tokenizne Input
    indexed_tokens = tokenizer.encode(question, answer, max_length=max_qa_seq_length, add_special_tokens=True)

    # Input Mask
    input_mask = [1] * len(indexed_tokens)
    # Pad to max_qa_seq_length using padding special token
    if padding:
        while len(indexed_tokens) < max_qa_seq_length:
            indexed_tokens.append(tokenizer.pad_token_id)
            input_mask.append(0)
    return torch.tensor(indexed_tokens).unsqueeze(dim=0), input_mask

# changes for sentiment classification expts [Author: Nikhil Madaan]
class SocialIQ_QA_Sentiment_Classification(Dataset):
    def __init__(self, answers, labels):
        self.answers = answers
        self.labels = labels

    def get_features(self, answer, max_qa_seq_length=40, padding=True):
        indexed_tokens = tokenizer.encode(answer, max_length=max_qa_seq_length, add_special_tokens=True)

        if padding:
            while len(indexed_tokens) < max_qa_seq_length:
                indexed_tokens.append(tokenizer.pad_token_id)

        return torch.tensor(indexed_tokens).unsqueeze(dim=0)

    def __getitem__(self, index):
        answer = self.get_features(self.answers[index])
        label = self.labels[index]
        return answer, label

    def __len__(self):
        return len(self.labels)
    
# changes for sentiment classification expts [Author: Claire Ko]
class SocialIQ_QA_Perplexity_Classification(Dataset):
    def __init__(self, answers, labels):
        self.answers = answers
        self.labels = labels

    def get_features(self, answer, max_qa_seq_length=40, padding=True):
        indexed_tokens = tokenizer.encode(answer, max_length=max_qa_seq_length, add_special_tokens=True)

        if padding:
            while len(indexed_tokens) < max_qa_seq_length:
                indexed_tokens.append(tokenizer.pad_token_id)

        return torch.tensor(indexed_tokens).unsqueeze(dim=0)

    def __getitem__(self, index):
        answer = self.get_features(self.answers[index])
        label = self.labels[index]
        return answer, label

    def __len__(self):
        return len(self.labels)

class SocialIQ_QA_Relevance_Classification(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.q[index]
        a = self.data.a[index]
        label = self.data.relevance[index]
        X_a, _ = prepare_features(question, a)
        return X_a, label, question, a

    def __len__(self):
        return self.len

class SocialIQ_QA_Only_A2(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.question[index]
        a = random.sample(population=self.data.a_list[index], k=1)[0]
        i = random.sample(population=self.data.i_list[index], k=1)[0]
        X_a, _ = prepare_features(question, a)
        X_i, _ = prepare_features(question, i)
        label = int(0) # For A2, the correct answer is always at index 0
        return X_a, X_i, label

    def __len__(self):
        return self.len

class MovieQA_QA_Only_A5_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.question[index]
        answer_idx = self.data.correct_index[index] if not np.isnan(self.data.correct_index[index]) else 0
        a = self.data.answers[index][answer_idx]

        incorrect_idx_list = [0, 1, 2, 3, 4]
        incorrect_idx_list.remove(answer_idx)

        incorrect_answer_list = []
        for incorrect_idx in incorrect_idx_list:
            incorrect_answer_list.append(self.data.answers[index][incorrect_idx])

        i_1, i_2, i_3, i_4 = incorrect_answer_list

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        X_i_4, _ = prepare_features(question, i_4)
        label = int(0) # For A4, the correct answer is always at index 0

        return X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, self.data.imdb_key[index], self.data.qid[index]

    def __len__(self):
        return self.len

class MovieQA_Answer_Only_A5_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = ''
        answer_idx = self.data.correct_index[index] if not np.isnan(self.data.correct_index[index]) else 0
        a = self.data.answers[index][answer_idx]

        incorrect_idx_list = [0, 1, 2, 3, 4]
        incorrect_idx_list.remove(answer_idx)

        incorrect_answer_list = []
        for incorrect_idx in incorrect_idx_list:
            incorrect_answer_list.append(self.data.answers[index][incorrect_idx])

        i_1, i_2, i_3, i_4 = incorrect_answer_list

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        X_i_4, _ = prepare_features(question, i_4)
        label = int(0) # For A4, the correct answer is always at index 0

        return X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, self.data.imdb_key[index], self.data.qid[index]

    def __len__(self):
        return self.len

class TVQA_QA_Only_A5_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.q[index]
        answer_idx = self.data.answer_idx[index]
        a = self.data['a' + str(answer_idx)][index]

        incorrect_idx_list = [0, 1, 2, 3, 4]
        incorrect_idx_list.remove(answer_idx)

        incorrect_answer_list = []
        for incorrect_idx in incorrect_idx_list:
            incorrect_answer_list.append(self.data['a' + str(incorrect_idx)][index])

        i_1, i_2, i_3, i_4 = incorrect_answer_list

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        X_i_4, _ = prepare_features(question, i_4)
        label = int(0) # For A4, the correct answer is always at index 0

        return X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, self.data.vid_name[index], self.data.qid[index]

    def __len__(self):
        return self.len

class TVQA_Answer_Only_A5_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = ''
        answer_idx = self.data.answer_idx[index]
        a = self.data['a' + str(answer_idx)][index]

        incorrect_idx_list = [0, 1, 2, 3, 4]
        incorrect_idx_list.remove(answer_idx)

        incorrect_answer_list = []
        for incorrect_idx in incorrect_idx_list:
            incorrect_answer_list.append(self.data['a' + str(incorrect_idx)][index])

        i_1, i_2, i_3, i_4 = incorrect_answer_list

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        X_i_4, _ = prepare_features(question, i_4)
        label = int(0) # For A4, the correct answer is always at index 0

        return X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, self.data.vid_name[index], self.data.qid[index]

    def __len__(self):
        return self.len

class TVQA_QA_Only_A5_NameThatAnnotator_Detail_Dataset(Dataset):
    def __init__(self, dataframe, annotator_map):
        self.len = len(dataframe)
        self.data = dataframe
        self.annotator_map = annotator_map

    def __getitem__(self, index):
        question = self.data.q[index]
        answer_idx = self.data.answer_idx[index]
        a = self.data['a' + str(answer_idx)][index]

        incorrect_idx_list = [0, 1, 2, 3, 4]
        incorrect_idx_list.remove(answer_idx)

        incorrect_answer_list = []
        for incorrect_idx in incorrect_idx_list:
            incorrect_answer_list.append(self.data['a' + str(incorrect_idx)][index])

        i_1, i_2, i_3, i_4 = incorrect_answer_list

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        X_i_4, _ = prepare_features(question, i_4)

        label = self.annotator_map[self.data.annotator[index]]

        return X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, self.data.vid_name[index], self.data.qid[index]

    def __len__(self):
        return self.len


class SocialIQ_QA_Only_A2_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.question[index]
        a = random.sample(population=self.data.a_list[index], k=1)[0]
        i_1 = random.sample(population=self.data.i_list[index], k=1)[0]

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        label = int(0) # For A4, the correct answer is always at index 0
        if 'annotator' in self.data.columns:
            return X_a, X_i_1, label, question, a, i_1, self.data.id[index], self.data.annotator[index]
        else:
            return X_a, X_i_1, label, question, a, i_1, self.data.id[index]

    def __len__(self):
        return self.len

class SocialIQ_Answer_Only_A2_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = ''  # answer only, so pass in empty question
        a = random.sample(population=self.data.a_list[index], k=1)[0]
        i_1 = random.sample(population=self.data.i_list[index], k=1)[0]

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        label = int(0) # For A4, the correct answer is always at index 0
        if 'annotator' in self.data.columns:
            return X_a, X_i_1, label, question, a, i_1, self.data.id[index], self.data.annotator[index]
        else:
            return X_a, X_i_1, label, question, a, i_1, self.data.id[index]

    def __len__(self):
        return self.len

class SocialIQ_QA_Only_A2_NameThatAnnotator_Detail_Dataset(Dataset):
    def __init__(self, dataframe, annotator_map):
        self.len = len(dataframe)
        self.data = dataframe
        self.annotator_map = annotator_map

    def __getitem__(self, index):
        question = self.data.question[index]
        a = self.data.a_list[index][0]
        i_1 = self.data.i_list[index][0]

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        label = self.annotator_map[self.data.annotator[index]]
        if 'annotator' in self.data.columns:
            return X_a, X_i_1, label, question, a, i_1, self.data.id[index], self.data.annotator[index]
        else:
            return X_a, X_i_1, label, question, a, i_1, self.data.id[index]

    def __len__(self):
        return self.len

class SocialIQ_Permute_QA_Only_A2_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        video_id = self.data.video_id[index]
        qai_id = self.data.qai_id[index]
        question = self.data.question[index]
        q_annotator = self.data.q_annotator[index]
        a = self.data.a[index]
        a_annotator = self.data.a_annotator[index]
        i_1 = self.data.i[index]
        i_1_annotator = self.data.i_annotator[index]

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        label = int(0) # For A4, the correct answer is always at index 0

        return X_a, X_i_1, label, question, a, i_1, q_annotator, a_annotator, i_1_annotator, qai_id, video_id

    def __len__(self):
        return self.len

class SocialIQ_Permute_QA_Only_A2_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        video_id = self.data.video_id[index]
        qai_id = self.data.qai_id[index]
        question = self.data.question[index]
        q_annotator = self.data.q_annotator[index]
        a = self.data.a[index]
        a_annotator = self.data.a_annotator[index]
        i_1 = self.data.i[index]
        i_1_annotator = self.data.i_annotator[index]

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        label = int(0)  # For A4, the correct answer is always at index 0

        return X_a, X_i_1, label, question, a, i_1, q_annotator, a_annotator, i_1_annotator, qai_id, video_id

    def __len__(self):
        return self.len

class SocialIQ_Permute_Answer_Only_A2_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        video_id = self.data.video_id[index]
        qai_id = self.data.qai_id[index]
        question = ''  # answer only, so pass in empty question
        q_annotator = self.data.q_annotator[index]
        a = self.data.a[index]
        a_annotator = self.data.a_annotator[index]
        i_1 = self.data.i[index]
        i_1_annotator = self.data.i_annotator[index]

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        label = int(0) # For A4, the correct answer is always at index 0

        return X_a, X_i_1, label, question, a, i_1, q_annotator, a_annotator, i_1_annotator, qai_id, video_id

    def __len__(self):
        return self.len

class SocialIQ_QA_Only_A4_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.question[index]
        a = random.sample(population=self.data.a_list[index], k=1)[0]
        try:
            i_1, i_2, i_3 = np.random.choice(self.data.i_list[index], size=3, replace=False)  # Sample three incorrect answers
        except ValueError:
            i_1, i_2, i_3 = np.random.choice(self.data.i_list[index], size=3, replace=True)  # if there is not enough incorrect answers, sample with replacement

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        label = int(0) # For A4, the correct answer is always at index 0
        return X_a, X_i_1, X_i_2, X_i_3, label, question, a, i_1, i_2, i_3, self.data.id[index]

    def __len__(self):
        return self.len


class SocialIQ_QA_Only_A5_Detail_Dataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        question = self.data.question[index]
        a = random.sample(population=self.data.a_list[index], k=1)[0]
        try:
            i_1, i_2, i_3, i_4 = np.random.choice(self.data.i_list[index], size=4, replace=False)  # Sample three incorrect answers
        except ValueError:
            i_1, i_2, i_3, i_4 = np.random.choice(self.data.i_list[index], size=4, replace=True)  # if there is not enough incorrect answers, sample with replacement

        X_a, _ = prepare_features(question, a)
        X_i_1, _ = prepare_features(question, i_1)
        X_i_2, _ = prepare_features(question, i_2)
        X_i_3, _ = prepare_features(question, i_3)
        X_i_4, _ = prepare_features(question, i_4)
        label = int(0) # For A4, the correct answer is always at index 0
        return X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, self.data.id[index]

    def __len__(self):
        return self.len
