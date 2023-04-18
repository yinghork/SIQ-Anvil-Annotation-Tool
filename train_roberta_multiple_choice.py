# command example:
# python train_roberta_multiple_choice.py --dataset_path /work/jianing3/lm-bias/data --dataset_name movieqa \
# --output_dir /results/jianing3/lm-bias/checkpoint/movieqa/freeze_pickup_from_tvqa_lr_1e-6_bs_3_unfreeze \
# --learning_rate 1e-6 --batch_size 3 \
# --resume_from /results/jianing3/lm-bias/checkpoint/tvqa/lr_1e-6_bs_3_unfreeze/roberta_state_dict_16_0.pth

import pandas as pd
from tqdm import tqdm
import time
import os
import argparse
import json
import pickle
import sys


## Torch Modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


## PyTorch Transformer
from transformers import RobertaConfig

from transformers import AdamW

from models.roberta_4mlp import RobertaForSocialIQClassification
import util
from datasets import MovieQA_QA_Only_A5_Detail_Dataset, TVQA_QA_Only_A5_Detail_Dataset, \
    SocialIQ_QA_Only_A4_Detail_Dataset, SocialIQ_QA_Only_A5_Detail_Dataset, SocialIQ_QA_Only_A2_Detail_Dataset, \
    SocialIQ_QA_Only_A2_NameThatAnnotator_Detail_Dataset, TVQA_QA_Only_A5_NameThatAnnotator_Detail_Dataset, \
    SocialIQ_Answer_Only_A2_Detail_Dataset, TVQA_Answer_Only_A5_Detail_Dataset, MovieQA_Answer_Only_A5_Detail_Dataset, \
    SocialIQ_Permute_QA_Only_A2_Detail_Dataset, SocialIQ_Permute_Answer_Only_A2_Detail_Dataset

log = None
import logging
logging.basicConfig(level=logging.ERROR)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="dataset path")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name: movieqa, tvqa, socialiq_a5, socialiq_a2, socialiq_a4")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir path")
    parser.add_argument("--resume_from", type=str, default=None, help="resume from this checkpoint file")
    parser.add_argument("--learning_rate", type=float, required=True, help="learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--freeze_lm", default=False, action='store_true',
                        help="whether to freeze language model parameters")
    parser.add_argument("--load_pretrained_lm", default=False, action='store_true',
                        help="whether to load pretrained language model parameters")
    parser.add_argument("--half_precision", default=False, action='store_true', help="whether to use half precision")
    parser.add_argument("--inference_only", default=False, action='store_true',
                        help="set this if we are not training but just evaluating")
    parser.add_argument("--do_test", default=False, action='store_true',
                        help="set this if we are evaluating on test set and want to output prediction into test_output.txt")
    parser.add_argument("--output_prediction_correctness", default=False, action='store_true',
                        help="set this if we want to output prediction correctness into prediction_correctness.json")
    parser.add_argument("--valid_subset_frac", type=float, default=1., help="fraction of validation set to use")
    parser.add_argument("--max_epochs", type=int, default=20, help="max epochs to train")
    parser.add_argument("--do_name_that_annotator", default=False, action='store_true', help="run the script for the Name That Annotator! task")
    parser.add_argument("--annotator_map_dict", type=str, required=('--do_name_that_annotator' in sys.argv), help="the path to the dict of annotator mapping, should be a json file")
    parser.add_argument("--do_answer_only", default=False, action='store_true',
                        help="set this if we are doint the answer only task")
    args = parser.parse_args()
    return args


class RobertaMultipleChoiceTask:
    def __init__(self, args):
        self.args = args

        self.create_output_dir()
        self.setup_logger()

        self.set_device()
        self.load_checkpoint()

        self.setup_dataset()

        self.setup_criterion()
        self.setup_model()
        self.setup_optimizer()


    def setup_logger(self):
        global log
        log = util.get_logging(self.args.output_dir, name=__name__)

        log.info(self.args)

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_output_dir(self):
        os.makedirs(self.args.output_dir, exist_ok=True)

    def load_checkpoint(self):
        self.checkpoint = None
        if self.args.resume_from != None:
            self.checkpoint = torch.load(self.args.resume_from)

    def setup_criterion(self):
        if self.args.dataset_name in ['socialiq_permute_a1']:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def setup_optimizer(self):
        self.optimizer = AdamW(params=self.model.parameters(),
                               lr=self.args.learning_rate, weight_decay=1e-2, correct_bias=False)

    def setup_dataset(self):
        dataset_path = self.args.dataset_path
        if self.args.dataset_name == 'movieqa':
            df_train_dataset = pd.read_json(os.path.join(dataset_path, 'movieqa_train.json'))
            if self.args.do_test:
                df_valid_subset = pd.read_json(os.path.join(dataset_path, 'movieqa_test.json'))
            else:
                df_valid_subset = pd.read_json(os.path.join(dataset_path, 'movieqa_valid.json'))
        elif self.args.dataset_name == 'tvqa':
            df_train_dataset = pd.read_json(os.path.join(dataset_path, 'tvqa_train.json'))
            df_valid_subset = pd.read_json(os.path.join(dataset_path, 'tvqa_valid.json'))
        elif self.args.dataset_name in ['socialiq_a5', 'socialiq_a2', 'socialiq_a4' ]:
            df_train_dataset = pd.read_json(os.path.join(dataset_path, 'socialiq_train.json'))
            df_valid_subset = pd.read_json(os.path.join(dataset_path, 'socialiq_valid.json'))
        elif self.args.dataset_name in ['socialiq_permute_a2', 'socialiq_permute_a1']:
            df_train_dataset = pd.read_json(os.path.join(dataset_path, 'socialiq_permute_train.json'))
            df_valid_subset = pd.read_json(os.path.join(dataset_path, 'socialiq_permute_valid.json'))
        else:
            raise ValueError('dataset_name must be either movieqa, tvqa or socialiq')

        df_valid_subset = df_valid_subset.sample(frac=self.args.valid_subset_frac, random_state=2020).reset_index(
            drop=True)  # use a fraction of validation set

        # print("FULL Dataset: {}".format(df_dataset.shape))
        print("Dataset name: {}".format(self.args.dataset_name))
        print("TRAIN dataset size: {}".format(df_train_dataset.shape))
        print("VALID dataset size: {}".format(df_valid_subset.shape))

        if self.args.dataset_name == 'movieqa':
            if self.args.do_answer_only:
                self.training_set = MovieQA_Answer_Only_A5_Detail_Dataset(df_train_dataset)
                self.valid_subset = MovieQA_Answer_Only_A5_Detail_Dataset(df_valid_subset)
            else:
                self.training_set = MovieQA_QA_Only_A5_Detail_Dataset(df_train_dataset)
                self.valid_subset = MovieQA_QA_Only_A5_Detail_Dataset(df_valid_subset)
        elif self.args.dataset_name == 'tvqa':
            if self.args.do_name_that_annotator:
                with open(self.args.annotator_map_dict, 'r') as annotator_map_file:
                    annotator_map_dict = json.load(annotator_map_file)
                self.training_set = TVQA_QA_Only_A5_NameThatAnnotator_Detail_Dataset(df_train_dataset, annotator_map_dict)
                self.valid_subset = TVQA_QA_Only_A5_NameThatAnnotator_Detail_Dataset(df_valid_subset, annotator_map_dict)
            elif self.args.do_answer_only:
                self.training_set = TVQA_Answer_Only_A5_Detail_Dataset(df_train_dataset)
                self.valid_subset = TVQA_Answer_Only_A5_Detail_Dataset(df_valid_subset)
            else:
                self.training_set = TVQA_QA_Only_A5_Detail_Dataset(df_train_dataset)
                self.valid_subset = TVQA_QA_Only_A5_Detail_Dataset(df_valid_subset)
        elif self.args.dataset_name == 'socialiq_a5':
            self.training_set = SocialIQ_QA_Only_A5_Detail_Dataset(df_train_dataset)
            self.valid_subset = SocialIQ_QA_Only_A5_Detail_Dataset(df_valid_subset)
        elif self.args.dataset_name == 'socialiq_a2':
            if self.args.do_name_that_annotator:
                with open(self.args.annotator_map_dict, 'r') as annotator_map_file:
                    annotator_map_dict = json.load(annotator_map_file)
                self.training_set = SocialIQ_QA_Only_A2_NameThatAnnotator_Detail_Dataset(df_train_dataset, annotator_map_dict)
                self.valid_subset = SocialIQ_QA_Only_A2_NameThatAnnotator_Detail_Dataset(df_valid_subset, annotator_map_dict)
            elif self.args.do_answer_only:
                self.training_set = SocialIQ_Answer_Only_A2_Detail_Dataset(df_train_dataset)
                self.valid_subset = SocialIQ_Answer_Only_A2_Detail_Dataset(df_valid_subset)
            else:
                self.training_set = SocialIQ_QA_Only_A2_Detail_Dataset(df_train_dataset)
                self.valid_subset = SocialIQ_QA_Only_A2_Detail_Dataset(df_valid_subset)
        elif self.args.dataset_name == 'socialiq_a4':
            self.training_set = SocialIQ_QA_Only_A4_Detail_Dataset(df_train_dataset)
            self.valid_subset = SocialIQ_QA_Only_A4_Detail_Dataset(df_valid_subset)
        elif self.args.dataset_name in ['socialiq_permute_a2', 'socialiq_permute_a1']:
            if self.args.do_answer_only:
                self.training_set = SocialIQ_Permute_Answer_Only_A2_Detail_Dataset(df_train_dataset)
                self.valid_subset = SocialIQ_Permute_Answer_Only_A2_Detail_Dataset(df_valid_subset)
            else:
                self.training_set = SocialIQ_Permute_QA_Only_A2_Detail_Dataset(df_train_dataset)
                self.valid_subset = SocialIQ_Permute_QA_Only_A2_Detail_Dataset(df_valid_subset)
        else:
            raise ValueError('dataset_name must be either movieqa, tvqa or socialiq')

        # Sanity check: no ID overlapping between train and valid
        # Sanity check: 97 out of 924 questions in valid set overlaps with questions in train set
        # pd.set_option('display.expand_frame_repr', False)
        # print(len(set(df_train_dataset.id.tolist()).intersection(set(df_valid_subset.id.tolist()))))
        # print(df_valid_subset[df_valid_subset['question'].isin(df_train_dataset.question.tolist())])

        # Parameters
        batch_size = self.args.batch_size
        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'drop_last': False,
                  }
                  # 'num_workers': 4}

        self.training_loader = DataLoader(self.training_set, num_workers=4, **params)
        self.testing_loader = DataLoader(self.valid_subset, num_workers=0, **params)
        self.training_loader_iter = iter(self.training_loader)


    def setup_model(self):
        config = RobertaConfig.from_pretrained('roberta-large')
        if self.args.do_name_that_annotator:
            config.num_labels = 10
        else:
            config.num_labels = 1
            
        config.output_hidden_states=True
        #claire - get hidden states for last layer embeddings

        # load pre-trained
        if self.checkpoint != None:
            self.model = RobertaForSocialIQClassification(config)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            if self.args.load_pretrained_lm:
                self.model = RobertaForSocialIQClassification.from_pretrained('roberta-large', config=config)
            else:
                self.model = RobertaForSocialIQClassification(config)

        # Freeze RoBERTa
        if self.args.freeze_lm:
            for param in self.model.roberta.parameters():
                param.requires_grad = False

        if self.args.half_precision:
            self.model.half()

        self.model = self.model.to(self.device)

    def train_step(self, total_train_time, train_time_iteration_counter):
        self.model = self.model.train()
        start_time = time.time()

        try:
            if self.args.dataset_name in ['movieqa', 'tvqa']:
                X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, vid_name, qid = next(self.training_loader_iter)
            elif self.args.dataset_name == 'socialiq_a5':
                X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, vid_name = next(self.training_loader_iter)
            elif self.args.dataset_name == 'socialiq_a2':
                value_to_unpack = next(self.training_loader_iter)
                if len(value_to_unpack) == 7:  # does not have annotator field
                    X_a, X_i_1, label, question, a, i_1, vid_name = value_to_unpack
                elif len(value_to_unpack) == 8:  # has annotator field
                    X_a, X_i_1, label, question, a, i_1, vid_name, annotator = value_to_unpack
            elif self.args.dataset_name == 'socialiq_a4':
                X_a, X_i_1, X_i_2, X_i_3, label, question, a, i_1, i_2, i_3, vid_name = next(self.training_loader_iter)
            elif self.args.dataset_name in ['socialiq_permute_a2', 'socialiq_permute_a1']:
                X_a, X_i_1, label, question, a, i_1, q_annotator, a_annotator, i_1_annotator, qai_id, video_id = next(self.training_loader_iter)
        except StopIteration:
            self.training_loader_iter = iter(self.training_loader)
            if self.args.dataset_name in ['movieqa', 'tvqa']:
                X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, vid_name, qid = next(self.training_loader_iter)
            elif self.args.dataset_name == 'socialiq_a5':
                X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, vid_name = next(self.training_loader_iter)
            elif self.args.dataset_name == 'socialiq_a2':
                value_to_unpack = next(self.training_loader_iter)
                if len(value_to_unpack) == 7:  # does not have annotator field
                    X_a, X_i_1, label, question, a, i_1, vid_name = value_to_unpack
                elif len(value_to_unpack) == 8:  # has annotator field
                    X_a, X_i_1, label, question, a, i_1, vid_name, annotator = value_to_unpack
            elif self.args.dataset_name == 'socialiq_a4':
                X_a, X_i_1, X_i_2, X_i_3, label, question, a, i_1, i_2, i_3, vid_name = next(self.training_loader_iter)
            elif self.args.dataset_name in ['socialiq_permute_a2', 'socialiq_permute_a1']:
                X_a, X_i_1, label, question, a, i_1, q_annotator, a_annotator, i_1_annotator, qai_id, video_id = next(self.training_loader_iter)

        X_a = X_a.squeeze(1)
        X_a = X_a.to(self.device)
        logit_a = self.model(X_a)[0]

        X_i_1 = X_i_1.squeeze(1)
        X_i_1 = X_i_1.to(self.device)
        logit_i_1 = self.model(X_i_1)[0]

        if self.args.dataset_name in ['movieqa', 'tvqa', 'socialiq_a4', 'socialiq_a5']:
            X_i_2 = X_i_2.squeeze(1)
            X_i_2 = X_i_2.to(self.device)
            logit_i_2 = self.model(X_i_2)[0]

            X_i_3 = X_i_3.squeeze(1)
            X_i_3 = X_i_3.to(self.device)
            logit_i_3 = self.model(X_i_3)[0]

            if self.args.dataset_name in ['movieqa', 'tvqa', 'socialiq_a5']:
                X_i_4 = X_i_4.squeeze(1)
                X_i_4 = X_i_4.to(self.device)
                logit_i_4 = self.model(X_i_4)[0]


        if self.args.do_name_that_annotator:
            if self.args.dataset_name in ['socialiq_a2']:
                logits = torch.cat((logit_a, logit_i_1), dim=0)
                label = label.repeat(2)
            elif self.args.dataset_name in ['socialiq_a4']:
                logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3), dim=0)
                label = label.repeat(4)
            elif self.args.dataset_name in ['socialiq_a5', 'tvqa', 'movieqa']:
                logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3, logit_i_4), dim=0)
                label = label.repeat(5)
        else:
            if self.args.dataset_name in ['socialiq_a2', 'socialiq_permute_a2']:
                logits = torch.cat((logit_a, logit_i_1), dim=1)
            elif self.args.dataset_name in ['socialiq_permute_a1']:
                logits = torch.cat((logit_a, logit_i_1), dim=0)
            elif self.args.dataset_name in ['socialiq_a4']:
                logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3), dim=1)
            elif self.args.dataset_name in ['socialiq_a5', 'tvqa', 'movieqa']:
                logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3, logit_i_4), dim=1)

        if self.args.dataset_name in ['socialiq_permute_a1']:
            label = torch.cat((torch.ones_like(logit_a), torch.zeros_like(logit_i_1)), dim=0).to(self.device)
        else:
            label = label.to(self.device)

        loss = self.criterion(logits, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_train_time += time.time() - start_time
        train_time_iteration_counter += self.args.batch_size
        return loss, total_train_time, train_time_iteration_counter

    def valid_step(self, epoch=-1, max_epochs=-1, i=0, train_loss=None, total_train_time=-1., train_time_iteration_counter=-1., iteration_per_epoch=-1):
        self.model = self.model.eval()
        correct = 0
        total = 0
        result_dict = {}
        prediction_correctness_dict = {}
        df_valid_dataset = pd.DataFrame()
        with torch.no_grad():
            for value_to_unpack in tqdm(self.testing_loader):
                if self.args.dataset_name in ['movieqa', 'tvqa']:
                    X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, vid_name, qid = value_to_unpack
                elif self.args.dataset_name == 'socialiq_a5':
                    X_a, X_i_1, X_i_2, X_i_3, X_i_4, label, question, a, i_1, i_2, i_3, i_4, vid_name = value_to_unpack
                elif self.args.dataset_name == 'socialiq_a2':
                    if len(value_to_unpack) == 7:  # does not have annotator field
                        X_a, X_i_1, label, question, a, i_1, vid_name = value_to_unpack
                    elif len(value_to_unpack) == 8:  # has annotator field
                        X_a, X_i_1, label, question, a, i_1, vid_name, annotator = value_to_unpack
                elif self.args.dataset_name == 'socialiq_a4':
                    X_a, X_i_1, X_i_2, X_i_3, label, question, a, i_1, i_2, i_3, vid_name = value_to_unpack
                elif self.args.dataset_name in ['socialiq_permute_a2', 'socialiq_permute_a1']:
                    X_a, X_i_1, label, question, a, i_1, q_annotator, a_annotator, i_1_annotator, qai_id, video_id = value_to_unpack


                # import ipdb
                # ipdb.set_trace()

                X_a = X_a.squeeze(1)
                X_a = X_a.to(self.device)
                logit_a = self.model(X_a)[0]

                X_i_1 = X_i_1.squeeze(1)
                X_i_1 = X_i_1.to(self.device)
                logit_i_1 = self.model(X_i_1)[0]

                if self.args.dataset_name in ['movieqa', 'tvqa', 'socialiq_a4', 'socialiq_a5']:
                    X_i_2 = X_i_2.squeeze(1)
                    X_i_2 = X_i_2.to(self.device)
                    logit_i_2 = self.model(X_i_2)[0]

                    X_i_3 = X_i_3.squeeze(1)
                    X_i_3 = X_i_3.to(self.device)
                    logit_i_3 = self.model(X_i_3)[0]

                    if self.args.dataset_name in ['movieqa', 'tvqa', 'socialiq_a5']:
                        X_i_4 = X_i_4.squeeze(1)
                        X_i_4 = X_i_4.to(self.device)
                        logit_i_4 = self.model(X_i_4)[0]

                if self.args.do_name_that_annotator:
                    if self.args.dataset_name in ['socialiq_a2', 'socialiq_permute_a2']:
                        logits = torch.cat((logit_a, logit_i_1), dim=0)
                        label = label.repeat(2)
                    elif self.args.dataset_name in ['socialiq_a4']:
                        logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3), dim=0)
                        label = label.repeat(4)
                    elif self.args.dataset_name in ['socialiq_a5', 'tvqa', 'movieqa']:
                        logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3, logit_i_4), dim=0)
                        label = label.repeat(5)
                else:
                    if self.args.dataset_name in ['socialiq_a2', 'socialiq_permute_a2']:
                        logits = torch.cat((logit_a, logit_i_1), dim=1)
                    elif self.args.dataset_name in ['socialiq_permute_a1']:
                        logits = torch.cat((logit_a, logit_i_1), dim=0)
                    elif self.args.dataset_name in ['socialiq_a4']:
                        logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3), dim=1)
                    elif self.args.dataset_name in ['socialiq_a5', 'tvqa', 'movieqa']:
                        logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3, logit_i_4), dim=1)

                if self.args.dataset_name in ['socialiq_permute_a1']:
                    label = torch.cat((torch.ones_like(logit_a), torch.zeros_like(logit_i_1)), dim=0).to(self.device)
                    predicted = (logits.data > 0) * 1
                else:
                    label = label.to(self.device)
                    _, predicted = torch.max(logits.data, dim=1)

                # import ipdb
                # ipdb.set_trace()
                if self.args.dataset_name in ['movieqa', 'tvqa']:
                    for idx_in_batch, my_question in enumerate(question):
                        my_qid = qid[idx_in_batch]
                        result_dict[my_qid] = int(predicted[idx_in_batch])
                        prediction_correctness_dict[my_qid] = (int(predicted[idx_in_batch]) == int(label[idx_in_batch]))
                elif self.args.dataset_name == 'socialiq_a5':
                    for single_predicted, single_label, single_question, single_a, single_i_1, single_i_2, single_i_3, single_i_4, single_vid_name \
                            in zip(predicted, label, question, a, i_1, i_2, i_3, i_4, vid_name):
                        df_valid_dataset = df_valid_dataset.append({'prediction_correctness': bool(int(single_predicted) == int(single_label)),
                                                                    'vid_name': single_vid_name,
                                                                    'label': int(single_label),
                                                                    'question': single_question,
                                                                    'a': single_a,
                                                                    'i_1': single_i_1,
                                                                    'i_2': single_i_2,
                                                                    'i_3': single_i_3,
                                                                    'i_4': single_i_4,
                                                                    }, ignore_index=True)
                elif self.args.dataset_name in ['socialiq_a2']:
                    if len(value_to_unpack) == 7:  # does not have annotator field
                        zipped_values = zip(logit_a, logit_i_1, predicted, label, question, a, i_1, vid_name, [None] * len(vid_name))
                    elif len(value_to_unpack) == 8:  # has annotator field
                        zipped_values = zip(logit_a, logit_i_1, predicted, label, question, a, i_1, vid_name, annotator)
                    for single_logit_a, single_logit_i_1, single_predicted, single_label, single_question, single_a, single_i_1, single_vid_name, single_annotator \
                            in zipped_values:
                        if self.args.do_name_that_annotator:
                            prob_a = nn.functional.softmax(single_logit_a).tolist()
                            prob_i_1 = nn.functional.softmax(single_logit_i_1).tolist()
                        else:
                            probs = nn.functional.softmax(torch.Tensor([float(single_logit_a.cpu()), float(single_logit_i_1.cpu())]))
                            prob_a = probs[0]
                            prob_i_1 = probs[1]
                        df_valid_dataset = df_valid_dataset.append({
                                                                    'vid_name': single_vid_name,
                                                                    'annotator': single_annotator,
                                                                    'question': single_question,
                                                                    'a': single_a,
                                                                    'i_1': single_i_1,
                                                                    'prob_a': prob_a,
                                                                    'prob_i_1': prob_i_1,
                                                                    'prediction_correctness': bool(int(single_predicted) == int(single_label)),
                                                                    'label': int(single_label),
                                                                    }, ignore_index=True)
                elif self.args.dataset_name in ['socialiq_permute_a2']:
                    zipped_values = zip(logit_a, logit_i_1, predicted, label, question, a, i_1,
                                        q_annotator, a_annotator, i_1_annotator, qai_id, video_id)
                    for single_logit_a, single_logit_i_1, single_predicted, single_label, single_question, single_a, single_i_1,\
                        single_q_annotator, single_a_annotator,  single_i_1_annotator, single_qai_id, single_video_id in zipped_values:
                        if self.args.do_name_that_annotator:
                            prob_a = nn.functional.softmax(single_logit_a).tolist()
                            prob_i_1 = nn.functional.softmax(single_logit_i_1).tolist()
                        else:
                            probs = nn.functional.softmax(torch.Tensor([float(single_logit_a.cpu()), float(single_logit_i_1.cpu())]))
                            prob_a = probs[0]
                            prob_i_1 = probs[1]
                        df_valid_dataset = df_valid_dataset.append({
                            'video_id': single_video_id,
                            'qai_id': int(single_qai_id),
                            'question': single_question,
                            'a': single_a,
                            'i': single_i_1,
                            'q_annotator': single_q_annotator,
                            'a_annotator': single_a_annotator,
                            'i_annotator': single_i_1_annotator,
                            'prob_a': float(prob_a),
                            'prob_i_1': float(prob_i_1),
                            'prediction_correctness': bool(int(single_predicted) == int(single_label)),
                        }, ignore_index=True)
                elif self.args.dataset_name in ['socialiq_permute_a1']:
                    zipped_values = zip(logit_a, logit_i_1, question, a, i_1,
                                        q_annotator, a_annotator, i_1_annotator, qai_id, video_id)
                    for single_logit_a, single_logit_i_1, single_question, single_a, single_i_1, \
                        single_q_annotator, single_a_annotator,  single_i_1_annotator, single_qai_id, single_video_id in zipped_values:
                        if self.args.do_name_that_annotator:
                            prob_a = nn.functional.softmax(single_logit_a).tolist()
                            prob_i_1 = nn.functional.softmax(single_logit_i_1).tolist()
                        else:
                            prob_a = torch.sigmoid(torch.tensor(single_logit_a.cpu()))
                            prob_i_1 = torch.sigmoid(torch.tensor(single_logit_i_1.cpu()))
                        df_valid_dataset = df_valid_dataset.append({
                            'video_id': single_video_id,
                            'qai_id': int(single_qai_id),
                            'question': single_question,
                            'a': single_a,
                            'i': single_i_1,
                            'q_annotator': single_q_annotator,
                            'a_annotator': single_a_annotator,
                            'i_annotator': single_i_1_annotator,
                            'prob_a': float(prob_a),
                            'prob_i_1': float(prob_i_1),
                        }, ignore_index=True)


                total += label.shape[0]
                correct += int((predicted == label).sum())  # For A2 task, correct answer is always at index 0
                # print(f'####correct: {correct}')
                # print(f'####total: {total}')
            accuracy = 100.00 * correct / total
            # Log and save
            log_msg = 'Epoch: {}/{}. Iteration: {}/{}. Train Loss: {}. Validation Accuracy: {}%. Train time per epoch: {}s.\n'.format(
                epoch, max_epochs, i * self.args.batch_size, len(self.training_loader) * self.args.batch_size, train_loss.item() if train_loss != None else 'N/A', accuracy,
                                   total_train_time / train_time_iteration_counter * iteration_per_epoch)


            if self.args.do_test:
                with open(os.path.join(self.args.output_dir, "test_output.txt"), "w") as pred_result_file:
                    if self.args.dataset_name == 'movieqa':
                        for test_qid in sorted(result_dict.keys(), key=lambda x: int(x.split(":")[1])):
                            pred_result_file.write(f'{test_qid} {result_dict[test_qid]}\n')
                    else:
                        raise NotImplementedError

            if self.args.output_prediction_correctness:
                if self.args.dataset_name in ['movieqa', 'tvqa']:
                    with open(os.path.join(self.args.output_dir, 'prediction_correctness.json'), 'w') as fp:
                        json.dump(prediction_correctness_dict, fp)
                elif self.args.dataset_name in ['socialiq_a2', 'socialiq_a5', 'socialiq_permute_a2', 'socialiq_permute_a1']:
                    df_valid_dataset.to_json(os.path.join(self.args.output_dir, 'prediction_correctness.json'))
                    df_valid_dataset.to_csv(os.path.join(self.args.output_dir, 'prediction_correctness.csv'))


            if not self.args.inference_only:
                log.info(log_msg)
                if i % int(0.5 * len(self.training_loader)) == 0:
                    torch.save({
                        'epoch': epoch,
                        'i': i,
                        'model_state_dict': self.model.state_dict(),
                        # 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                    }, os.path.join(self.args.output_dir, 'roberta_state_dict_' + str(epoch) + '_' + str(i * self.args.batch_size) + '.pth'))
            else:
                print(log_msg)


# Training
if __name__ == '__main__':

    args = setup_args()
    util.set_seed(2020)

    task = RobertaMultipleChoiceTask(args)
    max_epochs = args.max_epochs


    start_epoch = 0
    start_i = 0
    if task.checkpoint != None:
       #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       start_epoch = task.checkpoint['epoch']
       start_i = task.checkpoint['i']
       loss = task.checkpoint['loss']

    ## Test Forward Pass
    with torch.no_grad():
        train_example = task.training_set.__getitem__(0)
        logit_a = task.model(train_example[0].to(task.device))[0]
        logit_i = task.model(train_example[1].to(task.device))[0]



    total_train_time = 0
    train_time_iteration_counter = 0

    iteration_per_epoch = task.args.batch_size * len(task.training_loader)

    if args.inference_only:
        print('Start inference only...')
        task.valid_step(start_epoch, max_epochs)
    else:
        print("Start training...")
        for epoch in tqdm(range(start_epoch, max_epochs)):
            print("EPOCH -- {}".format(epoch))
            if epoch > start_epoch:
                start_i = 0 # continuing from checkpoint, reset i to start from 0
            for i in tqdm(range(start_i, len(task.training_loader))):
                train_loss, total_train_time, train_time_iteration_counter = \
                    task.train_step(total_train_time, train_time_iteration_counter)

                if i % (int(1.0*len(task.training_loader)) + 1) == 0:
                    task.valid_step(epoch, max_epochs, i, train_loss, total_train_time, train_time_iteration_counter, iteration_per_epoch)

###





#
#
#
#
#
#
#
#
# df_predict = pd.DataFrame()
#
# epoch = start_epoch
# i = start_i
#
#
# correct = 0
# total = 0
# for X_a, X_i_1, X_i_2, X_i_3, label, question, a, i_1, i_2, i_3, id in tqdm(testing_loader):
#     X_a = X_a.squeeze(1)
#     X_i_1 = X_i_1.squeeze(1)
#     X_i_2 = X_i_2.squeeze(1)
#     X_i_3 = X_i_3.squeeze(1)
#     X_a = X_a.to(device)
#     X_i_1 = X_i_1.to(device)
#     X_i_2 = X_i_2.to(device)
#     X_i_3 = X_i_3.to(device)
#     label = label.to(device)
#
#     logit_a = model(X_a)[0]
#     logit_i_1 = model(X_i_1)[0]
#     logit_i_2 = model(X_i_2)[0]
#     logit_i_3 = model(X_i_3)[0]
#
#     logits = torch.cat((logit_a, logit_i_1, logit_i_2, logit_i_3), dim=1)
#
#
#
#     _, predicted = torch.max(logits.data, 1)
#     total += label.shape[0]
#     correct += (predicted.cpu() == 0).sum() # For A2 task, correct answer is always at index 0
#
#
#     probability = torch.nn.functional.softmax(logits, dim=1)
#
#     # import ipdb
#     # ipdb.set_trace()
#
#     df_predict = df_predict.append([{'id': my_id, 'question': my_question,
#                                      'a': my_a,
#                                      'i_1': my_i_1,
#                                      'i_2': my_i_2,
#                                      'i_3': my_i_3,
#                                      'a_prob': a_prob,
#                                      'i_1_prob': i_1_prob,
#                                      'i_2_prob': i_2_prob,
#                                      'i_3_prob': i_3_prob,
#                                      'correct': my_correct
#                                      }
#                                     for my_question, my_a, my_i_1, my_i_2, my_i_3, my_id,
#                                         (a_prob, i_1_prob, i_2_prob, i_3_prob), my_correct
#                                     in zip(question, a, i_1, i_2, i_3, id, probability.tolist(), (predicted.cpu() == 0).tolist() )] )
#
#
# accuracy = 100.00 * correct.numpy() / total
#
# # Log and save
# log_msg = 'Epoch: {}/{}. Iteration: {}/{}. Train Loss: {}. Validation Accuracy: {}%.\n'.format(
#     epoch, max_epochs, i * batch_size, len(training_loader) * batch_size, loss.item(), accuracy)
# print(log_msg)
# # with open(output_dir + "train_log.txt", "a+") as log_file:
# #     log_file.write(log_msg)
# # torch.save({
# #     'epoch': epoch,
# #     'i': i,
# #     'model_state_dict': model.state_dict(),
# #     # 'optimizer_state_dict': optimizer.state_dict(),
# #     'loss': loss,
# # }, output_dir + 'roberta_state_dict_' + str(epoch) + '_' + str(i * batch_size) + '.pth')
#
# df_predict.reset_index(drop=True, inplace=True)
# df_predict.to_csv('explore_bias_train.csv')
