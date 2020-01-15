import argparse
import random
import numpy as np
import math
import os
import json
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
from torch.nn.modules.loss import _Loss
import pandas as pd

import time
import itertools as it

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn

def setup_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.001, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    
    parser.add_argument('--df', action='store_true')
    parser.add_argument('--save_embeddings', action='store_true')
    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_param_groups(args, model):
	no_decay = ['bias', 'LayerNorm.weight']
	if args.df:
		group1 = ['layer.0', 'layer.1.'] 
		group2 = ['layer.2', 'layer.3']
		group3 = ['layer.4', 'layer.5'] 
		group4 = ['layer.6', 'layer.7']
		group5 = ['layer.8', 'layer.9']
		group6 = ['layer.10', 'layer.11']
		group_all = ['layer.0', 'layer.1.', 'layer.2', 'layer.3', 'layer.4', 'layer.5', \
		'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], \
			'weight_decay': args.weight_decay},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**5},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**4},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**3},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2**2},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group5)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate/2},
			{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group6)], \
			'weight_decay': args.weight_decay, 'lr': args.learning_rate},

			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], \
			'weight_decay': 0.0},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**5},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**4},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**3},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group4)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2**2},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group5)], \
			'weight_decay': 0.0, 'lr': args.learning_rate/2},
			{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group6)], \
			'weight_decay': 0.0, 'lr': args.learning_rate},
		]
	else:
		optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]            
	return optimizer_grouped_parameters


def get_errors_topk(mistakes, preds, k, example_ids, label_id_to_string):
    preds = preds[:, :k]
    preds = preds[mistakes.view(-1)]
    preds = preds.numpy()
    example_ids = example_ids[mistakes.view(-1)]
    columns = ['pred_{}'.format(i+1) for i in range(k)]
    df = pd.DataFrame(preds, index=example_ids, columns=columns)
    for c in columns:
        df[c] = df[c].map(label_id_to_string)
    return df
    


def simple_accuracy(preds, labels):
    # preds = np.argmax(preds, axis=1)
    return (preds == labels).mean()

def accuracy(preds, labels, topk):
    # k = topk
    maxk = max(topk)
    batch_size = labels.shape[0]
    _, preds = torch.topk(preds, maxk, dim=1, largest=True, sorted=True)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    #at this point, you can take the complement of corrects and return the indices, then use that to index incorrect preds while converting all other indices to the correct label and plot the confusion matrix using that, so that all examples that get correct top-5 will be marked as correct while the others wont; this way, we can visualize what examples are "harder" by looking at which ones are still incorrect after k tries.
    res = []
    mistakes = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        incorrect_at_k = correct[:k].float().sum(0, keepdim=True)
        incorrect_at_k = (incorrect_at_k==0)
        mistakes.append(incorrect_at_k)
        res.append(correct_k.mul_(1.0 / batch_size))
        
    #now we also want to know the topk predictions, which are in preds
    
    return res, mistakes, preds[:maxk]

def combined_accuracy(preds_ce, preds_hinge, labels, id_to_idx, topk=(1,)):
    # k = topk
    ks = topk
    batch_size = labels.shape[0]
    _, preds_ce = torch.topk(preds_ce, 10, dim=1, largest=True, sorted=True)
    preds_ce = preds_ce.numpy()
    
    ##new part to switch out labe id for embedding idx
    for i in range(231):
        preds_ce[preds_ce==i] = id_to_idx[i]
    
    both = np.concatenate([preds_ce, preds_hinge], axis=1)
    new_preds = None
    t = time.time()
    for r, row in enumerate(both):
        scores = {}
        for i in np.unique(row):
            if len(np.where(row==i)[0])==1:
                if np.where(row==i)[0][0] < 10:
                    scores[i] = 20 - np.where(row==i)[0][0]
                else:
                    scores[i] = 30 - np.where(row==i)[0][0]
            elif len(np.where(row==i)[0])==2:
                # score = (np.sum(np.where(row==i)) - 20) / 2
                # scores[i] = 40 - score
                scores[i] = 20 - np.where(row==i)[0][0] + 5
            else:
                continue
        if r == 0:
            new_preds = torch.tensor(sorted(scores, key=scores.get, reverse=True)[:10]).view(1,-1)
        else:
            new_preds = torch.cat((new_preds, torch.tensor(sorted(scores, key=scores.get, reverse=True)[:10]).view(1,-1)), dim=0)
        if r % 20000 == 0:
            time_per_10krow = time.time() - t
            print("row {} processed in {} s".format(r, time_per_10krow))
            t = time.time()
            print(row)
            print(new_preds[r])
            print('true label: {}'.format(labels[r]))
            
    new_preds = new_preds.t()
    correct = new_preds.eq(labels.view(1,-1).expand_as(new_preds))
    
    accuracies = {}
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
        accuracies['combined_acc_top_{}'.format(k)] = correct_k.mul(1.0 / batch_size)
    return accuracies
        

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average="weighted")
    
    precision, recall, f, support = precision_recall_fscore_support(y_true=labels, y_pred=preds, average="weighted")
    return {
        # "acc": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        # "acc_and_f1": (acc + f1_weighted) / 2,
        "precision": precision,
        "recall": recall,
  
    }

def compute_f1pr_topk(preds, labels, k):
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average="weighted")
    
    precision, recall, f, support = precision_recall_fscore_support(y_true=labels, y_pred=preds, average="weighted")
    return {
        "top{}_f1_micro".format(k): round(f1_micro, 4),
        "top{}_f1_macro".format(k): round(f1_macro, 4),
        "top{}_f1_weighted".format(k): round(f1_weighted, 4),
        "top{}_precision".format(k): round(precision, 4),
        "top{}_recall".format(k): round(recall, 4),
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)
    
    
    
def plot_confusion_matrix_topk(y_true, y_pred, labels, output_dir, k, normalize=True):

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    size = len(labels) // 1.7
    plt.figure(figsize=(size,size))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm = df_cm.round(4)
    df_cm.to_csv(os.path.join(output_dir, 'confusion_matrix_at_{}.csv'.format(k)))
    heatmap = sn.heatmap(df_cm, cmap='Blues', annot=True, fmt='.2f')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=6)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=6)
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_top{}.png'.format(k)))
    
    errors = []
    
    for true_label in df_cm.index:
        l = df_cm.loc[true_label]
        l.pop(true_label)
        l = l[l > 0]
        d = l[:5].to_dict()
        d_sorted = sorted(d.items(), key=lambda x:x[1], reverse=True)
        errors.append({true_label: d_sorted})
        
    with open(os.path.join(output_dir, 'misclassification_top{}.jsonl'.format(k)), 'w') as f:
        for line in errors:
            json.dump(line, fp=f)
            f.write('\n')
        


        
        
    
    
    

class HingeTripletLoss(_Loss):
    def __init__(self, margin=0.4):
        super(HingeTripletLoss, self).__init__()
        self.margin = margin
    def forward(self, outputs, labels, negatives):
        distance_positive = F.cosine_similarity(outputs, labels, 1)
        distance_negative = F.cosine_similarity(outputs, negatives, 1)
        losses = F.relu(self.margin - distance_positive + distance_negative)
        return losses.mean()
        








class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss



