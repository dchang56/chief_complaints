import argparse
import logging
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertModel,
                                  BertForSequenceClassification, BertTokenizer)
from utils import *
from data_utils import *

logger = logging.getLogger(__name__)

   #first, load and cache examples
    #run batch through model
    #map to labels
    #save outputs with "probability" scores for each example
    #stdprint fewer than n examples, save all to file

class InputFeatures_predict(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, example_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.example_id = example_id
        
def convert_chief_complaints_to_features(examples, max_seq_length,
                                         tokenizer, cls_token='[CLS]',
                                         cls_token_segment_id=1,
                                         sep_token='[SEP]',
                                         pad_token=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        special_tokens_count = 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

  
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 4:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
                InputFeatures_predict(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              example_id=example.guid))
    return features

def load_and_cache_chief_complaints(args, tokenizer):
    processor = processors['cc']()
    cached_features_file = os.path.join(args.output_dir, 'cached_data')
    if os.path.exists(cached_features_file):
        logger.info('loading cached features from cached file {}'.format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info('Creating features from input file')
        examples = processor.get_chief_complaints(args.input_file)
        features = convert_chief_complaints_to_features(examples, args.max_seq_length,
                                                        tokenizer)
        logger.info('Saving features into cached file {}'.format(cached_features_file))
        torch.save(features, cached_features_file)
    #convert to tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_ids = torch.tensor([f.example_id for f in features],
                                   dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_ids)
    return dataset

    
def predict(args, model, tokenizer):
    dataset = load_and_cache_chief_complaints(args, tokenizer)
    with open(args.label_map, 'r') as reader:
        label_map = json.load(reader)
    label_id_to_string = {i:label for label,i in label_map.items()}

    
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    logger.info('**** Processing chief complaints ****')
    logger.info('   Num samples = {}'.format(dataset))
    
    preds = None
    
    for batch in tqdm(dataloader, desc="Processing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      }
            outputs = model(**inputs)
            logits = outputs[0] #confirm that this has same dim as num labels
        if preds is None:
            preds = F.softmax(logits.detach().cpu(), dim=1)
            example_ids = batch[3].detach().cpu().numpy()
        else:
            preds = torch.cat((preds, F.softmax(logits.detach().cpu(), dim=1)), axis=0)
            examples_ids = np.append(example_ids, batch[4].detach().cpu().numpy(), axis=0)
        
    k = 5
    #now we just want to print the top 3 predictions per sample
    preds_prob, preds_topk = torch.topk(preds, k, dim=1, largest=True, sorted=True)
    #now you have a matrix with 5 columns and n rows
    #we just want to map those indices to strings along w probs
    columns = ['pred_{}'.format(i+1) for i in range(k)]
    preds_topk = preds_topk.numpy()
    preds_prob = preds_prob.numpy()
    df_labels = pd.DataFrame(preds_topk, index=example_ids, columns=columns)
    for c in columns:
        df_labels[c] = df_labels[c].map(label_id_to_string)
    df_probs = pd.DataFrame(preds_prob, index=example_ids, columns=columns)
    
    df_labels.to_csv(os.path.join(args.output_dir, 'prediction_labels.csv'))
    df_probs.to_csv(os.path.join(args.output_dir, 'prediction_probs.csv'))

    return df_labels, df_probs

def main():
    parser = argparse.ArgumentParser()
    
    ## Required arguments
    parser.add_argument('--input_file', default=None, type=str, required=True,
                        help='path to the input file with one chief complaint per row')
    parser.add_argument('--model_path', default=None, type=str, required=True,
                        help='path to pretrained model')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='output directory for predictions')
    parser.add_argument('--label_map', default=None, type=str, required=True,
                        help='path to label map for the corresponding model')
    
    ## Other arguments
    parser.add_argument('--config_name', default="", type=str,
                        help='pretrained config path')
    parser.add_argument('--tokenizer_name', default="", type=str,
                        help='pretrained tokenizer path')
    parser.add_argument('--max_seq_length', default=64, type=int,
                        help='max total input sequence length after tokenization')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='uncased model')
    parser.add_argument('--batch_size', default=1000, type=int,
                        help='batch size for processing chief complaint file')
    
    args = parser.parse_args()
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
                        
    ## Setting up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # with open(os.path.join(args.input_file, 'label_map.json'), 'r') as reader:
        # label_map = json.load(reader)
    # label_id_to_string = {i:label for label,i in label_map.items()}
    # num_labels = len(label_id_to_string)
    
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_path, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.to(args.device)
    
    #Predict
    df_labels, df_probs = predict(args, model, tokenizer)
    
    #Display
    output_file = os.path.join(args.output_dir, 'output.txt')
    with open(output_file, 'w') as writer:
        with open(args.input_file, 'r') as f:
            print('**** Printing sample outputs ****')
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    writer.write("sample {}\n".format(i))
                    writer.write(line)
                    writer.write("\n")
                    x = pd.concat([df_labels.loc[i], df_probs.loc[i]], axis=1)
                    x.columns = ["label", "score"]
                    writer.write("{}".format(x))
                    writer.write("\n\n")
                    if i < 5:
                        print("sample {}".format(i))
                        print(line)
                        print('{}'.format(x))
                        print('\n\n')
            print("results saved to {}".format(args.output_dir))
                    
    
    
    
if __name__ == "__main__":
    main()