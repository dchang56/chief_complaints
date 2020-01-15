import argparse
import glob
import logging
import os
import random
import numpy as np
import json
import pandas as pd
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertModel,
                                  BertForSequenceClassification, BertTokenizer)
##experimenting w optimizers
from transformers import WarmupLinearSchedule
from utils import RAdam

from data_utils import (convert_examples_to_features,
                        output_modes, processors)

from utils import (setup_parser, set_seed, get_param_groups, compute_metrics, accuracy, plot_confusion_matrix_topk, compute_f1pr_topk, get_errors_topk)
from parallel import DataParallelModel, DataParallelCriterion
import parallel

from torch.nn import CrossEntropyLoss

from model import BertForSequenceClassification_with_emb

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification_with_emb, BertTokenizer)
}

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = t_total // 100

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_param_groups(args, model)
    optimizer = RAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = DataParallelModel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    args.logging_steps = len(train_dataloader) // 1
    args.save_steps = args.logging_steps
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        args.current_epoch = _
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,}  # XLM and RoBERTa don't use segment_ids
                    #   'labels':         batch[3]}
            outputs = model(**inputs)
            outputs = [outputs[i][0] for i in range(len(outputs))]
            
            loss_fct = CrossEntropyLoss()
            loss_fct = DataParallelCriterion(loss_fct)
        
            loss = loss_fct(outputs, batch[3])
    
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name, )
    eval_outputs_dirs = (args.output_dir, )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as reader:
            label_map = json.load(reader)
        label_id_to_string = {i:label for label,i in label_map.items()}

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        preds = None
        out_label_ids = None
        example_ids = None
        pooled_outputs = None
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,}  # XLM and RoBERTa don't use segment_ids
                        #   'labels':         batch[3]}
                outputs = model(**inputs)
                
                # outputs = [outputs[i][0] for i in range(len(outputs))]
                outputs = [outputs[i] for i in range(len(outputs))]
                
                
            logits = parallel.gather([output[0] for output in outputs], target_device='cuda:0')
            pooled_output = parallel.gather([output[1] for output in outputs], target_device='cuda:1')

            if preds is None:
                preds = logits.detach().cpu()#.numpy()
                pooled_outputs = pooled_output.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu()#.numpy()
                example_ids = batch[4].detach().cpu().numpy()
            else:
                preds = torch.cat((preds, logits.detach().cpu()), axis=0)
                pooled_outputs = np.append(pooled_outputs, pooled_output.detach().cpu().numpy(), axis=0)
                out_label_ids = torch.cat((out_label_ids, batch[3].detach().cpu()), axis=0)
                example_ids = np.append(example_ids, batch[4].detach().cpu().numpy(), axis=0)
            
            
        k_values = (1, 2, 3, 4, 5)
        topk_accuracies, mistakes_at_k, preds_topk = accuracy(preds, out_label_ids, k_values)
        preds_topk = preds_topk.t()
        
        ##in order to extract specific examples at topk, need to take difference of mistakes at k and mistakes at k-1
        #those would be the examples that were correctly predicted at k
        #so if we do that for k 1-5, we can show concrete examples of chief complaints at increasing levels of "difficulty"
        
        #so, mistakes1 != mistakes2 would give you true for indices that were wrong at top1 but correct at top2
        #all you gotta do now is match these with text and labels
        
        #we could show this for top1-2, and top4-5 to emphasize the different types of errors we see
        
        ### Top-1 to Top-2
        top1to2 = (mistakes_at_k[0] != mistakes_at_k[1])
        df_1to2 = get_errors_topk(top1to2, preds_topk, 2, example_ids, label_id_to_string)
        df_1to2.to_csv(os.path.join(eval_output_dir, 'df_1to2.csv'))
        ### Top-4 to Top-5
        top4to5 = (mistakes_at_k[3] != mistakes_at_k[4])
        df_4to5 = get_errors_topk(top4to5, preds_topk, 5, example_ids, label_id_to_string)
        df_4to5.to_csv(os.path.join(eval_output_dir, 'df_4to5.csv'))
        ### Top-8 to Top-9
        # top8to9 = (mistakes_at_k[7] != mistakes_at_k[8])
        # df_8to9 = get_errors_topk(top8to9, preds_topk, 9, example_ids, label_id_to_string)
        # df_8to9.to_csv(os.path.join(eval_output_dir, 'df_8to9.csv'))
        
        

        
        
        
        
        
        preds = preds.numpy()
        out_label_ids = out_label_ids.numpy()
        preds = np.argmax(preds, axis=1)
        # result = compute_metrics(eval_task, preds, out_label_ids)
        for k, acc in zip(k_values, topk_accuracies):
            results['top{}_acc'.format(k)] = round(acc.item(), 4)
        # results.update(result)
        
        labels = [label_id_to_string[i] for i in out_label_ids]
        label_ids = range(len(label_map.keys()))
        label_list = [label_id_to_string[i] for i in label_ids]
        #we want to copy over non-mistake indices from out_label_ids to preds
        for k, mistakes in zip(k_values, mistakes_at_k):
            m_k = mistakes.view(-1).numpy()
            preds[~m_k] = out_label_ids[~m_k]
            result = compute_f1pr_topk(preds, out_label_ids, k)
            if args.final_eval:
                plot_confusion_matrix_topk(out_label_ids, preds, label_list, args.output_dir, k)      
            results.update(result)
        
        
        metrics = ['acc', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision', 'recall']
        results_lists = []
        for i in k_values:
            metrics_for_k = []
            for m in metrics:
                metrics_for_k.append(results["top{}_{}".format(i, m)])
            results_lists.append(metrics_for_k)
        results_df = pd.DataFrame(results_lists, index=k_values, columns=metrics)
        results_df.to_csv(os.path.join(eval_output_dir, "results_df.csv"))
        
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                if 'acc' in key:
                    logger.info("  %s = %s", key, str(results[key]))
                    writer.write("%s\t%s\n" % (key, str(round(results[key], 4))))
            writer.write('\n')

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.data_dir)
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features, label_map = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            with open(os.path.join(args.data_dir, 'label_map.json'), 'w') as writer:
                json.dump(label_map, writer)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_example_ids = torch.tensor([f.example_id for f in features],
                                   dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_example_ids)
    return dataset

def save_embeddings(args, model, tokenizer):
    with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as reader:
        label_map = json.load(reader)
    label_id_to_string = {i:label for label,i in label_map.items()}
    
    #Train set
    pooled_output_file = os.path.join(args.output_dir, 'train_embeddings.csv')
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.eval_batch_size)
    logger.info("***** Generating Embeddings for Train Examples")
    for batch in tqdm(train_dataloader, desc="Train Embedding"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,}  # XLM and RoBERTa don't use segment_ids
                        #   'labels':         batch[3]}
            outputs = model(**inputs)
            outputs = [outputs[i] for i in range(len(outputs))]
        pooled_outputs = parallel.gather([output[1] for output in outputs], target_device='cuda:0')
        
        pooled_outputs = pooled_outputs.detach().cpu().numpy()
        out_label_ids = batch[3].detach().cpu().numpy()
        example_ids = batch[4].detach().cpu().numpy()
        labels = [label_id_to_string[i] for i in out_label_ids]
        pooled_outputs = pd.DataFrame(pooled_outputs)
        pooled_outputs['example_id'] = example_ids
        pooled_outputs['label'] = labels
        pooled_outputs.to_csv(pooled_output_file, mode='a')
    
    #Dev set
    logger.info("***** Generating Embeddings for Dev Examples")
    pooled_output_file = os.path.join(args.output_dir, 'dev_embeddings.csv')
    eval_task = args.task_name
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)    
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    for batch in tqdm(eval_dataloader, desc="Dev Embedding"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,}  # XLM and RoBERTa don't use segment_ids
                        #   'labels':         batch[3]}
            outputs = model(**inputs)
            outputs = [outputs[i] for i in range(len(outputs))]
        pooled_outputs = parallel.gather([output[1] for output in outputs], target_device='cuda:0')
        
        pooled_outputs = pooled_outputs.detach().cpu().numpy()
        out_label_ids = batch[3].detach().cpu().numpy()
        example_ids = batch[4].detach().cpu().numpy()
        labels = [label_id_to_string[i] for i in out_label_ids]
        pooled_outputs = pd.DataFrame(pooled_outputs)
        pooled_outputs['example_id'] = example_ids
        pooled_outputs['label'] = labels
        pooled_outputs.to_csv(pooled_output_file, mode='a')

def main():
    args = setup_parser()
    args.final_eval = False

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    # logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
        if args.n_gpu > 1:
            model = DataParallelModel(model)


    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            if args.n_gpu > 1:
               model = DataParallelModel(model)
            args.final_eval = True
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    
    if args.save_embeddings:
        save_embeddings(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()