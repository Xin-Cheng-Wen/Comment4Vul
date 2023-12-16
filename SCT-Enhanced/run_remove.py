# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import torch
import numpy as np
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model import Model
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification)

logger = logging.getLogger(__name__)
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                #  comment_tokens,
                #  comment_ids,
                 symbolic_tokens,
                 symbolic_ids,
                 index,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        # self.comment_tokens = comment_tokens
        # self.comment_ids = comment_ids
        self.symbolic_tokens = symbolic_tokens
        self.symbolic_ids = symbolic_ids
        self.index = index
        self.label = label

        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    # code = ' '.join(js['func'].split())
    # code_tokens = tokenizer.tokenize(code)[:args.block_size-4]

    code = js['func']
    # .replace('\n', '<n>')
    code = ' '.join(code.split())
    # code = code.replace('<n>', '\n')
    code_tokens = tokenizer.tokenize(code)[:args.block_size-4]

    comment_code = js['choices']
    # .replace('\n', '<n>')
    comment_code = ' '.join(comment_code.split())
    # comment_code = comment_code.replace('<n>', '\n')
    comment_code_tokens = tokenizer.tokenize(comment_code)[:args.block_size-4]

    symbolic_code = js['clean_code']
    # .replace('\n', '<n>')
    symbolic_code = ' '.join(symbolic_code.split())
    # symbolic_code = symbolic_code.replace('<n>', '\n')
    symbolic_code_tokens = tokenizer.tokenize(symbolic_code)[:args.block_size-4]
    
    # code_tokens = tokenizer.tokenize(str(js['func']))[:args.block_size-4]
    source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length

    # comment_source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + comment_code_tokens + [tokenizer.sep_token]
    # comment_source_ids = tokenizer.convert_tokens_to_ids(comment_source_tokens)
    # comment_padding_length = args.block_size - len(comment_source_ids)
    # comment_source_ids += [tokenizer.pad_token_id]*comment_padding_length

    symbolic_source_tokens = [tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token] + symbolic_code_tokens + [tokenizer.sep_token]
    symbolic_source_ids = tokenizer.convert_tokens_to_ids(symbolic_source_tokens)
    symbolic_padding_length = args.block_size - len(symbolic_source_ids)
    symbolic_source_ids += [tokenizer.pad_token_id]*symbolic_padding_length

    # return InputFeatures(source_tokens,source_ids, comment_source_tokens, comment_source_ids, symbolic_source_tokens, symbolic_source_ids, js['idx'],int(js['target']))
    return InputFeatures(source_tokens,source_ids, symbolic_source_tokens, symbolic_source_ids, js['idx'],int(js['target']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []

        with open(file_path) as f:
            for line in f:
                
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        
        # return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].comment_ids), torch.tensor(self.examples[i].symbolic_ids)
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].symbolic_ids)
             

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    
    args.max_steps = args.num_train_epochs*len( train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_acc = [], 0
    
    model.zero_grad()
    for idx in range(args.num_train_epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))

        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)    
            labels = batch[1].to(args.device)
            # comment_inputs = batch[2].to(args.device)    
            symbolic_inputs = batch[2].to(args.device)    

            model.train()
            # loss,logits = model(inputs, comment_inputs, symbolic_inputs, labels)
            loss,logits = model(inputs, symbolic_inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            # if (step+1)% 100==0:
            #     logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  

        results = evaluate(args, model, tokenizer, args.eval_data_file)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))                    

        if results['eval_acc'] > best_acc:
            best_acc = results['eval_acc']
            logger.info("  "+"*"*20)  
            logger.info("  Best map:%s",round(best_acc,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))   
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, data_file):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size, num_workers=4)
    
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)    
        label = batch[1].to(args.device)
        # comment_inputs = batch[2].to(args.device)  
        symbolic_inputs = batch[2].to(args.device)  
        with torch.no_grad():
            # lm_loss, logit = model(inputs, comment_inputs, symbolic_inputs, label)
            lm_loss, logit = model(inputs, symbolic_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits,0)
    labels = np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    eval_f1 = f1_score(labels, preds) 
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
          
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4)
    }

    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("acc:",acc)
    print("pre:",pre)
    print("rec:",rec)
    print("f1:",f1)
    return result

                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels=1

    # model = RobertaModel.from_pretrained(args.model_name_or_path) 
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path) 


    model = Model(model,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training     
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        train(args, train_dataset, model, tokenizer)
        
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],2)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],2)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    main()


