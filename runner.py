import os
import copy
import utils
import torch
import math
import random
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

from constants import *
from transformers import *
from models import CorefModel
from argparse import ArgumentParser
from data import prepare_dataset, combine_datasets
from utils import evaluate, RunningAverage, prepare_configs

# Main Functions
def train(config_name):
    # Prepare the config, the tokenizer, and the model
    configs = prepare_configs(config_name)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    model = CorefModel(configs)
    if PRETRAINED_MODEL:
        checkpoint = torch.load(PRETRAINED_MODEL)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Prepare datasets
    ace05_dataset = prepare_dataset(ACE05, tokenizer)
    kbp2016_dataset = prepare_dataset(KBP2016, tokenizer)
    kbp2017_dataset = prepare_dataset(KBP2017, tokenizer)
    kbp_dataset = combine_datasets([kbp2016_dataset, kbp2017_dataset])
    ontonote_dataset = prepare_dataset(ONTONOTE, tokenizer)
    dataset = combine_datasets([ontonote_dataset, ace05_dataset, kbp_dataset])
    print('Number of train: {}'.format(len(dataset.examples[TRAIN])))
    print('Number of dev: {}'.format(len(dataset.examples[DEV])))
    print('Number of test: {}'.format(len(dataset.examples[TEST])))

    # Prepare the optimizer and the scheduler
    num_train_docs = len(dataset.examples[TRAIN])
    num_epoch_steps = math.ceil(num_train_docs / configs['batch_size'])
    num_train_steps = int(num_epoch_steps * configs['epochs'])
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer = model.get_optimizer(num_warmup_steps, num_train_steps)
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start training
    accumulated_loss = RunningAverage()
    best_dev_f1, iters, batch_loss = 0, 0, 0
    for i in range(configs['epochs']):
        print('Starting epoch {}'.format(i+1), flush=True)
        train_indices = list(range(num_train_docs))
        random.shuffle(train_indices)
        for train_idx in train_indices:
            iters += 1
            tensorized_example = dataset.tensorized_examples[TRAIN][train_idx]
            iter_loss = model(*tensorized_example)[0]
            iter_loss /= configs['batch_size']
            iter_loss.backward()
            batch_loss += iter_loss.data.item()
            if iters % configs['batch_size'] == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0
            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, accumulated_loss()), flush=True)
                accumulated_loss = RunningAverage()

        # Evaluation after each epoch
        with torch.no_grad():
            print('Evaluation on the (aggregated) dev set')
            dev_f1 = evaluate(model, dataset, DEV)
            print('Evaluation on the (aggregated) test set')
            evaluate(model, dataset, TEST)
            # Individual Test Set
            print('Evaluation on the Ontonote test set')
            evaluate(model, ontonote_dataset, TEST)
            print('Evaluation on the ACE05 test set')
            evaluate(model, ace05_dataset, TEST)
            print('Evaluation on the KBP test set')
            evaluate(model, kbp_dataset, TEST)


        # Save model if it has better F1 score
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            # Save the model
            save_path = os.path.join(configs['saved_path'], 'model.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)

if __name__ == '__main__':
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_name', default='spanbert_large')
    args = parser.parse_args()

    # Start training
    train(args.config_name)
