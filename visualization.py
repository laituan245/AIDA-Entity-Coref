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

def visualize(pretrained_model):
    configs = prepare_configs('basic')
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    dataset = combine_datasets([prepare_dataset(ONTONOTE, tokenizer)])
    model = CorefModel(configs)
    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('Prepared model, tokenizer, and dataset')

if __name__ == '__main__':
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--pretrained_model', default=None)
    args = parser.parse_args()

    # Visualize
    visualize(args.pretrained_model)
