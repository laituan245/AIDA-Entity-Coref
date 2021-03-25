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
from utils import get_predicted_antecedents, evaluate, RunningAverage, prepare_configs

def flatten_input_ids(input_ids, mask_windows):
    nb_windows, window_len = input_ids.shape
    flattened_input_ids = []
    for i in range(nb_windows):
        cur_windows = input_ids[i, :].tolist()
        cur_masks = mask_windows[i,:].tolist()
        for j in range(window_len):
            if cur_masks[j] > 0:
                flattened_input_ids.append(cur_windows[j])
    return flattened_input_ids

def doc_to_html(doc_words, event_mentions):
    doc_words = [str(word) for word in doc_words]
    for e in event_mentions:
        t_start, t_end = e['trigger']['start'], e['trigger']['end']
        doc_words[t_start] = '<span style="color:blue">' + doc_words[t_start]
        doc_words[t_end] = doc_words[t_end] + '</span>'
    return ''.join(doc_words)

def event_mentions_to_html(doc_words, em):
    trigger_start = em['trigger']['start']
    trigger_end = em['trigger']['end']
    context_left = ''.join(doc_words[trigger_start-10:trigger_start])
    context_right = ''.join(doc_words[trigger_end:trigger_end+10])
    final_str = context_left + ' <span style="color:red">' + em['trigger']['text'] + '</span> ' + context_right
    final_str = '<i>Entity {} </i> | '.format(em['id']) + final_str
    return final_str

def visualize(pretrained_model):
    output_file = open('visualization.html', 'w+')

    configs = prepare_configs('basic')
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    dataset = prepare_dataset(ONTONOTE, tokenizer)
    model = CorefModel(configs)
    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('Prepared model, tokenizer, and dataset')

    # Evaluation
    with torch.no_grad():
        print('Evaluation on the (combined) dev set')
        evaluate(model, dataset, DEV)
        print('Evaluation on the (combined) test set')
        evaluate(model, dataset, TEST)

    # Visualize predictions on the test set
    split = 'test'
    total_examples = len(dataset.examples[split])
    eval_data = zip(dataset.tensorized_examples[split], dataset.examples[split])
    for example_num, (tensorized_example, example) in enumerate(eval_data):
        input_ids, input_masks, is_training, gold_starts, gold_ends, _, mask_windows = tensorized_example
        gold_starts, gold_ends = gold_starts.tolist(), gold_ends.tolist()
        locs = list(zip(gold_starts, gold_ends))
        flattened_input_ids = flatten_input_ids(input_ids, mask_windows)
        tokens = tokenizer.convert_ids_to_tokens(flattened_input_ids)
        doc_string = tokenizer.convert_tokens_to_string(tokens)
        entities = []
        for entity_id, loc in enumerate(locs):
            entity = {}
            start, end = loc
            entity['id'] = entity_id
            entity['trigger'] = {'start': start, 'end': end}
            entity['trigger']['text'] = tokenizer.convert_tokens_to_string(tokens[start:end+1])
            entities.append(entity)
        # Apply the model for prediction
        with torch.no_grad():
            preds = model(*tensorized_example)[1]
            preds = [x.cpu().data.numpy() for x in preds]
            mention_starts, mention_ends, top_antecedents, top_antecedent_scores = preds
            predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

            # Decide cluster from predicted_antecedents
            predicted_clusters, m2cluster = [], {}
            for ix, (s, e) in enumerate(zip(mention_starts, mention_ends)):
                if predicted_antecedents[ix] < 0:
                    cluster_id = len(predicted_clusters)
                    predicted_clusters.append([entities[ix]])
                else:
                    antecedent_idx = predicted_antecedents[ix]
                    p_s, p_e = mention_starts[antecedent_idx], mention_ends[antecedent_idx]
                    cluster_id = m2cluster[(p_s, p_e)]
                    predicted_clusters[cluster_id].append(entities[ix])
                m2cluster[(s,e)] = cluster_id
            doc_key = example['doc_key']
            event_mentions = entities
            output_file.write('<b>Document {}</b><br>'.format(doc_key))
            output_file.write('{}<br><br><br>'.format(doc_to_html(tokens, event_mentions)))
            for ix, cluster in enumerate(predicted_clusters):
                if len(cluster) == 1: continue
                output_file.write('<b>Cluster {}</b></br>'.format(ix+1))
                for em in cluster:
                    output_file.write('{}<br>'.format(event_mentions_to_html(tokens, em)))
                output_file.write('<br><br>')
            output_file.write('<br><hr>')
            output_file.write(f'</br></hr>')
    output_file.close()

if __name__ == '__main__':
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--pretrained_model', default='model.pt')
    args = parser.parse_args()

    # Visualize
    visualize(args.pretrained_model)
