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
from data.edl import load_edl_datasets
from utils import *

def entity_mentions_to_html(doc_words, em):
    trigger_start = em['start_token']
    trigger_end = em['end_token']
    context_left = ''.join(doc_words[trigger_start-10:trigger_start])
    context_right = ''.join(doc_words[trigger_end:trigger_end+10])
    final_str = context_left + ' <span style="color:red">' + ''.join(em['mention_text']) + '</span> ' + context_right
    final_str = '<i>Entity {} </i> | '.format(em['mention_id']) + final_str
    return final_str

def visualize(jsons_dir='samples/jsons', tab_file='samples/edl_output.tab',
              output_fp='visualization.html'):
    id2doc = {}
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', do_basic_tokenize=False)
    edl_docs = load_edl_datasets(jsons_dir, tokenizer)
    id2entity = {}
    for doc in edl_docs:
        for e in doc.entity_mentions:
            id2entity[e['mention_id']] = e
    for d in edl_docs: id2doc[d.doc_id] = d
    entity_mentions = read_tab(tab_file)
    print('Loaded data')

    # Build clusters
    clusters = {}
    for e in entity_mentions:
        kb_id = e['kb_id']
        if not kb_id in clusters: clusters[kb_id] = []
        clusters[kb_id].append(e)
    singleton_clusters = [c for c in clusters if len(clusters[c]) == 1]
    for c in singleton_clusters: del clusters[c]
    print(f'Number of clusters: {len(clusters)}')

    # Visualization
    output_f = open(output_fp, 'w+', encoding='utf-8')
    for ix, cluster in enumerate(clusters.values()):
        output_f.write(f'<h2>Cluster {ix+1} (Size {len(cluster)})</h2>')
        for e in cluster:
            doc_id = e['doc_id']
            start_char = e['start_char']
            end_char = e['end_char']
            doc_words = id2doc[e['doc_id']].words
            output_f.write(entity_mentions_to_html(doc_words, id2entity[f'{doc_id}:{start_char}-{end_char}']))
            output_f.write('</br>')
    output_f.close()

if __name__ == '__main__':
    visualize()
