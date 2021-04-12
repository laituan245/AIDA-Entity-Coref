import os
import math
import torch
import pyhocon
import utils.metrics as metrics
import numpy as np
import xml.etree.ElementTree as ET

from models import *
from transformers import *
from constants import *
from boltons.iterutils import pairwise, windowed
from os import listdir
from os.path import isfile, join

def read_bio_file(bio_fp):
    id2words, id2kbids, id2tags = {}, {}, {}
    with open(bio_fp, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            token, loc, kb_id, tag = line.split()
            doc_id, text_loc = loc.split(':')
            if not doc_id in id2words: id2words[doc_id] = []
            if not doc_id in id2kbids: id2kbids[doc_id] = []
            if not doc_id in id2tags: id2tags[doc_id] = []
            id2words[doc_id].append(token)
            id2kbids[doc_id].append(kb_id)
            id2tags[doc_id].append(tag)

    # Extract entity_mentions
    entity_mentions = []
    for doc_id in id2words:
        tags = id2tags[doc_id]
        kbids = id2kbids[doc_id]
        for i in range(len(tags)):
            if tags[i].startswith('B-'):
                j = i + 1
                while j < len(tags) and tags[j].startswith('I-'): j += 1
                entity_mentions.append((doc_id, i, j-1, kbids[i]))

    return id2words, entity_mentions


def read_tab(tab_fp):
    entity_mentions = []
    with open(tab_fp, 'r', encoding='utf8') as f:
        for line in f:
            es = line.strip().split('\t')
            loc = es[3]
            doc_id, text_loc = loc.split(':')
            start_char, end_char = text_loc.split('-')
            entity_mentions.append({
                'mention_id': es[1],
                'text': es[2],
                'doc_id': doc_id,
                'start_char': start_char,
                'end_char': end_char,
                'kb_id': es[4],
                'entity_type': es[5],
            })
    return entity_mentions


def read_ltf_files(ltf_folder_path):
    doc2tokens = {}
    ltf_files = [f for f in listdir(ltf_folder_path) if isfile(join(ltf_folder_path, f)) if f.endswith('.ltf.xml')]
    for ltf_file in ltf_files:
        doc_id = ltf_file[:ltf_file.find('.ltf.xml')]
        fp = join(ltf_folder_path, ltf_file)
        tokens = read_ltf(fp)
        doc2tokens[doc_id] = tokens
    return doc2tokens

def read_ltf(ltf_file_path):
    tokens = []
    tree = ET.parse(ltf_file_path)
    root = tree.getroot()
    for doc in root:
        for text in doc:
            for seg in text:
                for token in seg:
                    if token.tag == "TOKEN":
                        token_beg = int(token.attrib["start_char"])
                        token_end = int(token.attrib["end_char"])
                        tokens.append((token_beg, token_end, token.text))
    return tokens

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def prepare_configs(config_name):
    # Extract the requested config
    print('Config {}'.format(config_name), flush=True)
    configs = pyhocon.ConfigFactory.parse_file(BASIC_CONF_PATH)[config_name]
    configs['saved_path'] = 'trained'
    print(configs, flush=True)

    # Create the log_root directory (if not exist)
    if not os.path.exists(configs['saved_path']):
        os.makedirs(configs['saved_path'])

    return configs

def extract_input_masks_from_mask_windows(mask_windows):
    input_masks = []
    for mask_window in mask_windows:
        subtoken_count = listRightIndex(mask_window, -3) + 1
        input_masks.append([1] * subtoken_count + [0] * (len(mask_window) - subtoken_count))
    input_masks = np.array(input_masks)
    return input_masks

def convert_to_sliding_window(expanded_tokens, sliding_window_size, tokenizer):
    """
    construct sliding windows, allocate tokens and masks into each window
    :param expanded_tokens:
    :param sliding_window_size:
    :return:
    """
    CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])
    SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])
    PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])
    expanded_masks = [1] * len(expanded_tokens)
    sliding_windows = construct_sliding_windows(len(expanded_tokens), sliding_window_size - 2)
    token_windows = []  # expanded tokens to sliding window
    mask_windows = []  # expanded masks to sliding window
    for window_start, window_end, window_mask in sliding_windows:
        original_tokens = expanded_tokens[window_start: window_end]
        original_masks = expanded_masks[window_start: window_end]
        window_masks = [-2 if w == 0 else o for w, o in zip(window_mask, original_masks)]
        one_window_token = CLS + original_tokens + SEP + PAD * (sliding_window_size - 2 - len(original_tokens))
        one_window_mask = [-3] + window_masks + [-3] + [-4] * (sliding_window_size - 2 - len(original_tokens))
        assert len(one_window_token) == sliding_window_size
        assert len(one_window_mask) == sliding_window_size
        token_windows.append(one_window_token)
        mask_windows.append(one_window_mask)
    return token_windows, mask_windows

def construct_sliding_windows(sequence_length: int, sliding_window_size: int):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows

def bucket_distance(distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = torch.floor(torch.log2(distances.float())).long() + 3
    use_identity = (distances <= 4).long()
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return torch.clamp(combined_idx, 0, 9)

def evaluate(model, dataset, split):
    assert(split in ['dev', 'test'])

    doc_keys = []
    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()

    total_examples = len(dataset.examples[split])
    eval_data = zip(dataset.tensorized_examples[split], dataset.examples[split])
    for example_num, (tensorized_example, example) in enumerate(eval_data):
        doc_keys.append(example['doc_key'])

        # Apply the model for prediction
        loss, preds = model(*tensorized_example)
        preds = [x.cpu().data.numpy() for x in preds]
        top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = preds
        predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)

        # Update coref_predictions
        coref_predictions[example['doc_key']] = evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example['clusters'], coref_evaluator)

    summary_dict = {}

    p,r,f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
    summary_dict["Average precision (py)"] = p
    print("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    print("Average recall (py): {:.2f}%".format(r * 100))
    return f

def get_predicted_antecedents(antecedents, antecedent_scores):
    predicted_antecedents = []
    try:
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
    except:
        pass
    return predicted_antecedents

def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            continue
        assert i > predicted_index, (i, predicted_index)
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster

        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

def evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

def prune(spans, mention_scores, T, LAMBDA=0.40):
    """ Keep up to λT spans with the highest mention scores, where T = len(doc)
    """

    STOP = int(LAMBDA * T) # Only take top λT spans, where T = len(doc)
    nb_spans = len(spans)
    span_ids = list(range(nb_spans))
    aggregated_spans = list(zip(span_ids, spans, mention_scores))

    # Sort by mention score, remove overlapping spans, prune to top λT spans
    sorted_spans = sorted(aggregated_spans, key=lambda s: s[2], reverse=True)
    nonoverlapping = remove_overlapping(sorted_spans)
    pruned_spans = nonoverlapping[:STOP]

    # Resort by start, end indexes
    spans = sorted(pruned_spans, key=lambda s: (s[1][0], s[1][-1]))

    return [s[0] for s in spans]

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score
    unless the current span i yields true to the following condition with any
    previously accepted span j:
    si.start < sj.start <= si.end < sj.end   OR
    sj.start < si.start <= sj.end < si.end """

    # Nonoverlapping will be accepted spans, seen is start, end indexes that
    # have already been seen in an accepted span
    nonoverlapping, seen = [], set()
    for s in sorted_spans:
        indexes = s[1]
        taken = [i in seen for i in indexes]
        if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
            nonoverlapping.append(s)
            seen.update(indexes)

    return nonoverlapping

def compute_idx_spans(sent_lengths, L):
    """
    Compute span indexes for all possible spans up to length L in each sentence

    >>> compute_idx_spans([3, 2], L = 2)
    [(0,), (1,), (2,), (0, 1), (1, 2), (3,), (4,), (3, 4)]
    >>> compute_idx_spans([3, 5], L = 3)
    [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2), (3,), (4,), (5,), (6,), (7,),
    (3, 4), (4, 5), (5, 6), (6, 7), (3, 4, 5), (4, 5, 6), (5, 6, 7)]
    """
    idx_spans, shift = [], 0
    for sent_length in sent_lengths:
        sent_spans = flatten([windowed(range(shift, sent_length+shift), length)
                              for length in range(1, L+1)])
        idx_spans.extend(sent_spans)
        shift += sent_length
    return idx_spans

def is_subsequence(needle, haystack):
    """
    Finds if a list is a subsequence of another.

    * args
        needle: the candidate subsequence
        haystack: the parent list

    * returns
        boolean

    >>> is_subsequence([1, 2, 3, 4], [1, 2, 3, 4, 5, 6])
    True
    >>> is_subsequence([1, 2, 3, 4], [1, 2, 3, 5, 6])
    False
    >>> is_subsequence([6], [1, 2, 3, 5, 6])
    True
    >>> is_subsequence([5, 6], [1, 2, 3, 5, 6])
    True
    >>> is_subsequence([[5, 6], 7], [1, 2, 3, [5, 6], 7])
    True
    >>> is_subsequence([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, [5, 6], 7])
    False
    """
    return any(
        haystack[i:i+len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )

def flatten(l):
    return [item for sublist in l for item in sublist]

def print_trainable_params(model):
    print('Parameters of the model:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

# Get total number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)
