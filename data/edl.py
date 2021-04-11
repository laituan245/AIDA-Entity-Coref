import os
import copy
import json

from os import listdir
from os.path import isfile, join
from utils import *

# Helper Functions
def load_edl_doc(doc_path, tokenizer, verbose=True):
    with open(doc_path, 'r', encoding='utf-8') as r:
        doc_id, words_ctx, tokens_ids, sentences, entity_mentions = None, 0, [], [], []
        for line in r:
            sent = json.loads(line)
            sentences.append(sent['tokens'])
            graph = sent['graph']
            tokens_ids += sent['token_ids']
            for entity in graph['entities']:
                entity[0] += words_ctx
                entity[1] += words_ctx-1  # Convert to inclusive endpoint
                entity_mentions.append({
                    'start_token': entity[0], 'end_token': entity[1]
                })
            # Update words_ctx
            words_ctx += len(sent['tokens'])
            # Update doc_id (if None)
            if doc_id is None: doc_id = sent['doc_id']
            else: assert(doc_id == sent['doc_id'])
    words = flatten(sentences)
    assert(len(words) == words_ctx)
    assert(len(words) == len(tokens_ids))

    # Build an EDLDocument
    aida_doc = EDLDocument(doc_id, words, tokens_ids, entity_mentions, tokenizer)

    # Logs
    if verbose: print(f'Loaded {doc_path} (Nb tokens {len(aida_doc.doc_tokens)})')

    return aida_doc

def load_edl_datasets(jsons_dir, tokenizer):
    edl_docs = []
    filenames = [f for f in listdir(jsons_dir) if isfile(join(jsons_dir, f)) and f.endswith('json')]
    for filename in filenames:
        file_path = join(jsons_dir, filename)
        edl_doc = load_edl_doc(file_path, tokenizer, verbose=True)
        edl_docs.append(edl_doc)
    print(f'Number of docs loaded from {jsons_dir}: {len(edl_docs)}')
    return edl_docs


# Classes
class EDLDocument:
    def __init__(self, doc_id, words, words_ids, entity_mentions, tokenizer):
        assert(len(words) == len(words_ids))
        self.doc_id = doc_id
        self.words = words
        self.words_ids = words_ids
        self.entity_mentions = entity_mentions

        # Append mention_id and mention_text to each entity mention
        self.id2entitymention = {}
        for e in self.entity_mentions:
            start_char_locs = words_ids[e['start_token']].split(':')[-1].split('-')
            end_char_locs = words_ids[e['end_token']].split(':')[-1].split('-')
            e['mention_id'] = f'{doc_id}:{start_char_locs[0]}-{end_char_locs[-1]}'
            e['mention_text'] = words[e['start_token']:e['end_token']+1]
            self.id2entitymention[e['mention_id']] = e

        # Build doc_tokens, self.word_starts_indexes, self.word_ends_indexes
        doc_tokens, word_starts_indexes, word_ends_indexes, start_index = [], [], [], 0
        for w in self.words:
            word_tokens = tokenizer.tokenize(w)
            doc_tokens += word_tokens
            word_starts_indexes.append(start_index)
            word_ends_indexes.append(start_index + len(word_tokens)-1)
            start_index += len(word_tokens)
        self.word_starts_indexes = word_starts_indexes
        self.word_ends_indexes = word_ends_indexes
        assert(len(self.word_starts_indexes) == len(self.words))
        assert(len(self.word_ends_indexes) == len(self.words))

        # Build token_windows, mask_windows, and input_masks
        self.doc_tokens = doc_tokens
        doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, 512, tokenizer)
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)

        # Compute gold_starts, gold_ends
        gold_starts, gold_ends = [], []
        for e in entity_mentions:
            e['start'] = self.word_starts_indexes[e['start_token']]
            e['end'] = self.word_ends_indexes[e['end_token']]
            gold_starts.append(e['start'])
            gold_ends.append(e['end'])
        self.gold_starts = np.array(gold_starts)
        self.gold_ends = np.array(gold_ends)

        # Tensorized Example
        self.tensorized_examples = (
            (np.array(self.token_windows), np.array(self.input_masks), False,
            self.gold_starts, self.gold_ends, np.array([]), np.array(self.mask_windows))
        )

class EDLDocumentPair:
    def __init__(self, doc1, doc2, tokenizer):
        self.doc1, self.doc2 = doc1, doc2

        # All words
        self.words = doc1.words + ['[SEP]'] + doc2.words
        self.words_ids = doc1.words_ids + ['[SEP]'] + doc2.words_ids
        self.doc_tokens = doc1.doc_tokens + ['[SEP]'] + doc2.doc_tokens

        # Build id2entitymention
        self.id2entitymention = {}
        for k in doc1.id2entitymention: self.id2entitymention[k] = doc1.id2entitymention[k]
        for k in doc2.id2entitymention: self.id2entitymention[k] = doc2.id2entitymention[k]

        # All entity mentions
        entity_mentions_1 = copy.deepcopy(doc1.entity_mentions)
        entity_mentions_2 = copy.deepcopy(doc2.entity_mentions)
        for m in entity_mentions_2:
            m['start'] += len(doc1.doc_tokens) + 1
            m['end'] += len(doc1.doc_tokens) + 1
        self.entity_mentions = entity_mentions_1 + entity_mentions_2

        # Build token_windows, mask_windows, and input_masks
        doc_token_ids = tokenizer.convert_tokens_to_ids(self.doc_tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, 512, tokenizer)
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)

        # Build gold_starts, gold_ends
        gold_starts, gold_ends = [], []
        for e in self.entity_mentions:
            gold_starts.append(e['start'])
            gold_ends.append(e['end'])
        self.gold_starts = np.array(gold_starts)
        self.gold_ends = np.array(gold_ends)

        # tensorized_example
        self.tensorized_example = (
            (np.array(self.token_windows), np.array(self.input_masks), False,
            self.gold_starts, self.gold_ends, np.array([]), np.array(self.mask_windows))
        )
