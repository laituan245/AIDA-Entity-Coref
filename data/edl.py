import os
import copy

from os import listdir
from os.path import isfile, join
from utils import *

class EDLDocument:
    def __init__(self, doc_id, words, words_ids, entity_mentions, tokenizer):
        assert(len(words) == len(words_ids))
        self.doc_id = doc_id
        self.words = words
        self.words_ids = words_ids
        self.entity_mentions = entity_mentions

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
