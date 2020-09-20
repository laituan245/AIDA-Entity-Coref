import nltk
import json
import numpy as np

from utils import *
from constants import *
from os.path import join


class SpanishDocument:
    def __init__(self, doc_id, words, mention_starts, mention_ends, cluster_ids, tokenizer):
        self.doc_id = doc_id
        self.words = words
        self.cluster_ids = cluster_ids
        self.num_words = len(self.words)

        doc_tokens = tokenizer.tokenize(' '.join(self.words))
        doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
        self.token_windows, self.mask_windows = \
            convert_to_sliding_window(doc_token_ids, 512, tokenizer)
        self.input_masks = extract_input_masks_from_mask_windows(self.mask_windows)

        # Compute the starting index of each word
        self.word_starts_indexes, self.word_ends_indexes = [], []
        for index, word in enumerate(doc_tokens):
            if not word.startswith('##'):
                self.word_starts_indexes.append(index)
                if index > 0: self.word_ends_indexes.append(index-1)
        self.word_ends_indexes.append(len(doc_tokens)-1)
        assert(len(self.word_starts_indexes) == len(self.words))
        assert(len(self.word_ends_indexes) == len(self.words))

        # Compute gold_starts, gold_ends, and cluster_ids
        self.gold_starts = np.array([self.word_starts_indexes[s] for s in mention_starts])
        self.gold_ends = np.array([self.word_ends_indexes[e] for e in mention_ends])
        self.cluster_ids = np.array(cluster_ids)

        # Compute clusters
        clusters = []
        cluster_nbs = set(cluster_ids)
        for cnb in cluster_nbs:
            if cnb == -1: continue
            _cluster = []
            for g_start, g_end, cid in zip(self.gold_starts, self.gold_ends, self.cluster_ids):
                if cid == cnb:
                    _cluster.append((g_start, g_end))
            clusters.append(_cluster)
        # Appending singleton clusters
        for g_start, g_end, cid in zip(self.gold_starts, self.gold_ends, self.cluster_ids):
            if cid == -1:
                clusters.append([(g_start, g_end)])
        self.clusters = clusters

class SpanishDataset:
    def __init__(self, train, dev, test):
        self.examples, self.tensorized_examples = {}, {}
        for split in [TRAIN, DEV, TEST]:
            if split == TRAIN:
                self.examples[split], self.tensorized_examples[split] = [], []
                is_training = True
                data = train
            if split == DEV:
                self.examples[split], self.tensorized_examples[split] = [], []
                is_training = False
                data = dev
            if split == TEST:
                self.examples[split], self.tensorized_examples[split] = [], []
                is_training = False
                data = test
            for inst in data:
                self.examples[split].append({
                    'doc_key': inst.doc_id,
                    'clusters': inst.clusters
                })
                self.tensorized_examples[split].append(
                    (np.array(inst.token_windows), np.array(inst.input_masks), is_training,
                    inst.gold_starts, inst.gold_ends, inst.cluster_ids, np.array(inst.mask_windows))
                )

def load_spanish_dataset(tokenizer):
    docs = []

    # Read Spanish TSV files
    for fp in SPANISH_TSV_FILES:
        with open(fp, 'r') as f:
            tokens, token_ctx, doc_id = [], 0, 0
            mention_starts, mention_ends, cluster_ids = [], [], []
            for line in f:
                line = line.strip()
                if line == '<DOC>':
                    spanish_doc = \
                        SpanishDocument('SPANISH_{}'.format(doc_id), tokens, mention_starts, mention_ends, cluster_ids, tokenizer)
                    docs.append(spanish_doc)
                    doc_id += 1
                    tokens, token_ctx = [], 0
                    mention_starts, mention_ends, cluster_ids = [], [], []
                else:
                    es = line.split()
                    if es[0].strip() == '[BREAK]':
                        continue
                    if not (es[0].startswith('[[') and es[0].endswith(']]')):
                        tokens.append(es[0])
                        token_ctx += 1
                    else:
                        es[0] = es[0][2:-2]
                        mention_tokens = [es[0]]
                        mention_starts.append(token_ctx)
                        mention_ends.append(token_ctx + len(mention_tokens)-1)
                        for t in mention_tokens: tokens.append(t)
                        if es[1].strip() == 'O': es[1] = -1
                        cluster_ids.append(int(es[1]))
                        token_ctx += len(mention_tokens)

    # Train/dev/test splits
    total_docs = len(docs)
    train_docs = docs[: total_docs // 3]
    dev_docs = docs[total_docs // 3 : 2 * total_docs // 3]
    test_docs = docs[2 * total_docs // 3 : ]

    # Read Spanish txt files
    for fp in SPANISH_TXT_FILES:
        if 'train' in fp: cur_docs = train_docs
        if 'devel' in fp: cur_docs = dev_docs
        if 'test' in fp: cur_docs = test_docs
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0: continue
                if line.startswith('#begin document'):
                    doc_id = line.split()[-1]
                    doc_id = doc_id[:doc_id.find('.tbf.xml')]
                    tokens, mention_infos = [], []
                elif line.startswith('#end document'):
                    mention_starts, mention_ends, cluster_ids = [], [], []
                    # Extract mentions from mention_infos
                    for i in range(len(mention_infos)):
                        for c in mention_infos[i]:
                            if c.startswith('(') and c.endswith(')'):
                                mention_starts.append(i)
                                mention_ends.append(i)
                                cluster_ids.append(int(c[1:-1]))
                            elif c.startswith('('):
                                mention_starts.append(i)
                                cluster_ids.append(int(c[1:]))
                                found = False
                                for j in range(i+1, len(mention_infos)):
                                    if found: break
                                    for c2 in mention_infos[j]:
                                        if c2.endswith(')') and not c2.startswith('(') \
                                        and int(c2[:-1]) == cluster_ids[-1]:
                                            mention_ends.append(j)
                                            found = True
                                            break
                    assert(len(mention_starts) == len(mention_ends))
                    assert(len(mention_starts) == len(cluster_ids))
                    spanish_doc = \
                        SpanishDocument(doc_id, tokens, mention_starts, mention_ends, cluster_ids, tokenizer)
                    cur_docs.append(spanish_doc)
                else:
                    es = line.split('\t')
                    tokens.append(es[1])
                    mention_infos.append(es[-1].split('|'))

    return SpanishDataset(train_docs, dev_docs, test_docs)
