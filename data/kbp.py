import os

from os import listdir
from os.path import isfile, join
from utils import *

class KBPDocument:
    def __init__(self, doc_id, words, entity_mentions, tokenizer):
        self.doc_id = doc_id
        self.words = words
        self.entity_mentions = entity_mentions

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

        # Build id2mentions
        kbid2mentions = {}
        for e in self.entity_mentions:
            kbid = e[-1]
            if not kbid in kbid2mentions: kbid2mentions[kbid] = []
            kbid2mentions[kbid].append(e)

        # Compute gold_starts, gold_ends, clusters, and cluster_ids
        clusters, gold_starts, gold_ends, cluster_ids = [], [], [], []
        for ix, _cluster in enumerate(kbid2mentions.values()):
            cluster = []
            for e in _cluster:
                start = self.word_starts_indexes[e[1]]
                end = self.word_ends_indexes[e[2]]
                gold_starts.append(start)
                gold_ends.append(end)
                cluster_ids.append(ix)
                cluster.append((start, end))
            clusters.append(cluster)
        self.clusters = clusters
        self.gold_starts = np.array(gold_starts)
        self.gold_ends = np.array(gold_ends)
        self.cluster_ids = np.array(cluster_ids)

class KBPDataset:
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


def load_kbp_dataset(clean, nom_clean, tokenizer):
    id2words_clean, entity_mentions = read_bio_file(clean)
    id2words_nom_clean, nom_mentions = read_bio_file(nom_clean)
    all_mentions = entity_mentions + nom_mentions
    doc_ids = set(id2words_clean.keys()).intersection(id2words_nom_clean.keys())

    # Build id2words
    id2words = {}
    for doc_id in doc_ids:
        tokens_1 = id2words_clean[doc_id]
        tokens_2 = id2words_nom_clean[doc_id]
        if len(tokens_1) != len(tokens_2): continue
        equal = True
        for token_1, token_2 in zip(tokens_1, tokens_2):
            if token_1 != token_2:
                equal = False
                break
        if equal: id2words[doc_id] = tokens_1

    # Create KBPDocuments
    documents = []
    for doc_id in id2words:
        tokens = id2words[doc_id]
        doc_entities = []
        for e in all_mentions:
            if e[0] == doc_id:
                doc_entities.append(e)
        documents.append(KBPDocument(doc_id, tokens, doc_entities, tokenizer))

    # Train/dev/test splits
    total_docs = len(documents)
    train_docs = documents[: total_docs // 3]
    dev_docs = documents[total_docs // 3 : 2 * total_docs // 3]
    test_docs = documents[2 * total_docs // 3 : ]

    return KBPDataset(train_docs, dev_docs, test_docs)
