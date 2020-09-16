import nltk
import json
import numpy as np

from utils import *
from os.path import join

class OneIEDocument:
    def __init__(self, doc_id, sentences, event_mentions, entity_mentions):
        self.doc_id = doc_id
        self.sentences = sentences
        self.words = flatten(sentences)
        self.event_mentions = event_mentions
        self.entity_mentions = entity_mentions
        self.num_words = len(self.words)

        # Post-process self.event_mentions
        for e in self.event_mentions:
            _arguments = []
            for argument in e['arguments']:
                for entity_mention in self.entity_mentions:
                    if entity_mention['id'] == argument['entity_id']:
                        _arguments.append({
                            'text': argument['text'],
                            'role': argument['role'],
                            'entity': entity_mention,
                        })
            assert(len(_arguments) == len(e['arguments']))
            e['arguments'] = _arguments

        # Build id2mentions
        id2mentions = {}
        for e in self.entity_mentions:
            mention_id = e['id']
            entity_id = mention_id[:mention_id.rfind('-')]
            if not entity_id in id2mentions: id2mentions[entity_id] = []
            id2mentions[entity_id].append(e)
        self.id2mentions = id2mentions


class OneIEDataset:
    def __init__(self, data, tokenizer, sliding_window_size = 512):
        '''
            data: A list of GroundTruthDocument
            tokenizer: A transformer Tokenizer
            sliding_window_size: Size of sliding window (for a long document, we split it into overlapping segments)
        '''
        self.data = data

        # Tokenize the documents
        for doc in self.data:
            doc_tokens = tokenizer.tokenize(' '.join(doc.words))
            doc_token_ids = tokenizer.convert_tokens_to_ids(doc_tokens)
            doc.token_windows, doc.mask_windows = \
                convert_to_sliding_window(doc_token_ids, sliding_window_size, tokenizer)
            doc.input_masks = extract_input_masks_from_mask_windows(doc.mask_windows)

            # Compute the starting index of each word
            doc.word_starts_indexes, doc.word_ends_indexes = [], []
            for index, word in enumerate(doc_tokens):
                if not word.startswith('##'):
                    doc.word_starts_indexes.append(index)
                    if index > 0: doc.word_ends_indexes.append(index-1)
            doc.word_ends_indexes.append(len(doc_tokens)-1)
            assert(len(doc.word_starts_indexes) == len(doc.words))
            assert(len(doc.word_ends_indexes) == len(doc.words))

            # Compute gold_starts, gold_ends, clusters, and cluster_ids
            clusters, gold_starts, gold_ends, cluster_ids = [], [], [], []
            for ix, _cluster in enumerate(doc.id2mentions.values()):
                cluster = []
                for e in _cluster:
                    start = doc.word_starts_indexes[e['start']]
                    end = doc.word_ends_indexes[e['end']-1]
                    gold_starts.append(start)
                    gold_ends.append(end)
                    cluster_ids.append(ix)
                    cluster.append((start, end))
                clusters.append(cluster)
            doc.clusters = clusters
            doc.gold_starts = np.array(gold_starts)
            doc.gold_ends = np.array(gold_ends)
            doc.cluster_ids = np.array(cluster_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class OneIEDatasetWrapper:
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
                if len(inst.gold_starts) > 250: continue
                self.examples[split].append({
                    'doc_key': inst.doc_id,
                    'clusters': inst.clusters
                })
                self.tensorized_examples[split].append(
                    (np.array(inst.token_windows), np.array(inst.input_masks), is_training,
                    inst.gold_starts, inst.gold_ends, inst.cluster_ids, np.array(inst.mask_windows))
                )


def load_oneie_dataset(base_path, tokenizer):
    id2split, id2sents = {}, {}

    # Read ground-truth data files
    for split in ['train', 'dev', 'test']:
        path = join(base_path, '{}.oneie.json'.format(split))
        with open(path, 'r', encoding='utf-8') as r:
            for line in r:
                sent_inst = json.loads(line)
                doc_id = sent_inst['doc_id']
                id2split[doc_id] = split
                # Update id2sents
                if not doc_id in id2sents:
                    id2sents[doc_id] = []
                id2sents[doc_id].append(sent_inst)

    # Parse documents one-by-one
    train, dev, test = [], [], []
    for doc_id in id2sents:
        words_ctx = 0
        sents = id2sents[doc_id]
        sentences, event_mentions, entity_mentions = [], [], []
        for sent_index, sent in enumerate(sents):
            sentences.append(sent['tokens'])
            # Parse entity mentions
            for entity_mention in sent['entity_mentions']:
                entity_mention['start'] += words_ctx
                entity_mention['end'] += words_ctx
                entity_mentions.append(entity_mention)
            # Parse event mentions
            for event_mention in sent['event_mentions']:
                event_mention['sent_index'] = sent_index
                event_mention['trigger']['start'] += words_ctx
                event_mention['trigger']['end'] += words_ctx
                event_mentions.append(event_mention)
            # Update words_ctx
            words_ctx += len(sent['tokens'])
        doc = OneIEDocument(doc_id, sentences, event_mentions, entity_mentions)
        split = id2split[doc_id]
        if len(doc.entity_mentions) == 0: continue
        if split == 'train': train.append(doc)
        if split == 'dev': dev.append(doc)
        if split == 'test': test.append(doc)

    # Convert to Document class
    train, dev, test = OneIEDataset(train, tokenizer), OneIEDataset(dev, tokenizer), OneIEDataset(test, tokenizer)

    return OneIEDatasetWrapper(train, dev, test)
