import os
import json
import utils
import random
import itertools
import numpy as np

from constants import *
from os.path import join

MAX_SEGMENT_LEN = 384
MAX_TRAINING_SENTENCES = 3

class _Dataset(object):
    def __init__(self, base_path, tokenizer, language, verbose = True):
        self.language = language
        self.tokenizer = tokenizer

        # Process train/dev/test files
        self.examples, self.tensorized_examples, self.sentence_offsets = {}, {}, {}
        self.gold, self.subtoken_maps = {}, {}
        for split in [TRAIN, DEV, TEST]:
            path = join(base_path, language, 'independent', '{}.{}.{}.jsonlines'.format(split, language, MAX_SEGMENT_LEN))
            with open(path) as f:
                self.examples[split] = [json.loads(jsonline) for jsonline in f.readlines()]
                self.examples[split] = [e for e in self.examples[split] if len(e['clusters']) > 0]

                # Tensorize the examples
                is_training = (split == 'train')
                self.tensorized_examples[split], self.sentence_offsets[split] = [], []
                for example in self.examples[split]:
                    example, sentence_offset = self.tensorize_example(example, is_training)
                    self.tensorized_examples[split].append(example)
                    self.sentence_offsets[split].append(sentence_offset)

    def tensorize_example(self, example, is_training):
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in utils.flatten(clusters))
        gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
          for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        assert (len(sentences) == len(speakers))
        assert (num_words == len(utils.flatten(speakers)))
        sentence_map = example['sentence_map']

        max_sentence_length = MAX_SEGMENT_LEN
        text_len = np.array([len(s) for s in sentences])

        input_ids, input_mask = [], []
        for i, sentence in enumerate(sentences):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        assert(num_words == np.sum(input_mask))

        doc_key = example['doc_key']
        self.gold[doc_key] = example["clusters"]
        self.subtoken_maps[doc_key] = example['subtoken_map']

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors = (input_ids, input_mask, text_len, is_training, gold_starts, gold_ends, cluster_ids, sentence_map)
        if is_training and len(sentences) > MAX_TRAINING_SENTENCES:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors, 0

    def truncate_example(self, input_ids, input_mask, text_len, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, sentence_offset=0):
        num_sentences = input_ids.shape[0]
        assert(num_sentences > MAX_TRAINING_SENTENCES)

        sentence_offset = random.randint(0, num_sentences - MAX_TRAINING_SENTENCES) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + MAX_TRAINING_SENTENCES].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + MAX_TRAINING_SENTENCES, :]
        input_mask = input_mask[sentence_offset:sentence_offset + MAX_TRAINING_SENTENCES, :]
        text_len = text_len[sentence_offset:sentence_offset + MAX_TRAINING_SENTENCES]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return (input_ids, input_mask, text_len, is_training,  gold_starts, gold_ends, cluster_ids, sentence_map), sentence_offset

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
          starts, ends = zip(*mentions)
        else:
          starts, ends = [], []
        return np.array(starts), np.array(ends)

class OntoNoteDataset(_Dataset):
    def __init__(self, base_path, tokenizer, language, verbose = True):
        super(OntoNoteDataset, self).__init__(base_path, tokenizer, language, verbose)

        # Convert to CorefQA format
        for split in [TRAIN, DEV, TEST]:
            nb_examples = len(self.tensorized_examples[split])
            for i in range(nb_examples):
                doc_key = self.examples[split][i]['doc_key']
                example = self.tensorized_examples[split][i]
                speakers = self.examples[split][i]["speakers"]
                if split == TRAIN:
                    sentence_offset = self.sentence_offsets[split][i]
                    speakers = speakers[sentence_offset:sentence_offset + MAX_TRAINING_SENTENCES]
                flattened_speaker = list(itertools.chain.from_iterable(speakers))
                example = list(example)
                example.append(flattened_speaker)
                example.append(doc_key)
                example = tuple(example)
                self.tensorized_examples[split][i] = self.convert_corefqa(*example)

                # Update clusters
                gold_starts, gold_ends = self.tensorized_examples[split][i][3:5]
                cluster_ids = self.tensorized_examples[split][i][5]
                clusters = []
                for _ in range(len(set(cluster_ids))):
                    clusters.append([])
                for g_start, g_end, c_id in zip(gold_starts, gold_ends, cluster_ids):
                    clusters[int(c_id)-1].append((int(g_start), int(g_end)))
                self.examples[split][i]['clusters'] = clusters


    def convert_corefqa(self, input_ids, input_mask, text_len, is_training,
                        gold_starts, gold_ends, cluster_ids, sentence_map,
                        flattened_speaker, doc_key):
        tokenizer = self.tokenizer

        subtoken_map = self.subtoken_maps[doc_key]

        # Build flattened_sentence
        tokens = []
        for itx, input_id in enumerate(input_ids):
            tokens += tokenizer.convert_ids_to_tokens(input_id)[:text_len[itx]]
        flattened_sentence = [[token, [], -1] for token in tokens]
        for itx, (g_start, g_end) in enumerate(list(zip(gold_starts, gold_ends))):
            for i in range(g_start, g_end+1):
                flattened_sentence[i][1].append(int(itx))

        for i in range(len(flattened_sentence)):
            flattened_sentence[i][2] = subtoken_map[i]
        flattened_sentence = [tuple(a) for a in flattened_sentence]

        # Build extended_sentence (extended with speaker infos)
        prev_speaker = SPL_SPEAKER
        extended_sentence, new_sentence_map  = [], []
        for idx, (token, cids1, subtoken) in enumerate(flattened_sentence):
            cur_speaker = flattened_speaker[idx]
            cur_sentence_map = sentence_map[idx]
            if cur_speaker != prev_speaker and cur_speaker != SPL_SPEAKER:
               speaker_tokens = [SPEAKER_START] + tokenizer.tokenize(cur_speaker) + [SPEAKER_END]
               speaker_tokens = [(s, [], -1) for s in speaker_tokens]
               extended_sentence.extend(speaker_tokens)
               new_sentence_map.extend([cur_sentence_map] * len(speaker_tokens))
            extended_sentence.append((token, cids1, subtoken))
            new_sentence_map.append(cur_sentence_map)
            prev_speaker = cur_speaker
        sentence_map = new_sentence_map
        extended_sentence = extended_sentence[1:-1]
        sentence_map = sentence_map[1:-1]

        # Re-compute subtoken_map
        subtoken_map = [a[-1] for a in extended_sentence]
        self.subtoken_maps[doc_key] = subtoken_map

        # Re-compute gold_starts, gold_ends, cluster_ids
        nb_gold_mentions = len(cluster_ids)
        gold_starts, gold_ends, c2loc = [], [], {}
        for itx in range(nb_gold_mentions):
            c2loc[itx] = {'start': None, 'end': None}
        for i in range(len(extended_sentence)):
            for itx in extended_sentence[i][1]:
                c2loc[itx]['end'] = i
                if c2loc[itx]['start'] is None:  c2loc[itx]['start'] = i
        for itx in range(nb_gold_mentions):
            gold_starts.append(c2loc[itx]['start'])
            gold_ends.append(c2loc[itx]['end'])
        gold_starts, gold_ends = np.array(gold_starts), np.array(gold_ends)

        # input_ids
        tokens = [tokenizer.convert_tokens_to_ids([w[0]])[0] for w in extended_sentence]
        input_ids, mask_windows = utils.convert_to_sliding_window(tokens, MAX_SEGMENT_LEN, tokenizer)
        input_ids = np.array(input_ids)

        # input_masks
        input_masks = []
        for mask_window in mask_windows:
            subtoken_count = utils.listRightIndex(mask_window, -3) + 1
            input_masks.append([1] * subtoken_count + [0] * (len(mask_window) - subtoken_count))
        input_masks = np.array(input_masks)

        mask_windows = np.array(mask_windows)

        return input_ids, input_masks, is_training, gold_starts, gold_ends, cluster_ids, mask_windows
