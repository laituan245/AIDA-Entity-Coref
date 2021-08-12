import os
import json

from os.path import join
from constants import *
from utils import *
from data import AIDADataset, AIDADocument
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.special import softmax

# Constants
PRETRAINED_MODEL = 'model.pt'
CONFIG_NAME = 'spanbert_large'
NIL_COUNT = 1

# Helper Function
def DFS(v, pairs, visited, scc):
    visited.add(v)
    scc.add(v)
    for u1, u2 in pairs:
        if u1 == v and not u2 in visited:
            DFS(u2, pairs, visited, scc)

def cluster_from_pairs(id2entity, pairs):
    global NIL_COUNT
    clusters, id2cluster = [], {}
    visited = set()
    for e in id2entity:
        if not e in visited:
            scc = set()
            DFS(e, pairs, visited, scc)
            clusters.append([id2entity[e_scc] for e_scc in scc])
            for e_scc in scc: id2cluster[e_scc] = len(clusters) - 1

    # Build clusterlabels
    clusterlabels = []
    for c in clusters:
        count = {}
        for e in c:
            official_id = e['official_kb_id']
            if official_id.startswith('NIL'): official_id = 'NIL'
            if official_id == 'NIL':
                continue
            count[official_id] = count.get(official_id, 0) + 1

        if len(count) == 0:
            nil_str = str(NIL_COUNT)
            while len(nil_str) < 8: nil_str = '0' + nil_str
            label = 'NIL' + nil_str
            NIL_COUNT += 1
        else:
            label = max(count, key=lambda k: count[k])
        clusterlabels.append(label)

    return clusters, id2cluster, clusterlabels

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-edl_official', '--edl_official', default='samples/en.linking.tab')
    parser.add_argument('-edl_freebase', '--edl_freebase', default='samples/en.linking.freebase.tab')
    parser.add_argument('-l', '--ltf_dir', default='samples/ltf')
    parser.add_argument('-o', '--output', default='samples/output.tab')
    parser.add_argument('-d', '--debug', default=False)

    args = parser.parse_args()

    # Read LTF files
    doc2tokens = read_ltf_files(args.ltf_dir)

    # Read Tab files
    loc2ctx, entity_mentions = {}, []
    entity_mentions_official = read_tab(args.edl_official)
    entity_mentions_freebase = read_tab(args.edl_freebase)
    for e in entity_mentions_official:
        loc = e['doc_id'], e['start_char'], e['end_char']
        assert(e['doc_id'] in doc2tokens)
        if not loc in loc2ctx: loc2ctx[loc] = len(loc2ctx)
    for e in entity_mentions_freebase:
        loc = e['doc_id'], e['start_char'], e['end_char']
        assert(e['doc_id'] in doc2tokens)
        assert(loc in loc2ctx)
    for loc in loc2ctx: entity_mentions.append({})
    for loc in loc2ctx:
        ctx = loc2ctx[loc]
        doc_id, start_char, end_char = loc
        # Compute mention_id
        ctx_str = str(ctx)
        while len(ctx_str) < 7: ctx_str = '0' + ctx_str
        entity_mentions[ctx]['doc_id'] = doc_id
        entity_mentions[ctx]['start_char'] = start_char
        entity_mentions[ctx]['end_char'] = end_char
        entity_mentions[ctx]['mention_id'] = 'EN_MENTION_{}'.format(ctx_str)

    # Combine entity linking results
    for e1 in entity_mentions_freebase:
        loc = e1['doc_id'], e1['start_char'], e1['end_char']
        ctx = loc2ctx[loc]
        entity_mentions[ctx]['freebase_id'] = e1['kb_id']
        entity_mentions[ctx]['text'] = e1['text']
        assert(entity_mentions[ctx]['start_char'] == e1['start_char'])
        assert(entity_mentions[ctx]['end_char'] == e1['end_char'])

    for e2 in entity_mentions_official:
        loc = e2['doc_id'], e2['start_char'], e2['end_char']
        ctx = loc2ctx[loc]
        entity_mentions[ctx]['official_kb_id'] = e2['kb_id']
        assert(entity_mentions[ctx]['start_char'] == e2['start_char'])
        assert(entity_mentions[ctx]['end_char'] == e2['end_char'])
        assert(entity_mentions[ctx]['text'] == e2['text'])

    # Prepare the config, the tokenizer, and the model
    configs = prepare_configs(CONFIG_NAME)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    model = CorefModel(configs)
    if PRETRAINED_MODEL and os.path.isfile(PRETRAINED_MODEL):
        if torch.cuda.is_available():
            checkpoint = torch.load(PRETRAINED_MODEL)
        else:
            checkpoint = torch.load(PRETRAINED_MODEL, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reload the model')
    elif not args.debug:
        raise Exception('A trained model is required')

    # Create AIDADocument
    docs = []
    for doc_id in doc2tokens:
        tokens = doc2tokens[doc_id]
        words = [t[-1] for t in tokens]
        start2word, end2word = {}, {}
        for ix, (start, end, _) in enumerate(tokens):
            start2word[int(start)] = ix
            end2word[int(end)] = ix
        doc_mentions = []
        for e in entity_mentions:
            if e['doc_id'] == doc_id:
                e['start_token'] = start2word[int(e['start_char'])]
                e['end_token'] = end2word[int(e['end_char'])]
                doc_mentions.append(e)
        if len(words) == 0: continue
        docs.append(AIDADocument(doc_id, words, doc_mentions, tokenizer))
    dataset = AIDADataset(docs)

    scores_logs = open('scores_logs.txt', 'w+')
    scores_logs.write('Current_Span\tAntecedent_Candidate\tScore\n')
    # Apply the coref model
    with torch.no_grad():
        doc2id2cluster, doc2clusterlabels = {}, {}
        for doc_index, tensorized_example in tqdm(enumerate(dataset.tensorized_examples[TEST])):
            entities = dataset.data[doc_index].entity_mentions
            if len(entities) == 0: continue
            preds = model(*tensorized_example)[1]
            preds = [x.cpu().data.numpy() for x in preds]
            mention_starts, mention_ends, top_antecedents, top_antecedent_scores = preds
            predicted_antecedents = get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            top_antecedent_scores = softmax(top_antecedent_scores, axis=-1)

            # Print out scores
            for ix, entity in enumerate(entities):
                # Sanity checks
                assert(entity['start'] == mention_starts[ix])
                assert(entity['end'] == mention_ends[ix])
                # Current span
                doc_id = entity['doc_id']
                start_char = entity['start_char']
                end_char = entity['end_char']
                cur_span_id = f'{doc_id}:{start_char}-{end_char}'
                # No antecedent
                scores_logs.write(f'{cur_span_id}\tNo_Antecedent\t{top_antecedent_scores[ix, 0]}\n')
                for jx in range(ix):
                    prev_doc_id = entities[jx]['doc_id']
                    prev_start_char = entities[jx]['start_char']
                    prev_end_char = entities[jx]['end_char']
                    prev_span_id = f'{prev_doc_id}:{prev_start_char}-{prev_end_char}'
                    scores_logs.write(f'{cur_span_id}\t{prev_span_id}\t{top_antecedent_scores[ix, jx+1]}\n')

            # Build id2entity
            id2entity = {}
            for entity in entities:
                id2entity[entity['mention_id']] = entity

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

            # Initialize predicted_pairs
            predicted_pairs = set()
            for c in predicted_clusters:
                if len(c) <= 1: continue
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        should_insert = True
                        predicted_pairs.add((c[i]['mention_id'], c[j]['mention_id']))
                        predicted_pairs.add((c[j]['mention_id'], c[i]['mention_id']))

            # Sanity check
            for e, m_start, m_end in zip(entities, mention_starts, mention_ends):
                assert(e['start'] == m_start and e['end'] == m_end)

            # First cluster
            clusters, id2cluster, clusterlabels = cluster_from_pairs(id2entity, predicted_pairs)
            doc2id2cluster[dataset.data[doc_index].doc_id] = id2cluster
            doc2clusterlabels[dataset.data[doc_index].doc_id] = clusterlabels

        # Output
        lines = []
        with open(args.edl_official, 'r', encoding='utf8') as f:
            for line in f: lines.append(line)
        with open(args.output, 'w+', encoding='utf8') as f:
            for line in lines:
                es = line.split('\t')
                doc_id, text_loc = es[3].split(':')
                start_char, end_char = text_loc.split('-')
                entity_id = entity_mentions[loc2ctx[(doc_id, start_char, end_char)]]['mention_id']
                es[1] = entity_id
                clusterlabels = doc2clusterlabels[doc_id]
                id2cluster = doc2id2cluster[doc_id]
                es[4] = clusterlabels[id2cluster[entity_id]]
                line = '\t'.join(es)
                f.write(line)

    scores_logs.close()
