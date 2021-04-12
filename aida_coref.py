import os
import json

from os.path import join
from constants import *
from utils import *
from data import AIDADataset, AIDADocument
from edl_inference import inference
from argparse import ArgumentParser

# Constants
PRETRAINED_MODEL = 'model.pt'
CONFIG_NAME = 'basic'
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
            count[official_id] = count.get(official_id, 0) + 1
        label = max(count, key=lambda k: count[k])
        clusterlabels.append(label)

    # label2cluster
    label2cluster = {}
    for ix, l in enumerate(clusterlabels):
        label2cluster[l] = ix
    for _id in id2entity:
        official_id = id2entity[_id]['official_kb_id']
        if official_id.startswith('NIL'): continue
        if official_id != clusterlabels[id2cluster[_id]]:
            if not official_id in label2cluster:
                label2cluster[official_id] = len(clusterlabels)
                clusterlabels.append(official_id)
            id2cluster[_id] = label2cluster[official_id]

    return id2cluster, clusterlabels

# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-edl_official', '--edl_official', default='/shared/nas/data/m1/manling2/aida_docker_test/edl_en_zh/testdata_zh/edl/zh.linking.tab')
    parser.add_argument('-edl_freebase', '--edl_freebase', default='/shared/nas/data/m1/manling2/aida_docker_test/edl_en_zh/testdata_zh/edl/zh.linking.freebase.tab')
    parser.add_argument('-l', '--json_dir', default='/shared/nas/data/m1/manling2/aida_docker_test/edl_en_zh/testdata_zh/edl/json/')
    parser.add_argument('-o', '--output', default='samples/edl_output.tab')

    args = parser.parse_args()

    # Read Tab files
    loc2ctx, entity_mentions, id2freebaseid, id2officialkbid, id2type = {}, [], {}, {}, {}
    entity_mentions_official = read_tab(args.edl_official)
    entity_mentions_freebase = read_tab(args.edl_freebase)
    for e in entity_mentions_official:
        loc = e['doc_id'], e['start_char'], e['end_char']
        if not loc in loc2ctx: loc2ctx[loc] = len(loc2ctx)
    for e in entity_mentions_freebase:
        loc = e['doc_id'], e['start_char'], e['end_char']
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
        entity_mentions[ctx]['mention_id'] = 'ZH_MENTION_{}'.format(ctx_str)

    # Combine entity linking results
    for e1 in entity_mentions_freebase:
        loc = e1['doc_id'], e1['start_char'], e1['end_char']
        ctx = loc2ctx[loc]
        entity_mentions[ctx]['freebase_id'] = e1['kb_id']
        entity_mentions[ctx]['text'] = e1['text']
        assert(entity_mentions[ctx]['start_char'] == e1['start_char'])
        assert(entity_mentions[ctx]['end_char'] == e1['end_char'])
        id2freebaseid['{}:{}-{}'.format(loc[0], loc[1], loc[2])] = e1['kb_id']
        id2type['{}:{}-{}'.format(loc[0], loc[1], loc[2])] = e1['entity_type']

    for e2 in entity_mentions_official:
        loc = e2['doc_id'], e2['start_char'], e2['end_char']
        ctx = loc2ctx[loc]
        entity_mentions[ctx]['official_kb_id'] = e2['kb_id']
        assert(entity_mentions[ctx]['start_char'] == e2['start_char'])
        assert(entity_mentions[ctx]['end_char'] == e2['end_char'])
        assert(entity_mentions[ctx]['text'] == e2['text'])
        id2officialkbid['{}:{}-{}'.format(loc[0], loc[1], loc[2])] = e2['kb_id']
        id2type['{}:{}-{}'.format(loc[0], loc[1], loc[2])] = e2['entity_type']

    # Extract predicted coreferential pairs
    assert(PRETRAINED_MODEL and os.path.isfile(PRETRAINED_MODEL))
    predicted_pairs, id2entity = inference(args.json_dir, pretrained_model=PRETRAINED_MODEL)
    for _id in id2freebaseid:
        id2entity[_id]['freebase_id'] = id2freebaseid[_id]
    for _id in id2officialkbid:
        id2entity[_id]['official_kb_id'] = id2officialkbid[_id]

    # Use entity linking results to remove some of the predicted pairs
    filtered_pairs = set()
    for e1, e2 in predicted_pairs:
        should_insert = True
        fb_id_i, fb_id_j = id2freebaseid[e1], id2freebaseid[e2]
        officialkb_id_i, officialkb_id_j = id2officialkbid[e1], id2officialkbid[e2]
        if not fb_id_i.startswith('NIL') and not fb_id_j.startswith('NIL') and fb_id_i != fb_id_j:
            should_insert = False
        if not officialkb_id_i.startswith('NIL') and not officialkb_id_j.startswith('NIL') \
        and officialkb_id_i != officialkb_id_j:
            should_insert = False
        if id2type[e1] != id2type[e2]:
            should_insert = False
        if should_insert:
            filtered_pairs.add((e1, e2))
            filtered_pairs.add((e2, e1))

    # Use entity linking results to add pairs across docs
    entity_mention_ids = list(id2entity.keys())
    for index1 in range(len(entity_mention_ids)):
        for index2 in range(index1+1, len(entity_mention_ids)):
            e1 = entity_mention_ids[index1]
            e2 = entity_mention_ids[index2]
            fb_id_i, fb_id_j = id2freebaseid[e1], id2freebaseid[e2]
            if not fb_id_i.startswith('NIL') and not fb_id_j.startswith('NIL') and fb_id_i == fb_id_j:
                filtered_pairs.add((e1, e2))
                filtered_pairs.add((e2, e1))

    # Determine final clusters
    id2cluster, clusterlabels = cluster_from_pairs(id2entity, filtered_pairs)

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
            es[4] = clusterlabels[id2cluster[f'{doc_id}:{start_char}-{end_char}']]
            line = '\t'.join(es)
            f.write(line)
