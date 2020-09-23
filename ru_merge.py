import os
import json

from os.path import join
from argparse import ArgumentParser

# Constants
NIL_COUNT = 1

# Helper Functions
def read_tab(tab_fp):
    entity_mentions = []
    with open(tab_fp, 'r', encoding='utf8') as f:
        for line in f:
            es = line.strip().split('\t')
            entity_mentions.append({
                'mention_id': es[1],
                'text': es[2],
                'loc': es[3],
                'kb_id': es[4]
            })
    return entity_mentions


# Main code
if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-edl_official', '--edl_official', default='ru.linking.tab')
    parser.add_argument('-edl_freebase', '--edl_freebase', default='ru.linking.freebase.tab')
    parser.add_argument('-o', '--output', default='ru_output.tab')

    args = parser.parse_args()

    # Read Freebase entity linking results
    loc2entity = {}
    entity_mentions_freebase = read_tab(args.edl_freebase)
    for e in entity_mentions_freebase:
        loc2entity[e['loc']] = e

    # Read original Official entity linking results
    buffer = []
    with open(args.edl_official, 'r', encoding='utf8') as f:
        buffer = f.readlines()

    # Combination and output
    fb2nil = {}
    with open(args.output, 'w+', encoding='utf8') as f:
        for line in buffer:
            es = line.strip().split('\t')
            if not es[4].startswith('NIL'):
                f.write(line)
            else:
                fb_id = loc2entity[es[3]]['kb_id']
                if not fb_id in fb2nil:
                    fb2nil
                    nil_str = str(NIL_COUNT)
                    while len(nil_str) < 8: nil_str = '0' + nil_str
                    fb2nil[fb_id] = 'NIL' + nil_str
                    NIL_COUNT += 1
                es[4] = fb2nil[fb_id]
                f.write('{}\n'.format('\t'.join(es)))
