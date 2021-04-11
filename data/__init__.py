import json

from utils import *
from data.ontonotes import OntoNoteDataset
from data.helpers import prepare_dataset, combine_datasets
from data.aida import AIDADataset, AIDADocument
from data.edl import EDLDocument

def load_oneie_doc(doc_path, tokenizer, verbose=True):
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
    if verbose: print('Loaded {}'.format(doc_path))

    return aida_doc
