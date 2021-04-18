import time
import torch
import os

from transformers import *
from models import CorefModel
from data.edl import EDLDocument, EDLDocumentPair, load_edl_datasets
from utils import *

def extract_predicted_pairs(entity_mentions, preds):
    preds = [x.cpu().data.numpy() for x in preds]
    _, _, top_antecedents, antecedent_scores = preds
    predicted_antecedents = get_predicted_antecedents(top_antecedents, antecedent_scores)

    predicted_pairs = set()
    for ix in range(len(entity_mentions)):
        if predicted_antecedents[ix] >= 0:
            antecedent_idx = predicted_antecedents[ix]
            mention_1 = entity_mentions[ix]
            mention_2 = entity_mentions[antecedent_idx]
            predicted_pairs.add((mention_1['mention_id'], mention_2['mention_id']))

    return predicted_pairs

def inference(jsons_dir, doc2entitymentions, config_name='basic',
              pretrained_model='/shared/nas/data/m1/tuanml2/edl_coref/trained_model/cn_en_entity_coref.pt'):
    # Prepare the config, the tokenizer, and the model
    configs = prepare_configs(config_name)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    model = CorefModel(configs)
    print('Prepared tokenizer and model')
    if pretrained_model and os.path.isfile(pretrained_model):
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded pretrained ckpt')
    else:
        print('No pretrained ckpt reloaded')

    # Load data for inference
    edl_docs = load_edl_datasets(jsons_dir, doc2entitymentions, tokenizer)
    id2entity = {}
    for doc in edl_docs:
        for e in doc.entity_mentions:
            id2entity[e['mention_id']] = e

    # Apply the coref model
    print('Applying the coref model')
    start_time = time.time()
    all_predicted_pairs = set()
    for i in range(len(edl_docs)):
        inst = edl_docs[i]
        with torch.no_grad():
            preds = model(*inst.tensorized_example)[1]
            predicted_pairs = extract_predicted_pairs(inst.entity_mentions, preds)
            for (p1, p2) in predicted_pairs:
                all_predicted_pairs.add((p1, p2))

    print(f'Running time took {time.time() - start_time} seconds')
    return all_predicted_pairs, id2entity


if __name__ == '__main__':
    inference('/shared/nas/data/m1/manling2/aida_docker_test/edl_en_zh/testdata_crosslingual/json', 'basic')
