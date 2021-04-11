import time
import torch

from transformers import *
from models import CorefModel
from data.edl import EDLDocument, EDLDocumentPair, load_edl_datasets
from utils import evaluate, RunningAverage, prepare_configs

def inference(jsons_dir, config_name='basic', pretrained_model=None):
    # Prepare the config, the tokenizer, and the model
    configs = prepare_configs(config_name)
    tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], do_basic_tokenize=False)
    model = CorefModel(configs)
    print('Prepared tokenizer and model')
    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded pretrained ckpt')
    else:
        print('No pretrained ckpt reloaded')

    # Load data for inference
    edl_docs = load_edl_datasets(jsons_dir, tokenizer)
    dummy_doc = EDLDocument('dummy_edl_doc', [], [], [], tokenizer)
    edl_docs = [dummy_doc] + edl_docs

    # Apply the coref model
    print('Applying the coref model')
    start_time = time.time()
    for i in range(len(edl_docs)-1):
        inst = EDLDocumentPair(edl_docs[i], edl_docs[i+1], tokenizer)
        with torch.no_grad():
            [_, _, top_antecedents, antecedent_scores] = model(*inst.tensorized_example)[1]
    print(f'Running time took {time.time() - start_time} seconds')


if __name__ == '__main__':
    inference('/shared/nas/data/m1/manling2/aida_docker_test/edl_en_zh/testdata_zh/edl/json', 'basic')
