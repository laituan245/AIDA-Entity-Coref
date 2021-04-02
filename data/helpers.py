from constants import *
from data.oneie import load_oneie_dataset
from data.ontonotes import OntoNoteDataset
from data.kbp import load_kbp_dataset
from data.spanish import load_spanish_dataset

class DatasetsWrapper:
    def __init__(self, datasets):
        self.examples, self.tensorized_examples = {}, {}
        for split in [TRAIN, DEV, TEST]:
            self.examples[split] = []
            self.tensorized_examples[split] = []
            for dataset in datasets:
                for e in dataset.examples[split]: self.examples[split].append(e)
                for e in dataset.tensorized_examples[split]: self.tensorized_examples[split].append(e)

def prepare_dataset(name, tokenizer):
    if name == ENGLISH_ONTONOTE:
        return OntoNoteDataset(ONTONOTE_BASE_PATH, tokenizer, 'english')
    if name == CHINESE_ONTONOTE:
        return OntoNoteDataset(ONTONOTE_BASE_PATH, tokenizer, 'chinese')
    if name == ACE05:
        return load_oneie_dataset('resources/ACE05-E', tokenizer)
    if name == KBP2016:
        return load_kbp_dataset(KBP2016_CLEAN, KBP2016_NOM_CLEAN, tokenizer)
    if name == KBP2017:
        return load_kbp_dataset(KBP2017_CLEAN, KBP2017_NOM_CLEAN, tokenizer)
    if name == SPANISH:
        return load_spanish_dataset(tokenizer)

def combine_datasets(datasets):
    return DatasetsWrapper(datasets)
