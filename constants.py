from os.path import dirname, join, realpath

PRETRAINED_MODEL = '/shared/nas/data/m1/tuanml2/aida_entity_coref/pretrained/model.pt'

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASIC_CONF_PATH = join(BASE_PATH, 'configs/basic.conf')
TRAIN, DEV, TEST = 'train', 'dev', 'test'

# Related to CorefQA example encoding
SPL_SPEAKER = '[SPL]'
MENTION_START, MENTION_END = '[unused3]', '[unused4]'
SPEAKER_START, SPEAKER_END = '[unused1]', '[unused2]'

# ACE05 Dataset
ACE05 = 'ACE05'

# OntoNote Dataset
ONTONOTE = 'ONTONOTE'
ONTONOTE_BASE_PATH = '/shared/nas/data/m1/tuanml2/datasets/ontonotes/data/'
ONTONOTE_DEV_PATH = '/shared/nas/data/m1/tuanml2/datasets/ontonotes/data/english/dev.english.v4_gold_conll'
ONTONOTE_TEST_PATH = '/shared/nas/data/m1/tuanml2/datasets/ontonotes/data/english/test.english.v4_gold_conll'

# KBP 2016
KBP2016 = 'KBP2016'
KBP2016_CLEAN = 'resources/KBP16/eng_edl16_eval.kbid.clean.wiki.bio'
KBP2016_NOM_CLEAN = 'resources/KBP16/eng_edl16_eval.kbid.nom.clean.wiki.bio'

# KBP 2017
KBP2017 = 'KBP2017'
KBP2017_CLEAN = 'resources/KBP17/eng_edl17_eval.kbid.clean.wiki.bio'
KBP2017_NOM_CLEAN = 'resources/KBP17/eng_edl17_eval.kbid.nom.clean.wiki.bio'

# SPANISH
SPANISH = 'SPANISH'
SPANISH_TSV_FILES = ['resources/ES-Coref/coref_es.tsv',
                     'resources/ES-Coref/coref_dcep_es.tsv']
SPANISH_TXT_FILES = ['resources/ES-Coref/es.train.txt',
                     'resources/ES-Coref/es.devel.txt',
                     'resources/ES-Coref/es.test.txt']
PRETRAINED_SPANISH_MODEL = '/shared/nas/data/m1/tuanml2/aida_entity_coref/spanish/model.pt'
