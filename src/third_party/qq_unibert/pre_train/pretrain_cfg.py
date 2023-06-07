# Pretrain file num
PRETRAIN_FILE_NUM = 20
LOAD_DATA_TYPE = 'mem'#'fluid'
# Training params
NUM_FOLDS = 1
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
WARMUP_RATIO = 0.06
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
LR = {'others':5e-4, 'roberta':1e-4, 'newfc_videoreg':5e-4}
LR_LAYER_DECAY = 1.0
PRETRAIN_TASK = ['mlm', 'mvm', 'itm'] #['tag', 'mlm', 'mvm', 'itm']
