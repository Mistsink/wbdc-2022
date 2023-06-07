import logging
import random
import shutil
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,get_constant_schedule_with_warmup

from category_id_map import lv2id_to_lv1id


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.log_save:
        handler = logging.FileHandler(args.log_path, encoding='UTF-8')
        logger.addHandler(handler)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    '''
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    '''
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ['bias',  'LayerNorm.weight','LayerNorm.weight']
    model_param = list(module.named_parameters())
    
    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'roberta':
            bert_param_optimizer.append((name, para))
        else:
            #print(name)
            other_param_optimizer.append((name, para))
    optimizer_grouped_parameters = [
        # bert other module //and 'encoder.layer' not in n
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay) ],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay) ],
         "weight_decay": 0.0, 'lr': args.learning_rate},
          # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay) ],
         "weight_decay": args.weight_decay, 'lr': args.linear_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.linear_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * args.warmup_ratio,num_training_steps=args.max_steps)
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * args.warmup_ratio,num_training_steps=args.max_steps)
    #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * args.warmup_ratio)

    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=4e-5)

    return optimizer, scheduler, swa_model, swa_scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results

def validate_and_save_model(args, val_dataloader, model, best_score, epoch, validate_fn, logger, model_save_path):
    loss, results = validate_fn(model, val_dataloader)
    results = {k: round(v, 4) for k, v in results.items()}
    logger.info(f"Epoch {epoch+1} : Loss {loss:.3f}, F1 {results['mean_f1']}")

        # 5. save checkpoint
    mean_f1 = results['mean_f1']
    if mean_f1 > best_score['val']:
        best_score['val'] = mean_f1
        state_dict = model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
                       model_save_path)


def setup_cp_files(args):
    os.makedirs(args.savedmodel_path, exist_ok=True)

    dst_file_list = ['config', 'data_helper', 'main-k','inference-k', 'main','model_gdy', 'model', 'util', 'category_id_map', 'evaluate', 'model_uni','inference-k-newswa', 'main-k-Copy1']
    for dst_file in dst_file_list:
        shutil.copyfile(f'./{dst_file}.py', f'{args.savedmodel_path}{dst_file}.py')

