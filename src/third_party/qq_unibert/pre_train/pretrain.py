from config import parse_args
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import torch
from utils import set_random_seed
from create_optimizer import create_optimizer
from qq_uni_model import QQUniModel
from qq_dataset import create_dataloaders
from pretrain_cfg import *
from model_cfg import *
from data_cfg import *
import numpy as np
from imp import reload
import logging
import os
import time
import gc
import psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"


reload(logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(
            f"train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)


gc.enable()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(SEED)


ARG = parse_args()

scaler = None
autocast = None

def get_pred_and_loss(model, inputs, task=None):
    pred, emb, loss = model(inputs, task)
    return pred, emb, loss


def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    model.eval()
    loss_l, emb_l, vid_l = [], [], []

    with torch.no_grad():
        batch_num = 0
        for item in tqdm(data_loader):
            if autocast != None:
                with autocast():
                    _, emb, loss = get_pred_and_loss(model, item, task=PRETRAIN_TASK)
            else:
                _, emb, loss = get_pred_and_loss(model, item, task=PRETRAIN_TASK)


            if loss is not None:
                loss_l.append(loss.to("cpu"))

            emb_l += emb.to("cpu").tolist()

            # vid_l.append(item['vid'][0].numpy())

            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
            batch_num += 1

    return np.mean(loss_l), np.array(emb_l), []


def train(model, model_path,
          train_loader, val_loader,
          optimizer, get_pred_and_loss, scheduler=None,
          num_epochs=5):
    best_val_loss, best_epoch, step = None, 0, 0
    start = time.time()

    for epoch in range(num_epochs):
        for item in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            if autocast != None:
                with autocast():
                    _, _, loss = get_pred_and_loss(model, item)

                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
            else:
                _, _, loss = get_pred_and_loss(model, item)
                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            if step == 20 or (step % 500 == 0 and step > 0):
                # Evaluate the model on val_loader.
                elapsed_seconds = time.time() - start

                val_loss, _, _ = eval(
                    model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=10000)

                # 中途 save 一份 model 是给 job2 使用的
                if step == 233000:
                    torch.save(model.state_dict(), '../../../../data/model_pretrain_job2.pth')


                improve_str = ''
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                logging.info(
                    f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}|time={elapsed_seconds:0.3}s" + improve_str)

                start = time.time()
            step += 1

    return best_val_loss


# Show config
logging.info("Start")
for fname in ['pretrain', 'model', 'data']:
    logging.info('=' * 66)
    with open(f'./{fname}_cfg.py') as f:
        logging.info(f"Config - {fname}:" + '\n' + f.read().strip())

list_val_loss = []
logging.info(f"Model_type = {MODEL_TYPE}")


for fold in range(NUM_FOLDS):
    logging.info('=' * 66)
    model_path = f"../../../../data/model_pretrain_job1.pth"
    logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={SEED+fold}")

    set_random_seed(SEED + fold)
    ARG.seed  += fold

    logging.info("Load data into memory")
    m0 = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30

    dataset, train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(
        ARG)

    delta_mem = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30 - m0
    logging.info(f"Dataset used memory = {delta_mem:.1f}GB")

    warmup_steps = int(WARMUP_RATIO * ARG.max_steps)
    logging.info(
        f'Total train steps={ARG.max_steps}, warmup steps={warmup_steps}')

    # model
    model = QQUniModel(MODEL_CONFIG, model_path=BERT_PATH, task=PRETRAIN_TASK)
    model.to(DEVICE)

    # optimizer
    optimizer = create_optimizer(
        model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

    # schedueler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=ARG.max_steps, num_warmup_steps=warmup_steps)

    # train
    val_loss = train(model, model_path, train_loader, val_loader, optimizer,
                     get_pred_and_loss=get_pred_and_loss,
                     scheduler=scheduler, num_epochs=NUM_EPOCHS)
    list_val_loss.append(val_loss)

    del dataset, train_dataset, val_dataset
    gc.collect()

    logging.info(f"Fold{fold} val_loss_list=" +
                 str([round(kk, 6) for kk in list_val_loss]))

logging.info(
    f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
logging.info("Train finish")
