import logging
import torch
from tqdm import tqdm


from config import parse_args
from data_helper import create_dataloaders, create_dataloaders_with_fold
from model import MultiModal, EMA, FGM, SWA
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate, setup_cp_files, validate_and_save_model


logger = None
scaler = None
autocast = None
fgm = None
ema = None
swa = None
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast

def validate(model, val_dataloader, args, fold, epoch):
    if ema != None:
        ema.apply_shadow()
    
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            if autocast != None:
                with autocast():
                    loss, _, pred_label_id, label = model(batch)
                    loss = loss.mean()
            else:
                loss, _, pred_label_id, label = model(batch)
                loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    
    if swa != None and epoch >= args.swa_start:
        swa.append_model(model)
    
    if ema != None:
        ema_save_path = f'{args.path_out}job1_ema_{fold}_{epoch}.bin'
        ema.save_model(ema_save_path)
        ema.restore()
    
    return loss, results


def train_and_validate(args, train_dataloader, val_dataloader, fold=0):
    

    # 2. build model and optimizers
    model = MultiModal(args)
    if args.device == 'cuda':
        model.cuda()
    if args.pre_train_path != '':
        model.load_state_dict(torch.load(args.pre_train_path), strict=False)
    
        
    optimizer, scheduler, _, _ = build_optimizer(args, model)
    
    if args.use_swa:
        global swa
        swa = SWA(model)
    
    if args.use_fgm:
        global fgm
        fgm = FGM(model, epsilon=args.fgm_epsilon, emb_name='bert.embeddings.word_embeddings')
    
    if args.use_ema:
        global ema
        ema = EMA(model, args.ema_decay)
        ema.register()
    

    # 3. training
    best_score = {'val':args.best_score}
    for epoch in range(args.max_epochs):
        if epoch >= 3:
            break
        
        
        for batch in tqdm(train_dataloader):
            model.train()
            if autocast != None:
                with autocast():
                    loss, accuracy, _, _ = model(batch)
                    loss = loss.mean()
                    accuracy = accuracy.mean()

                scaler.scale(loss).backward()
                
                if fgm != None:
                    with autocast():
                        fgm.attack()
                        loss_adv,_,_,_ = model(batch)
                        loss_adv = loss_adv.mean()
                        scaler.scale(loss_adv).backward()
                        fgm.restore() # 恢复embedding参数
                
                scaler.step(optimizer)

                if swa != None and epoch >= args.swa_start:
                    swa.append_model(model)
                
                scheduler.step()
                scaler.update()                
            else:
                loss, accuracy, _, _ = model(batch)
                loss = loss.mean()
                accuracy = accuracy.mean()
                loss.backward()
                
                if fgm != None:
                    fgm.attack()
                    loss_adv,_,_,_ = model(batch)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore() # 恢复embedding参数
                
                optimizer.step()
                if swa != None and epoch >= args.swa_start:
                    swa.append_model(model)
                
                scheduler.step()

            if ema != None:
                ema.update()
            
            optimizer.zero_grad()

        # 4. validation
        model_save_path = f'./{fold}.bin'
       
        validate_and_save_model(args, val_dataloader, model, best_score, epoch, validate, logger, model_save_path, fold)
    



def main():
    args = parse_args()

    global logger
    logger = setup_logging(args)
    setup_device(args)
    setup_seed(args)

    
    # 1. load data
    with_fold = create_dataloaders_with_fold(args)
    for i in range(args.n_splits):
        train_dataloader, val_dataloader = with_fold(i)
        train_and_validate(args, train_dataloader, val_dataloader, fold=i)


if __name__ == '__main__':
    main()


