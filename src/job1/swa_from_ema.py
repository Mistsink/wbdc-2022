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
    
    return loss, results


def train_and_validate(args, train_dataloader, val_dataloader, fold=0):
    

    # 2. build model and optimizers
    model = MultiModal(args)
    if args.device == 'cuda':
        model.cuda()
        
    global swa
    swa = SWA(model)
    

    # 3. training
    for epoch in range(args.max_epochs):
        if epoch >= 3:
            break
       
        if swa != None and epoch >= args.swa_start:
            checkpoint = torch.load(f'{args.path_out}job1_ema_{fold}_{epoch}.bin')
            model.load_state_dict(checkpoint)
            swa.append_model(model)

    
    loss, results = validate(swa.model(), val_dataloader, args, fold, epoch)
    results = {k: round(v, 4) for k, v in results.items()}
    logger.info(f"SWA : Loss {loss:.3f}, F1 {results['mean_f1']}")

    swa_save_path = f'{args.path_out}job1_swa_{fold}.bin'
    swa.save_model(swa_save_path)




def main():
    args = parse_args()
#     setup_cp_files(args)

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


