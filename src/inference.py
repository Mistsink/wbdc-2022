import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
import numpy as np
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal

model_list = [    
    # job1
    f'job1_swa_0.bin',
    f'job1_swa_1.bin',
    f'job1_swa_2.bin',
    f'job1_swa_3.bin',
    f'job1_swa_4.bin',
    
    f'job1_ema_0_3.bin',
    f'job1_ema_1_3.bin',
    f'job1_ema_2_3.bin',
    f'job1_ema_3_3.bin',
    f'job1_ema_4_3.bin',

    # job2
    f'job2_swa_0.bin',
    f'job2_swa_1.bin',
    f'job2_swa_2.bin',
    f'job2_swa_3.bin',
    f'job2_swa_4.bin',
]

cnt = 0

def inference():
    
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    pre = []
    
    for i in range(len(model_list)):
        checkpoint = torch.load(f'../data/{model_list[i]}', map_location='cpu')
        model = MultiModal(args)
        model.load_state_dict(checkpoint)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                pred_label_id = model(batch, inference=True)
                predictions.extend(pred_label_id.cpu().numpy())
                
        if i==0:
            pre = [predictions]
        else:
            pre = pre + [predictions]
    l = []
    for i in range(len(pre[0])):
        tmp = []
        for j in range(len(pre)):
            tmp.append(pre[j][i])
        x = max(set(tmp),key=tmp.count)
        l.append(x)
    
    with open('../data/result.csv', 'w') as f:
        for pred_label_id, ann in zip(l, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    


if __name__ == '__main__':
    inference()
