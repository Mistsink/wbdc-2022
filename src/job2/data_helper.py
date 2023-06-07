import json
import random
import zipfile
from io import BytesIO
from functools import partial
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold


from category_id_map import category_id_to_lv2id


def create_dataloaders_with_fold(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    
    with open(args.train_annotation, 'r', encoding='utf8') as f:
        anns = json.load(f)
    ids = [item['id'] for item in anns]
    labels = [item['category_id'] for item in anns]
    idx_fold = []
    kfold = StratifiedKFold(n_splits=args.n_splits, shuffle=args.shuffle, random_state=args.seed)
    for _, idx in enumerate(kfold.split(ids, labels)):
        idx_fold.append(idx)
    
    train_size = len(idx_fold[0][0])
    total_steps = (train_size // args.batch_size) * args.max_epochs if train_size % args.batch_size == 0 else (
        train_size // args.batch_size + 1) * args.max_epochs
    args.max_steps = total_steps
    
        
    def with_fold(fold: int):
        train_idx, val_idx = idx_fold[fold]
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        
        if args.num_workers > 0:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
        else:
            dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        train_dataloader = dataloader_class(train_dataset,
                                            batch_size=args.batch_size,
                                            sampler=train_sampler,
                                            drop_last=True)
        val_dataloader = dataloader_class(val_dataset,
                                          batch_size=args.val_batch_size,
                                          sampler=val_sampler,
                                          drop_last=False)
        return train_dataloader, val_dataloader
    
    return with_fold



def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_size = size - val_size
    total_steps = (train_size // args.batch_size) * args.max_epochs if train_size % args.batch_size == 0 else (
        train_size // args.batch_size + 1) * args.max_epochs
    args.max_steps = total_steps

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
#     train_sampler = WeightedRandomSampler(train_dataset)

    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):

        self.text_parts = args.text_parts
        self.bert_seq_length = args.bert_seq_length


        self._init(args, ann_path, zip_feats, test_mode)
        

    def _init(self, args, ann_path, zip_feats, test_mode):
        self.max_frame = args.max_frames
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        
        self.pattern = re.compile(r'#[\d\w]*')
        self.cnt = 0


    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def get_text_parts(self, idx: int):
        res = []
        for part_k in self.text_parts:
            text = ''
            if part_k == 'ocr':
                text += ''.join(ocr['text'] for ocr in self.anns[idx]['ocr'])
            else:
                text = self.anns[idx][part_k]
            res.append(text)
        return res

    def process_text_hh(self, text_parts: list, item_len: int) -> list:
        parts = text_parts
        parts = [self.tokenizer.tokenize(part) for part in text_parts]

        parts = [''.join(part[:item_len // 2] + part[-item_len // 2:])
                 for part in parts]
        return parts
    
    def process_text_avg(self, text_parts: list, item_len: int) -> list:
        parts = text_parts
        parts = [self.tokenizer.tokenize(part) for part in text_parts]

        parts = [''.join(part[:item_len]) for part in parts]
        return parts


    def tokenize_text(self, processed_text: str or list, max_length, add_special_tokens=True, ) -> tuple:
        if type(processed_text) is list:
            processed_text = '[SEP]'.join(processed_text)
        encoded_inputs = self.tokenizer(processed_text,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=add_special_tokens)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        frame_input, frame_mask = self.get_visual_feats(idx)

        text_part_list = self.get_text_parts(idx=idx)
        processed_list = self.process_text_avg(text_part_list, item_len=self.bert_seq_length//len(self.text_parts))
#         processed_list = self.process_text_gdy(text_part_list)

        if self.cnt < 1:
            print(f'processed_text: {processed_list}')
            self.cnt += 1
        title_input, title_mask = self.tokenize_text(processed_list, max_length=self.bert_seq_length)
        

        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        )

        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data

    def process_text_gdy(self, text_parts: list, with_tags=True, cat_then_trunc=False) -> str:
        parts = text_parts

        if with_tags:
            tags = ','.join(self.extract_tags(parts[0]))
            if len(parts) >= 2:
                tags += ','.join(self.extract_tags(parts[1]))
            tags = '' if tags == '' else tags
            parts.insert(0, tags)

        parts = [self.tokenizer.tokenize(part) for part in text_parts]
        parts = [list(filter(lambda x: x != '[UNK]', part)) for part in parts]
        if self.cnt < 1:
            print(f'parts: {parts}')

        # if cat_then_trunc:
        #     res = []
        #     for part in parts:
        #         res += part
        #     if len(text_parts) == 4:

        if len(text_parts) == 4:
            while len(parts[0]) + len(parts[1]) + len(parts[2]) > self.bert_seq_length - 5:
                i = 0
                if len(parts[0]) < len(parts[1]):
                    i = 1
                if len(parts[i]) < len(parts[2]):
                    i = 2
                if len(parts[i]) < len(parts[3]):
                    i = 3
                parts[i].pop()
        elif len(text_parts) == 3:
            while len(parts[0]) + len(parts[1]) + len(parts[2]) > self.bert_seq_length - 4:
                i = 0
                if len(parts[0]) < len(parts[1]):
                    i = 1
                if len(parts[i]) < len(parts[2]):
                    i = 2
                parts[i].pop()
        elif len(text_parts) == 2:
            while len(parts[0]) + len(parts[1]) > self.bert_seq_length - 3:
                i = 0
                if len(parts[0]) < len(parts[1]):
                    i = 1
                parts[i].pop()
        else:
            while len(parts[0]) > self.bert_seq_length - 2:
                parts[0].pop()
        processed_parts = [''.join(part) for part in parts]
        processed_text = ''
        for part in processed_parts:
            if part != '':
                processed_text += part
                processed_text += '[SEP]'
        if processed_text == '':
            processed_text += '[SEP]'
        return f"[CLS]{processed_text}"

    def extract_tags(self, tar: str) -> list:
        res = self.pattern.findall(tar)
        
        return res
    


