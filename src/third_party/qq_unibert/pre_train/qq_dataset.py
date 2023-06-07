import json
import random
# import zipfile38 as zipfile
import zipfile
import re
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id


def create_dataloaders(args):
    train_val_dataset = MultiModalDataset(
        args, args.train_annotation, args.train_zip_feats)
    test_dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    unlabeled_dataset = MultiModalDataset(args, args.unlabeled_annotation, args.unlabeled_zip_feats, test_mode=True)

    
    dataset = torch.utils.data.ConcatDataset([train_val_dataset, test_dataset, unlabeled_dataset])
    
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_size = size - val_size

    total_steps = (train_size // args.batch_size) * args.max_epochs if train_size % args.batch_size == 0 else (
        train_size // args.batch_size + 1) * args.max_epochs
    args.max_steps = total_steps

    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [size - val_size,
                                                                   val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

#     train_dataset = torch.utils.data.ConcatDataset([train_val_dataset, unlabeled_dataset])
#     val_dataset = test_dataset
    
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return None, train_dataset, val_dataset, train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.reined_ratio = args.reined_ratio
        self.text_parts = args.text_parts
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        # pattern for extracting tags by re
        self.pattern = re.compile(r'#[\d\w]*')

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.handles = [None for _ in range(args.num_workers)]

        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        self.cnt = 0

    def __len__(self) -> int:
        return int(len(self.anns) * self.reined_ratio)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(
            name=f'{vid}.npy')), allow_pickle=True)
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

    def tokenize_text(self, processed_text: str, max_length, add_special_tokens=True, ) -> tuple:
        if type(processed_text) is list:
            processed_text = '[SEP]'.join(processed_text)
        encoded_inputs = self.tokenizer(processed_text,
                                        max_length=max_length,
                                        padding='max_length',
                                        truncation=True,
                                        add_special_tokens=add_special_tokens)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        token_type_ids = torch.LongTensor(encoded_inputs['token_type_ids'])
        return input_ids, mask, token_type_ids

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title tokens
        text_parts = self.get_text_parts(idx)
        processed_parts = self.process_text_pp(text_parts, item_len=128)
        title_input, title_mask, _ = self.tokenize_text(processed_parts,
                                                        max_length=self.bert_seq_length, add_special_tokens=True)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        )

        # Step 4, load label if not test mode
#         if not self.test_mode:
#             label = category_id_to_lv2id(self.anns[idx]['category_id'])
#             data['label'] = torch.LongTensor([label])

        return data

    def get_text_parts(self, idx: int) -> list:
        res = []
        for part_k in self.text_parts:
            text = ''
            if part_k == 'ocr':
                text += '。'.join(ocr['text'] for ocr in self.anns[idx]['ocr'])
            else:
                text = self.anns[idx][part_k]
            res.append(text)
        return res

    def process_text_pp(self, text_parts: list, item_len: int) -> list:
        parts = text_parts
        parts = [self.tokenizer.tokenize(part) for part in text_parts]

        parts = [''.join(part[:item_len]) for part in parts]
        return parts
