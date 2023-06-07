import torch
import torch.nn as nn

from util import validate_and_save_model

import sys
sys.path.append('../third_party/qq_unibert/model')
from model_uni import MultiModal as uni


MultiModal = uni

import copy

class SWA:
    def __init__(self, model):
        self.model_list = []
        self.swa_n = 0.
        self.swa_model = copy.deepcopy(model)
        
    def append_model(self, model):
         with torch.no_grad():
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (self.swa_n + 1.)

            for name, para in self.swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            self.swa_n += 1
            
    def model(self):
        return self.swa_model
    
    def save_model(self, swa_model_path):
        torch.save(self.swa_model.state_dict(), swa_model_path)

class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def save_swa(train_dataloader, val_dataloader, args, validate_fn, swa, logger, epoch, best_score, model_save_path):
    torch.optim.swa_utils.update_bn(train_dataloader, swa)
    validate_and_save_model(args, val_dataloader, swa, best_score, epoch, validate_fn, logger, model_save_path)
    
class FGM(object):
 
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                #print(param)
                #print(param.grad)
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}