import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from category_id_map import CATEGORY_ID_LIST
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead



class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(768, config.hidden_size)
        self.video_embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.act = nn.Sigmoid()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):        
        text_emb = self.embeddings(input_ids=text_input_ids)
        
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]
        
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        
        # reduce frame feature dimensions : 1536 -> 1024
        video_feature = self.video_fc(video_feature)
        video_feature = self.act(video_feature)
        video_emb = self.video_embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_emb, text_emb], 1)
        
        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        
        encoder_outputs = self.encoder(embedding_output, attention_mask=mask)['last_hidden_state']
        return encoder_outputs

class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert(config)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        encoder_outputs = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]: , :]
        else:
            return encoder_outputs, None       

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        bert_config = BertConfig.from_pretrained(f'{args.bert_dir}/config.json')
        self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config = bert_config)
        self.meaningpooling = nn.AdaptiveAvgPool2d((1,bert_config.hidden_size))
        self.newfc_tag = nn.Linear(bert_config.hidden_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        text_ids = inputs['title_input'].cuda()
        text_mask = inputs['title_mask'].cuda()
        frame_emb = inputs['frame_input'].cuda()
        frame_mask = inputs['frame_mask'].cuda()
        if 'label' in inputs:
            label = inputs['label'].cuda()
        bert_embedding,_ = self.roberta(frame_emb, frame_mask,text_ids, text_mask)

        final_embedding = self.meaningpooling(bert_embedding).squeeze(1)
        prediction = self.newfc_tag(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, label)

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
