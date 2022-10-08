import os
import torch.nn as nn
from transformers import BertModel
from collections import OrderedDict
from transformers.models.bert.modeling_bert import BertConfig, BertLayer


class MDCSpeller(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_ignore_id = config.label_ignore_id
        bert_config = BertConfig().from_pretrained(config.pretrained_model)
        self.encoder = nn.ModuleList([BertLayer(bert_config) for _ in range(config.num_detect_layers)])
        self.correction_layer = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
        self.detection_layer = nn.Linear(bert_config.hidden_size, config.detect_labels)
        self.bert_model = BertModel.from_pretrained(config.pretrained_model)
        self.correct_loss = nn.CrossEntropyLoss(ignore_index=self.label_ignore_id)
        self.detect_loss = nn.CrossEntropyLoss()
        encoder_params = OrderedDict()
        for name, _ in self.encoder.named_parameters():
            init_param = self.bert_model.state_dict()['encoder.layer.' + name]
            encoder_params[name] = init_param
        self.encoder.load_state_dict(encoder_params)

    def correction(self, bert_feat, detect_embed):
        bert_feat += detect_embed
        output = self.correction_layer(bert_feat)
        return output

    def detection(self, embed_feat):
        for layer in self.encoder:
            embed_feat = layer(embed_feat)[0]
        output = self.detection_layer(embed_feat)
        return output, embed_feat

    def forward(self, input_ids, attention_mask, token_type_ids,
                trg_ids=None, trg_detect=None):
        feature = self.bert_model(input_ids, attention_mask, token_type_ids)
        embed = self.bert_model.embeddings(input_ids, attention_mask, token_type_ids)
        detect_output, detect_embed = self.detection(embed)
        correct_output = self.correction(feature.last_hidden_state, detect_embed)
        loss = 0.0
        if trg_ids is not None:
            cor_loss = self.correct_loss(correct_output.view(-1, self.bert_model.config.vocab_size), trg_ids.view(-1))
            loss += self.config.alpha * cor_loss

        if trg_detect is not None:
            detect_loss = self.detect_loss(detect_output.view(-1, self.config.detect_labels), trg_detect.view(-1))
            loss += (1 - self.config.alpha) * detect_loss

        return loss, correct_output, detect_output
