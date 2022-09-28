import os


class Config:
    def __init__(self):
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.correct_labels = 21128
        self.detect_labels = 2
        self.block_size = 2
        self.hidden_size = 768
        self.plm_name = 'bert-base-chinese'
        self.max_seq_len = 512
        self.take_grads = 1  # embedding
        self.num_detect_layers = 2
        self.plm_path = '/data/bert_models/chinese_L-12_H-768_A-12'
