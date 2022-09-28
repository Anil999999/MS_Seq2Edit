import os
import torch
from transformers import BertTokenizer
from utils import *
from models import MDCSpeller
from tqdm import tqdm
import argparse
from openccpy import Opencc

opencc = Opencc()


class Decoder:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
        self.test_loader = init_dataloader(config.test_path, config, "test", self.tokenizer)
        # self.model = BERT_Model(config, self.test_loader.dataset)
        self.model = MDCSpeller(config)
        self.model.to(self.device)
        self.config = config

    def __forward_prop(self, dataloader, back_prop=True):
        collected_outputs = []
        logits_all = []
        for batch in tqdm(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, logits, det_logits = self.model(**batch)
            outputs = torch.argmax(logits, dim=-1)
            logits = torch.softmax(logits, dim=-1)
            logits = logits.cpu()
            # det_outputs = torch.argmax(det_logits, dim=-1)
            # texts = []
            for idx, outputs_i in enumerate(outputs):
                tmp_prob = []
                collected_outputs.append(outputs_i)
                for _idx, _output in enumerate(outputs_i):
                    tmp_prob.append(logits[idx, _idx, _output].item())
                logits_all.append(tmp_prob)
        return collected_outputs, logits_all

    def decode(self):
        model = self.model
        model.load_state_dict(torch.load(self.config.model_path))
        model.eval()
        with torch.no_grad():
            outputs, prob = self.__forward_prop(dataloader=self.test_loader, back_prop=False)
            save_decode_result_lbl(outputs, prob, self.test_loader.dataset.data, self.config.save_path)


def main(config):
    decoder = Decoder(config)
    decoder.decode()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--num_detect_layers", default=2, type=int,
                        help="the number of detection layers")
    parser.add_argument('--detect_labels', default=2, type=int)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--label_ignore_id", default=0, type=int)

    args = parser.parse_args()
    main(args)
