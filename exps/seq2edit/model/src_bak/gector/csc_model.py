import os
import numpy as np
import torch
import math
import jieba.posseg as posseg
from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast


class Model(torch.nn.Module):
    def __init__(self, config_path):
        super().__init__()
        self.hidden_size = 768
        self.vocab_size = 21128
        config = BertConfig.from_json_file(config_path)
        self.bert_model = BertModel(config)
        self.linear = torch.nn.Linear(in_features=self.hidden_size,
                                      out_features=self.vocab_size)

    def forward(self, x):
        bert_output = self.bert_model(**x)
        _output = bert_output["last_hidden_state"]
        logits = self.linear(_output)
        logits = torch.softmax(logits, dim=-1)
        return logits


class CSCModel:
    def __init__(self, model_path):
        self._max_len = 512
        self._batch_size = 8
        self._threshold = 0.5
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=False)
        self.model = Model(os.path.join(model_path, 'config.json'))
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'cbert_model.bin')))

    def data_preprocess(self, text_list):
        _text_list = []
        for value in text_list:
            _text_list.append(value)
            if len(_text_list) == self._batch_size:
                token_output = self.tokenizer.batch_encode_plus(_text_list, max_length=self._max_len,
                                                                return_tensors='pt', truncation=True,
                                                                padding=True, return_offsets_mapping=True)
                yield token_output, _text_list
                _text_list = []
        if _text_list:
            token_output = self.tokenizer.batch_encode_plus(_text_list, max_length=self._max_len,
                                                            return_tensors='pt', truncation=True,
                                                            padding=True, return_offsets_mapping=True)
            yield token_output, _text_list

    def data_postprocess(self, inputs, length):
        outputs = []
        for _input, _length in zip(inputs, length):
            _input = _input[1:_length + 1]
            pred_txt = self.tokenizer.convert_ids_to_tokens(_input)
            outputs.append(pred_txt)
        return outputs

    @staticmethod
    def _is_chinese_char(x):
        return True if u'\u4e00' <= x <= u'\u9fa5' else False

    def get_static_results(self, predict_lst, origin_lst, model_output, results: list):
        for idx, (pred, origin) in enumerate(zip(predict_lst, origin_lst)):
            if len(pred) != len(origin):
                results.append(''.join(pred))
                continue

            pred = [x.lstrip('##') if len(x) == 1 and self._is_chinese_char(x) and (len(x) == 1
                    and self._is_chinese_char(origin[i])) else origin[i].lstrip('##') for i, x in enumerate(pred)]
            text_len = len(origin)
            start = 0
            while start < text_len:
                if pred[start] == origin[start].lstrip('##'):
                    start += 1
                else:
                    end = start + 1
                    while end < text_len and pred[end] != origin[end]:
                        end += 1
                    if end < text_len:
                        for _idx in range(start, end):
                            prob = model_output[idx][_idx + 1]
                            prob_idx = np.argmax(prob)
                            if prob[prob_idx] < self._threshold:
                                pred[_idx] = origin[_idx]
                            # elif _idx <= 1:
                            #     pred[_idx] = origin[_idx]
                        start = end + 1
                    else:
                        start += 1
            pred = ''.join(pred)
            results.append(pred)

    def predict(self, text):
        results = []
        self.model.eval()
        with torch.no_grad():
            for token_feat, txt_lst in self.data_preprocess(text):
                offset_mapping = token_feat.pop('offset_mapping').cpu().numpy()
                output = self.model(token_feat)
                output = output.cpu().numpy()
                preds = np.argmax(output, axis=-1)
                offset_mapping = [[list(x) for x in value if sum(x) != 0] for value in offset_mapping]
                batch_length = [len(x) for x in offset_mapping]
                pred_tokens = self.data_postprocess(preds, batch_length)
                origin_tokens = [self.tokenizer.convert_ids_to_tokens(_token[1:_length+1])
                                 for _token, _length in zip(token_feat['input_ids'], batch_length)]
                origin_post = []
                for _token, offset, _text in zip(origin_tokens, offset_mapping, txt_lst):
                    tmp_word = []
                    for word, _offset in zip(_token, offset):
                        if word == '[UNK]':
                            tmp_word.append(_text[_offset[0]: _offset[1]])
                        else:
                            tmp_word.append(word)
                    origin_post.append(tmp_word)
                self.get_static_results(pred_tokens, origin_post, output, results)
        return results


if __name__ == "__main__":
    model_path = '../plm/bert'
    model = CSCModel(model_path)
    text_list = ['吸咽引起疾病的根原。']
    _result = model.predict(text_list)
    print(_result)
