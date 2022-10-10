# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
import jieba
from collections import defaultdict
import Levenshtein
sys.path.append('../')
from fluent.models import MaskedBert

REGEX_BATCH_SIZE = 10000


class IdiomCorrection:

    def __init__(self):
        self._pattern_lst = []
        self._bigram_inverted_dic = defaultdict(list)
        self.white_list = []
        self._model = MaskedBert.from_pretrained('bert-base-chinese')
        self.load()

    def load(self):
        # self.white_list =
        pattern_lst = [line.strip() for line in open('./data/dict_pattern_idiom.txt') if line.strip()]
        for i in range(0, len(pattern_lst), REGEX_BATCH_SIZE):
            self._pattern_lst.append(re.compile('|'.join(pattern_lst[i:i + REGEX_BATCH_SIZE])))

        with open('./data/idioms.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                word = line.strip()
                if word:
                    for idx in range(0, len(word) - 1):
                        self._bigram_inverted_dic[word[idx:idx + 2]].append(word)

    def error_check(self, content):
        check_ret = []
        error_indexs = []
        for pattern in self._pattern_lst:
            for di in pattern.finditer(content):
                start_offset, end_offset, di_str = di.start(), di.end(), di.group(0)
                if (start_offset, end_offset) in error_indexs:
                    continue
                else:
                    error_indexs.append((start_offset, end_offset))
                error_correction = self.get_simi_word(di_str)
                if not re.search('^[\u4e00-\u9fa5]+$', di_str) or error_correction == []:
                    continue
                if (error_correction, di_str) in self.white_list:
                    continue
                item = {
                    "start_offset": start_offset,
                    "end_offset": end_offset - 1,
                    "origin": di_str,
                    "error_correction": error_correction
                }
                if self.item_postprocess(content, item):
                    continue
                check_ret.append(item)
        if check_ret:
            print(check_ret)
            check_ret = sorted(check_ret, key=lambda x: x['start_offset'], reverse=True)
            for item in check_ret:
                if len(item['error_correction']) == 1:
                    content = content[:item['start_offset']] + item['error_correction'][0] +\
                              content[item['end_offset']+1:]
            return content
        else:
            return None

    def get_simi_word(self, regex_match_str):
        candi_ret = []
        tmp_ret = []
        for idx in range(0, len(regex_match_str) - 1):
            if regex_match_str[idx:idx + 2] in self._bigram_inverted_dic:
                tmp_ret.extend(self._bigram_inverted_dic[regex_match_str[idx:idx + 2]])
        tmp_ret = list(set(tmp_ret))
        for simi_word in tmp_ret:
            norm_dist = Levenshtein.distance(regex_match_str, simi_word)
            if norm_dist <= 1 or (len(simi_word) > 4 and norm_dist <= 2):
                candi_ret.append(simi_word)
        return candi_ret

    def item_postprocess(self, content, item):
        """根据困惑度比较纠错前后句子通顺程度，进行后处理"""
        start_offset = item['start_offset']
        end_offset = item['end_offset']
        while re.search(r'[%sA-Z“”\"]' % r'\u3400-\u9FFF', content[start_offset]):
            start_offset -= 1
            if start_offset == -1:
                break
        while end_offset < len(content) and re.search(r'[%sA-Z“”\"]' % r'\u3400-\u9FFF', content[end_offset]):
            end_offset += 1
        origin_s = content[start_offset+1:end_offset]
        cor_s = origin_s.replace(item['origin'], item['error_correction'][0])

        origin_s_prob = self._model.perplexity(x=jieba.lcut(origin_s), verbose=False)
        cor_prob = self._model.perplexity(x=jieba.lcut(cor_s), verbose=False)

        if cor_prob < origin_s_prob:
            return False
        else:
            return True


def main(config):
    correct = IdiomCorrection()
    input_data = [line.strip().split() for line in open(config.input_data)]
    data_all = []

    for value in input_data:
        result = correct.error_check(value[1])
        if result is not None:
            data_all.append([value[0], result])
        else:
            data_all.append(value)
    with open(args.save_path, 'w', encoding='utf-8') as wf:
        for value in data_all:
            wf.write(' '.join(value) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('idiom correct')
    parser.add_argument('--input_data', type=str, default='../../data/cged_test.txt')
    parser.add_argument('--pattern_path', type=str, default='./data/idioms.txt')
    parser.add_argument('--save_path', type=str, default='../../data/cged_idiom.txt')
    args = parser.parse_args()
    main(args)
