import random
import re
import numpy as np
import threading
from jieba import posseg
import logging
import csv
import pandas as pd
import json
from openccpy import Opencc
import argparse
import html


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class DataGenerator(object):

    # 乱序：漏字：多字：用词错误=5%:10%:15%:70%
    def __init__(self, save_path):
        self.probs_all = 0.05
        self.disorder_prob = 0.05
        self.missing_prob = 0.1
        self.selection_prob = 0.7
        self.redundant_prob = 0.15
        self.confusion_pronounce_path = "./data/pinyin_confusion_all.json"
        self.confusion_shape_path = "./data/shape_confusion_all.json"
        self.extra_candidate_path = "./data/extra_candidate.json"
        self.vocab_path = "./data/vocab.txt"
        self.vocab_threshold = 100
        self.extra_candidate = json.load(open(self.extra_candidate_path, 'r'))
        self.confusion_pronounce = json.load(open(self.confusion_pronounce_path, 'r'))
        self.confusion_shape = json.load(open(self.confusion_shape_path, 'r'))
        self.stopword_path = "./data/stop_words"
        self.not_common_path = "./data/生僻字.txt"
        self.words_path = "./data/words.txt"
        self.vocab, self.stopwords, self.words = self.load_data()
        self.mx1 = threading.Lock()
        headers = ['id', 'source', 'target']
        self.csv_writer = csv.DictWriter(open(save_path, 'w', encoding='utf-8'), headers, delimiter='\t')
        self.csv_writer.writeheader()

    def load_data(self):
        stop_words, vocab = [], []
        with open(self.stopword_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                cont = line.strip('\n').strip()
                if self.is_chinese_char(cont):
                    if len(cont) < 2:
                        stop_words.append(cont)
                else:
                    stop_words.append(cont)

        non_common = [line.strip() for line in open(self.not_common_path, 'r', encoding='utf-8')]

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                cont = line.strip('\n').split('\t')
                if int(cont[1]) > self.vocab_threshold and cont[0] not in stop_words and cont[0] not in non_common:
                    if cont[0] != Opencc.to_simple(cont[0]):
                        vocab.append(Opencc.to_simple(cont[0]))
                    else:
                        vocab.append(cont[0])

        words = []
        with open(self.words_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                if len(line) <= 2:
                    words.append(line)

        return vocab, stop_words, words

    def data_generate(self, input_data):
        data_all = []
        for doc_idxes, (doc_idx, data) in enumerate(input_data):
            if not isinstance(data, str):
                continue
            data = data.replace(' ', '').replace('\n', '').replace('【', '').replace('】', '')
            sentence_lst = self.cut_sentence(data)

            # 计算句子在文章中的总位移
            sentence_length = [len(sentence) for sentence in sentence_lst]
            start_offset_map = [0 for _ in range(len(sentence_lst))]
            for sentence_id, sentence in enumerate(sentence_lst):
                start_offset_map[sentence_id] = sum([sentence_length[i] for i in range(sentence_id)])

            # 将句子中的空格，制表符去除
            sentence_cleaned = self.sentence_cleaning(sentence_lst)
            if len(sentence_lst) != len(sentence_cleaned):
                continue

            masked_sentence_idx = []
            for idx, sentence in enumerate(sentence_cleaned):
                if 20 < len(sentence) < 300:
                    masked_sentence_idx.append(idx)

            masked_sentence_length = len(masked_sentence_idx)
            if masked_sentence_length == 0:
                continue

            # all
            for mask_idx in masked_sentence_idx:
                sentence = sentence_cleaned[mask_idx]
                sentence_len = len(sentence)
                mask_idx_len = int(np.round(sentence_len * self.probs_all))
                tmp_data = {"id": str(doc_idx) + '_' + str(mask_idx), "source": sentence}
                while mask_idx_len > 0:
                    if random.random() >= self.selection_prob:
                        # 生成音似句子
                        result = self.get_select_sim_sentence(sentence)
                        if result is not None and result['mode'] != "":
                            sentence = result['error_sentence']
                            mask_idx_len -= 1

                    elif random.random() >= self.redundant_prob:
                        result = self.get_extra_sentence(sentence)
                        if result is not None and result['mode'] != "":
                            sentence = result['error_sentence']
                            mask_idx_len -= 1

                    elif random.random() >= self.missing_prob:
                        result = self.get_extra_sentence(sentence)
                        if result is not None and result['mode'] != "":
                            sentence = result['error_sentence']
                            mask_idx_len -= 1
                    else:
                        result = self.get_disorder_sim_sentence(sentence)
                        if result is not None and result['mode'] != "":
                            sentence = result['error_sentence']
                            mask_idx_len -= 1
                tmp_data['target'] = sentence
                data_all.append(tmp_data)
            if len(data_all) == 1000:
                self.save_data(data_all)
                data_all = []
        if len(data_all) != 0:
            self.save_data(data_all)

    def get_select_sim_sentence(self, sentence):
        if random.random() >= 0.7:
            result = self.get_pronounce_sim_sentence(sentence)
        else:
            result = self.get_shape_sim_sentence(sentence)
        return result

    def get_disorder_sim_sentence(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        _, word_mapping, mask_mapping = self.sentence_preprocess(sentence)
        mask_idx = list(np.where(mask_mapping == 1)[0])
        if not mask_idx:
            return None
        res_mapping = [(i + 1) * j for i, j in zip(word_mapping, mask_mapping)]
        word_span = []
        l_idx, r_idx = 0, 1
        while r_idx < len(res_mapping):
            if res_mapping[l_idx] == res_mapping[r_idx]:
                r_idx += 1
            else:
                if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                    l_idx = r_idx
                    r_idx += 1
                else:
                    word_span.append((l_idx, r_idx))
                    l_idx = r_idx
                    r_idx += 1
        span_idx = range(len(word_span))
        left_idx = random.choice(span_idx)
        right_span = list(range(left_idx+1, left_idx+5 if left_idx+5 < len(span_idx) else len(span_idx)-1))
        if not right_span:
            return None
        right_idx = random.choice(right_span)
        if right_idx - left_idx == 1:
            left_word = word_span[left_idx]
            right_word = word_span[right_idx]
            sentence = sentence[:left_word[0]] + sentence[right_word[0]: right_word[1]] + \
                       sentence[left_word[0]: left_word[1]] + sentence[right_word[1]:]
            correct_word = sentence[left_word[0]: right_word[1]]
        else:
            pivot_idx = list(range(left_idx+1, right_idx))
            pivot_idx = random.choice(pivot_idx)
            left_word = word_span[left_idx]
            right_word = word_span[right_idx]
            pivot = word_span[pivot_idx]
            sentence = sentence[:left_word[0]] + sentence[pivot[0]: right_word[1]] + \
                       sentence[left_word[0]: pivot[0]] + sentence[right_word[1]:]
            correct_word = sentence[pivot[0]: right_word[1]] + sentence[left_word[0]: pivot[0]]

        item["mode"] = "disorder"
        item["error_sentence"] = sentence
        item["start_offset"] = left_word[0]
        item["end_offset"] = right_word[1] + 1
        item["error_word"] = ''
        item["correct_word"] = correct_word
        return item

    def get_pronounce_sim_sentence(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        _, word_mapping, mask_mapping = self.sentence_preprocess(sentence)
        mask_idx = list(np.where(mask_mapping == 1)[0])
        times = 6

        while times > -1:
            if not mask_idx:
                break
            if random.random() <= 0.15:
                # 选词
                res_mapping = [(i+1) * j for i, j in zip(word_mapping, mask_mapping)]
                word_span = []
                l_idx, r_idx = 0, 1
                while r_idx < len(res_mapping):
                    if res_mapping[l_idx] == res_mapping[r_idx]:
                        r_idx += 1
                    else:
                        if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                            l_idx = r_idx
                            r_idx += 1
                        else:
                            word_span.append((l_idx, r_idx))
                            l_idx = r_idx
                            r_idx += 1
                if not word_span:
                    continue
                word_span = random.choice(word_span)
                words = sentence[word_span[0]:word_span[1]]
                word_idxs = [i for i in range(len(words))]
                while len(word_idxs) > 0:
                    char_idx = np.random.choice(word_idxs)
                    candidate = self.confusion_pronounce.get(words[char_idx], '')
                    char_offset = char_idx + word_span[0]
                    if candidate:
                        candi_word = random.choice(candidate)
                        item["mode"] = "pronounce_smi"
                        item["error_sentence"] = sentence[:char_offset] + candi_word + sentence[char_offset + 1:]
                        item["start_offset"] = char_offset
                        item["end_offset"] = char_offset + 1
                        item["error_word"] = candi_word
                        item["correct_word"] = sentence[char_offset]
                        return item
                    else:
                        word_idxs = list(word_idxs)
                        word_idxs.remove(char_idx)
                        times -= 1
            else:
                sen_idx = random.choice(mask_idx)
                word = sentence[sen_idx]
                candidate = self.confusion_pronounce.get(word, '')
                if candidate:
                    candi_word = random.choice(candidate)
                    item["mode"] = "pronounce_smi"
                    item["error_sentence"] = sentence[:sen_idx] + candi_word + sentence[sen_idx + 1:]
                    item["start_offset"] = sen_idx
                    item["end_offset"] = sen_idx + 1
                    item["error_word"] = candi_word
                    item["correct_word"] = sentence[sen_idx]
                    return item
                else:
                    mask_idx = list(mask_idx)
                    mask_idx.remove(sen_idx)
                    times -= 1
        return None

    def sentence_cleaning(self, sentences):
        """
        删除句子中的制表符，空格等
        :param sentences:
        :return:
        """
        if isinstance(sentences, list):
            new_sentences = []
            for sentence in sentences:
                sentence = html.escape(sentence)
                sentence = re.sub(r'\s', '', sentence)
                new_sentences.append(sentence)
            return new_sentences
        else:
            sentences = html.escape(sentences)
            sentences = re.sub(r'\s', '', sentences)
            return sentences

    def get_shape_sim_sentence(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        words_lst, word_mapping, mask_mapping = self.sentence_preprocess(sentence)
        mask_idx = list(np.where(mask_mapping == 1)[0])

        times = 6
        while times >= 0:
            if len(mask_idx) < 0 or not mask_idx:
                break
            if random.random() <= 0.15:
                # 选词
                res_mapping = [(i+1) * j for i, j in zip(word_mapping, mask_mapping)]
                word_span = []
                l_idx, r_idx = 0, 1
                while r_idx < len(res_mapping):
                    if res_mapping[l_idx] == res_mapping[r_idx]:
                        r_idx += 1
                    else:
                        if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                            l_idx = r_idx
                            r_idx += 1
                        else:
                            word_span.append((l_idx, r_idx))
                            l_idx = r_idx
                            r_idx += 1
                if not word_span:
                    continue
                word_span = random.choice(word_span)
                words = sentence[word_span[0]:word_span[1]]
                word_idxs = [i for i in range(len(words))]
                while len(word_idxs) > 0:
                    char_idx = np.random.choice(word_idxs)
                    candidate = self.confusion_shape.get(words[char_idx], '')
                    char_offset = char_idx + word_span[0]
                    if candidate:
                        candi_word = random.choice(candidate)
                        item["mode"] = "shape_smi"
                        item["error_sentence"] = sentence[:char_offset] + candi_word + sentence[char_offset + 1:]
                        item["start_offset"] = char_offset
                        item["end_offset"] = char_offset + 1
                        item["error_word"] = candi_word
                        item["correct_word"] = sentence[char_offset]
                        return item
                    else:
                        word_idxs = list(word_idxs)
                        word_idxs.remove(char_idx)
                        times -= 1
            else:
                sen_idx = random.choice(mask_idx)
                _char = sentence[sen_idx]
                candidate = self.confusion_shape.get(_char, '')
                if candidate:
                    cand_char = random.choice(candidate)
                    item["mode"] = "shape_smi"
                    item["error_sentence"] = sentence[:sen_idx] + cand_char + sentence[sen_idx + 1:]
                    item["start_offset"] = sen_idx
                    item["end_offset"] = sen_idx + 1
                    item["error_word"] = cand_char
                    item["correct_word"] = sentence[sen_idx]
                    return item
                else:
                    times -= 1
                    mask_idx = list(mask_idx)
                    mask_idx.remove(sen_idx)
        return None

    def get_extra_sentence(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        word_lst, word_mapping, mask_mapping = self.sentence_preprocess(sentence, mask_last=True)
        mask_idx = list(np.where(mask_mapping == 1)[0])
        if len(sentence) in mask_idx:
            mask_idx = mask_idx.remove(len(sentence))
        if not mask_idx:
            return None
        if random.random() <= 0.15:
            # 选词
            res_mapping = [(i + 1) * j for i, j in zip(word_mapping, mask_mapping)]
            word_span = []
            l_idx, r_idx = 0, 1
            while r_idx < len(res_mapping):
                if res_mapping[l_idx] == res_mapping[r_idx]:
                    r_idx += 1
                else:
                    if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                        l_idx = r_idx
                        r_idx += 1
                    else:
                        word_span.append((l_idx, r_idx))
                        l_idx = r_idx
                        r_idx += 1

            if not word_span:
                return None
            word_span = random.choice(word_span)
            insert_word = np.random.choice(self.words)
            insert_idx = random.randint(0, 1)
            item["mode"] = "extra"
            item["error_sentence"] = sentence[:word_span[insert_idx]] + insert_word + sentence[insert_idx:]
            item["start_offset"] = word_span[insert_idx]
            item["end_offset"] = word_span[insert_idx] + len(insert_word)
            item["error_word"] = insert_word
            item["correct_word"] = ""

        else:
            sen_idx = random.choice(mask_idx)
            map_start = word_mapping.index(word_mapping[sen_idx])
            map_end = len(word_mapping) - word_mapping[::-1].index(word_mapping[sen_idx])
            sub_word = sentence[map_start:map_end]
            if len(sub_word) == 1:
                # 从字典中随机选择一个插入其前后
                if random.random() < 0.33:
                    candidate_word = random.choice(self.vocab)
                    item["mode"] = "extra"
                    item["error_sentence"] = sentence[:sen_idx] + candidate_word + sentence[sen_idx:]
                    item["start_offset"] = sen_idx
                    item["end_offset"] = sen_idx + 1
                    item["error_word"] = candidate_word
                    item["correct_word"] = ""
                elif random.random() < 0.67:
                    # tmp_item
                    candidate_word = random.choice(self.vocab)
                    item["mode"] = "extra"
                    item["error_sentence"] = sentence[:sen_idx + 1] + candidate_word + sentence[sen_idx + 1:]
                    item["start_offset"] = sen_idx + 1
                    item["end_offset"] = sen_idx + 2
                    item["error_word"] = candidate_word
                    item["correct_word"] = ""
                else:
                    item["mode"] = "extra"
                    item["error_sentence"] = sentence[:sen_idx] + sentence[sen_idx] * 2 + sentence[sen_idx + 1:]
                    item["start_offset"] = sen_idx + 1
                    item["end_offset"] = sen_idx + 2
                    item["error_word"] = sentence[sen_idx]
                    item["correct_word"] = ""

            else:
                # 以一定的比例按键盘布局来生成
                if random.random() < 0.6:
                    if random.random() < 0.5 or map_end >= len(sentence) - 1:
                        if self.extra_candidate.get("pre" + sub_word, ""):
                            candidate_word = self.extra_candidate["pre" + sub_word]
                            candidate_word = random.choice(candidate_word)
                            item["mode"] = "extra"
                            item["error_sentence"] = sentence[:map_start] + candidate_word + sentence[map_start:]
                            item["start_offset"] = map_start
                            item["end_offset"] = map_start + 1
                            item["error_word"] = candidate_word
                            item["correct_word"] = ''
                        else:
                            candidate_word = self.extra_candidate.get("post" + sub_word, '')
                            if candidate_word:
                                candidate_word = random.choice(candidate_word)
                                item["mode"] = "extra"
                                item["error_sentence"] = sentence[:map_end + 1] + candidate_word + sentence[map_end + 1:]
                                item["start_offset"] = map_end + 1
                                item["end_offset"] = map_end + 2
                                item["error_word"] = candidate_word
                                item["correct_word"] = ''
                else:
                    # 其他的则随机生成
                    if random.random() < 0.5 or map_end >= len(sentence) - 1:
                        candidate_word = random.choice(self.vocab)
                        item["mode"] = "extra"
                        item["error_sentence"] = sentence[:map_start] + candidate_word + sentence[map_start:]
                        item["start_offset"] = map_start
                        item["end_offset"] = map_start + 1
                        item["error_word"] = candidate_word
                        item["correct_word"] = ""
                    else:
                        # tmp_item
                        candidate_word = random.choice(self.vocab)
                        item["mode"] = "extra"
                        item["error_sentence"] = sentence[:map_end + 1] + candidate_word + sentence[map_end + 1:]
                        item["start_offset"] = map_end + 1
                        item["end_offset"] = map_end + 2
                        item["error_word"] = candidate_word
                        item["correct_word"] = ""
        if item['start_offset'] == 0 and item['end_offset'] == 0:
            return None
        else:
            return item

    def get_misssing_data(self, sentence):
        item = {
            "mode": "",
            "error_sentence": '',
            "start_offset": 0,
            "end_offset": 0,
            "error_word": '',
            'correct_word': ''
        }
        _, word_mapping, mask_mapping = self.sentence_preprocess(sentence)
        mask_idx = np.where(mask_mapping == 1)[0]
        if not mask_idx:
            return None
        if random.random() <= 0.15:
            # 选词
            res_mapping = [(i + 1) * j for i, j in zip(word_mapping, mask_mapping)]
            word_span = []
            l_idx, r_idx = 0, 1
            while r_idx < len(res_mapping):
                if res_mapping[l_idx] == res_mapping[r_idx]:
                    r_idx += 1
                else:
                    if r_idx - l_idx == 1 or sum(res_mapping[l_idx:r_idx]) == 0:
                        l_idx = r_idx
                        r_idx += 1
                    else:
                        word_span.append((l_idx, r_idx))
                        l_idx = r_idx
                        r_idx += 1
            word_span = random.choice(word_span)
            del_idx = np.random.randint(0, 1)
            item["mode"] = "missing"
            item["error_sentence"] = sentence[:word_span[del_idx]] + sentence[word_span[del_idx] + 1:]
            item["start_offset"] = word_span[del_idx]
            item["end_offset"] = word_span[del_idx] + 1
            item["error_word"] = ''
            item["correct_word"] = sentence[word_span[del_idx]]
            return item

        else:
            idx = random.choice(mask_idx)
            item["mode"] = "missing"
            item["error_sentence"] = sentence[:idx] + sentence[idx + 1:]
            item["start_offset"] = idx
            item["end_offset"] = idx + 1
            item["error_word"] = ''
            item["correct_word"] = sentence[idx]
        return item

    def sentence_preprocess(self, sentence, mask_last=False):
        """
        将句子中的停用词，实体，非中文字符过滤掉
        :param sentence:
        :param stop_words:
        :return: mask_flag_list [0, 0, 1, 0]; 0->不mask； 1-> 以概率mask
        """
        # 滤掉停用词， 分词， word_mapping
        word_lst, _, word_mapping = self.get_wordseg_result(sentence)
        masked_flag = np.ones(shape=(len(sentence,)), dtype=int)
        for idx, word in enumerate(word_lst):
            if word in self.stopwords:
                start_idx = word_mapping.index(idx)
                end_idx = len(word_mapping) - word_mapping[::-1].index(idx)
                masked_flag[start_idx: end_idx] = 0
            elif not self.is_chinese_char(word):
                start_idx = word_mapping.index(idx)
                end_idx = len(word_mapping) - word_mapping[::-1].index(idx)
                masked_flag[start_idx: end_idx] = 0

        # 句尾不设错
        if mask_last:
            if masked_flag[-1] == 1:
                masked_flag[-1] = 0
        #
        return word_lst, word_mapping, masked_flag

    def is_chinese_char(self, word):
        for w in word:
            if u'\u4e00' > w or w > u'\u9fff':
                return False
        return True

    def get_wordseg_result(self, text):
        """
        :param text:
        :return: {'wordseg_mapping': [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 11]}
        """
        wordseg_lst, wordpos_lst, inverse_idx_lst = [], [], []
        for idx, (word, pos) in enumerate(posseg.cut(text)):
            wordseg_lst.append(word)
            wordpos_lst.append(pos)
            inverse_idx_lst.extend([idx] * len(word))
        return wordseg_lst, wordpos_lst, inverse_idx_lst

    def cut_sentence(self, passage, splits=['。', '？', '\?', '！', '!']):
        matches = re.split("(" + '|'.join(splits) + ")", passage)
        values = matches[::2]
        delimiters = matches[1::2] + ['']
        pieces = [v + d for v, d in zip(values, delimiters)]
        result = []
        for piece in pieces:
            if len(piece) < 10 or self.digit_len(piece) > len(piece) / 2:
                continue
            else:
                result.append(piece)
        return result

    @staticmethod
    def digit_len(sentence):
        result = re.findall(r'\d+', sentence)
        length = sum([len(x) for x in result])
        return length

    def save_data(self, values):
        self.mx1.acquire()
        self.csv_writer.writerows(values)
        self.mx1.release()


class ThreadClass(threading.Thread):
    def __init__(self, target, args):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args
        self.result = []

    def run(self):
        self.result = self.target(*self.args)

    def get_results(self):
        return self.result


def main(args):
    if args.data_path.endswith('txt'):
        data = [[idx, line.strip()] for idx, line in enumerate(open(args.data_path, 'r').readlines())]
    elif args.data_path.endswith('csv'):
        data = pd.read_csv(args.data_path)
        data = data[['id', 'content']].values
    elif args.data_path.endswith('xlsx'):
        data = pd.read_excel(args.data_path)
        data = data[['id', 'content']].values
    elif args.data_path.endswith('json'):
        data = [[idx, json.loads(line)['content']] for idx, line in enumerate(open(args.data_path,
                                                                                   'r', encoding='utf-8'))]
    else:
        logger.info('the data format is not support !')
        return

    generator = DataGenerator(save_path=args.save_path)
    thread_lst = []
    sub_size = len(data) // args.thread_num if len(data) % args.thread_num == 0 \
        else len(data) // args.thread_num + 1

    for idx in range(args.thread_num):
        thread_lst.append(threading.Thread(target=generator.data_generate,
                                           args=(data[idx * sub_size:(idx+1) * sub_size if len(data) > (idx+1) * sub_size else len(data)],)))

    for thread in thread_lst:
        thread.setDaemon(True)
        thread.start()

    for thread in thread_lst:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('data augmentation parameters')
    parser.add_argument('--data_path', type=str, default='/data/weixin_public_corpus-master/articles.json',
                        help='the seed data to generate data')
    parser.add_argument('--save_path', type=str, default='../../data/correction_generator.csv',
                        help='the save data path')
    parser.add_argument('--thread_num', type=int, default=20,
                        help='the thread number to generate data')
    args = parser.parse_args()
    main(args)

