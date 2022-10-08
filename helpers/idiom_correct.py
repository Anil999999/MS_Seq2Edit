import os
import re
import traceback
from collections import defaultdict
import Levenshtein

REGEX_BATCH_SIZE = 10000
delimiters = {'?', '!', ';', ',', '.', '？', '！', '。', '；', '，', '、', '\n', ' ', "。", "，",
              "、", "！", "？", "@", "——", "＿", "：", "；", "～", "......", "（", "）", "【", "】",
              "《", "》", "(", ")", ''}


class IdiomCorrection:
    def __init__(self, pattern_path=None):
        self._pattern_path = pattern_path
        self._bigram_inverted_dic = defaultdict(list)

    def load(self):
        word_file = open(self._pattern_path, 'r', encoding='utf-8')
        with open(self._pattern_path, 'r', encoding='utf-8')
            for line in word_file:
                word = line.strip()
                if word:
                    for idx in range(0, len(word) - 1):
                        self._bigram_inverted_dic[word[idx: idx+2]].append(word)
        word_file.close()

    def error_check(self, content):
        check_ret = []

        error_indexs = []
        for pattern in self._pattern_lst:
            for di in pattern.finditer( content ):
                start_offset, end_offset, di_str = di.start(), di.end(), di.group( 0 )
                if (start_offset, end_offset) in error_indexs:
                    continue
                else:
                    error_indexs.append( (start_offset, end_offset) )
                error_correction = self.get_simi_word( di_str )
                if not re.search( '^[\u4e00-\u9fa5]+$', di_str ) or error_correction == []:
                    continue
                white_list = [(['联席主承销商'], '联席承销商'), (['承销协议之补充协议'], '承销协议补充协议')]
                if (error_correction, di_str) in white_list:
                    continue
                # 后处理：切分过长的词
                start_offset, end_offset, di_str, error_correction = shorten_result( start_offset, end_offset, di_str,
                                                                                     error_correction )

                item = {
                    "error_type": "semantic_error",
                    "sub_error_type": ERROR_TYPE,
                    "level": ErrorLevel.SENTIMENT_DEFINITE_ERROR,
                    "start_offset": start_offset,
                    "end_offset": end_offset - 1,
                    "origin": di_str,
                    "error_correction": error_correction,
                    "reference": [],
                    "tips": NOMENCLATURE_ERROR
                }
                if self.item_postprocess( content, item ):
                    continue
                check_ret.append( item )
        return check_ret

    def get_simi_word(self, regex_match_str):
        candi_ret = []
        tmp_ret = []
        for idx in range( 0, len( regex_match_str ) - 1 ):
            if regex_match_str[idx:idx + 2] in self._bigram_inverted_dic:
                tmp_ret.extend( self._bigram_inverted_dic[regex_match_str[idx:idx + 2]] )
        tmp_ret = list( set( tmp_ret ) )
        for simi_word in tmp_ret:
            norm_dist = Levenshtein.distance( regex_match_str, simi_word )
            if norm_dist <= 1:
                candi_ret.append( simi_word )
        return candi_ret

    def item_postprocess(self, content, item):
        """根据ngram比较纠错前后句子通顺程度，进行后处理"""
        start_offset = item['start_offset']
        end_offset = item['end_offset']
        while re.search( r'[%sA-Z“”\"]' % r'\u3400-\u9FFF', content[start_offset] ):
            start_offset -= 1
            if start_offset == -1:
                break
        while end_offset < len( content ) and re.search( r'[%sA-Z“”\"]' % r'\u3400-\u9FFF', content[end_offset] ):
            end_offset += 1
        origin_s = content[start_offset + 1:end_offset]
        origin_s_prob = self._ngram_checker.check_sentence( origin_s, raw_only=True )[2]
        cor_s = origin_s.replace( item['origin'], item['error_correction'][0] )
        cor_prob = self._ngram_checker.check_sentence( cor_s, raw_only=True )[2]
        # 阈值为1
        if cor_prob > origin_s_prob * 0.8:
            return False
        else:
            return True
