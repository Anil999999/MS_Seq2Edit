"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
import logging
import os
import sys
from time import time

import numpy
import numpy as np
import torch
from functools import reduce
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn import util
from pypinyin import lazy_pinyin
from gector.seq2labels_model import Seq2Labels
from gector.fluency_models import MaskedBert
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


# GECToR源代码提供了两种模型，分别是：1）gec_model.py; 2）seq2labels_model.py
# 真正意义上，GECToR进行预测和训练使用的是seq2labels模型，它是一个简单的encoder+classifier架构的模型，用于标注一个编辑序列（encode+tag）。
# 而模型集成、利用编辑序列获得目标序列、恢复模型等操作的逻辑，被作者写在了GecBERTModel这个类里。
# 它相当于对seq2labels模型的高级封装，用于使用训练好的seq2labels模型预测并输出纠错结果。
# 训练阶段，我们只训练seq2labels模型；而预测阶段，我们才需要用到完整的gec_model模型。

class GecBERTModel(object):
    def __init__(self,
                 vocab_path=None,
                 vocab=None,
                 model_paths=None,
                 weights_names=None,
                 weigths=None,
                 max_len=50,
                 min_len=3,
                 log=False,
                 iterations=3,
                 min_probability=0.0,
                 is_ensemble=True,
                 min_error_probability=0.0,
                 min_replace_prob=0.0,
                 min_delete_prob=0.0,
                 min_append_prob=0.0,
                 confidence=0,
                 resolve_cycles=False,
                 cuda_device=0,
                 append_noum=None
                ):
        """
        GECToR模型类的构造函数
        :param vocab_path: 词典路径
        :param model_paths: 训练好的模型路径（可以是多个模型，用于模型集成）
        :param weigths: 模型集成时，每个模型输出的权重
        :param max_len: 纠错最大长度，句子中超过这个长度的部分将不会被纠错
        :param min_len: 句子最小长度，小于这个长度的句子不会被纠错
        :param log: 是否输出日志
        :param iterations: 迭代纠正轮数（解决NAT缺陷）
        :param min_probability: token的最小纠错概率，小于它则不对当前token纠错
        :param model_name: 模型名称
        :param special_tokens_fix: 是否对[CLS]、[SEP]等token化(it should be 0 for BERT/XLNet and 1 for RoBERTa.)(这些都是BERT等预训练模型的特殊标记）
        :param is_ensemble: 是否进行模型集成
        :param min_error_probability: 句子的最小纠错概率，小于它则不对当前句子纠错
        :param confidence: 给$KEEP标签添加一个置信度bias，防止模型过多地纠错，属于一个小trick
        :param resolve_cycles: Todo 似乎没有用到
        :param cuda_device: 使用的GPU编号
        """
        self.model_weights = list(map(float, weigths)) if weigths else [1] * len(
            model_paths)  # 如果不预置模型权重，那么就默认各模型在集成时权重相同
        self.cuda_device = int(cuda_device)
        self.device = torch.device(
            "cuda:" + str(cuda_device) if self.cuda_device >= 0 and torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.min_len = min_len
        self.min_probability = min_probability
        self.min_error_probability = min_error_probability
        self.min_replace_prob = min_replace_prob
        self.min_delete_prob = min_delete_prob
        self.min_append_prob = min_append_prob
        self.vocab = Vocabulary.from_files(vocab_path) if vocab_path else vocab  # 从文件里读取词典
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.append_noum = ['我', '了', '的']
        self.tail_noum = ['的', '了']
        self.common_set = ['的', '了', '我', '你', '他', '着', '得', '地', '这', '那', '里', '会', '个', '还',
                           '只', '而', '是', '还', '会', '给']
        self.mask_bert = MaskedBert.from_pretrained(path='bert-base-chinese',
                                                    device=self.device,
                                                    sentence_length=256)
        self.resolve_cycles = resolve_cycles
        # set training parameters and operations
        self.indexers = []  # 各模型对应的indexer
        self.models = []  # 各模型实例化的对象

        for model_path, weights_name in zip(model_paths, weights_names):  # 遍历所有模型保存的参数文件，重新获取模型对象（反序列化）
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            self.indexers.append(self._get_indexer(weights_name))  # 获取模型对应的indexer
            model = Seq2Labels(vocab=self.vocab,
                               text_field_embedder=self._get_embbeder(weights_name),
                               confidence=self.confidence,
                               cuda_device=self.cuda_device,
                              ).to(self.device)  # 实例化当前选择的预训练MLM对应的seq2labels模型对象（真正用来预测编辑序列的模型）
            # 读取已完成训练的seq2labels模型的参数
            pretrained_dict = torch.load(model_path,
                                         map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.eval()  # 将seq2labels模型设为预测模式，准备预测输出（实际上，gec_model只用在预测阶段）
            self.models.append(model)

    @staticmethod
    def _get_model_data(model_path):
        """
        根据模型的路径，提取出模型的名称，以及该模型是否需要将[CLS]、[SEP]等token化。如：bert_0_gector.th，model_name为bert，stf为0。
        :param model_path:模型路径
        :return:模型的名称，以及该模型是否需要将[CLS]、[SEP]等token化
        """
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def predict(self, batches):
        """
        批量预测推理，返回当前batch的句子被不同模型预测的结果。
        预测结果包括：
        preds: 当前batch内，各源句子的各词，标注为各编辑label的最大概率
        idx: 当前batch内，各源句子的各词，标注为所有编辑label的最大概率索引
        error_probs: 当前batch内，各源句子的出错概率
        :param batches: 所有模型针对当前batch的句子instance表示
        :return:当前batch的句子被不同模型预测的结果
        """
        with torch.cuda.device(self.cuda_device):
            t11 = time()
            predictions = []
            for batch, model in zip(batches, self.models):  # 每个模型预测自己对应的batch
                batch = util.move_to_device(batch.as_tensor_dict(), self.cuda_device if torch.cuda.is_available() else -1)
                with torch.no_grad():  # 这里是预测，不需要累计梯度
                    prediction = model.forward(**batch)  # 通过seq2labels模型的前向传播，计算当前batch各word-token处的预测结果
                predictions.append(prediction)

            preds, idx, error_probs, d_tags_idx, d_tag_prob, all_corr_probs = self._convert(predictions)
            t55 = time()
            if self.log:
                print(f"Inference time {t55 - t11}")
            return preds, idx, error_probs, d_tags_idx, d_tag_prob, all_corr_probs

    def get_token_action(self, index, prob, sugg_token, iter_num, length):
        """
        从编辑label转化为edit。
        label是word-token级别的，对单个词进行纠正，也是模型预测的结果。
        edit是span级别的，对某个范围进行纠正。
        :param token: 当前word-token
        :param index: 当前word-token的索引
        :param prob: 当前word-token的最大概率编辑label的概率
        :param sugg_token: 当前word-token的最大概率编辑label
        :return:编辑开始位置，编辑结束位置，编辑内容（比如要替换的词），当前编辑操作的概率
        """
        start_pos = 0
        end_pos = 0
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if (prob < self.min_probability and not sugg_token.startswith('$MOVE')) or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if iter_num == 0 and prob < 0.6:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') \
                or sugg_token == '$DELETE' or sugg_token.startswith('$MOVE'):
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    @staticmethod
    def _get_embbeder(weight_name):
        """
        获取模型对应的embbeder，用于将token进行嵌入，转为对应的特征向量表示
        :param weigth_name: 模型名称
        :param special_tokens_fix: 是否需要对MLM模型的特殊标记token化，如[CLS]等
        :return: 模型对应的embbeder
        """
        bert_token_emb = PretrainedTransformerEmbedder(model_name=weight_name, last_layer_only=True,
                                                       train_parameters=False)
        token_embedders = {'bert': bert_token_emb}
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=token_embedders)
        return text_field_embedder

    def _get_indexer(self, weight_name):
        """
        获取模型对应的indexer（编号器，用于完成token<—>id的转换）
        :param weight_name: 模型名称
        :return: 模型对应的indexer
        """
        bert_token_indexer = PretrainedTransformerIndexer(model_name=weight_name, namespace="bert")
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        """
        对即将进行预测的句子batch进行预处理
        :param token_batch: 一个token_batch，如：[['I', 'have', 'book.'], ['I', 'likes', 'to', 'swimming.'], ['I', 'am', 'fine.']]
        :return: 各模型对应的instance化后的token_batch，例如要ensemble三个模型，最后返回一个包含三个模型的instance化token_batch的列表
        """
        with torch.cuda.device(self.cuda_device):
            seq_lens = [len(sequence) for sequence in token_batch if sequence]  # 得到每个句子的长度
            if not seq_lens:
                return []
            max_len = min(max(seq_lens), self.max_len)  # 当前batch内的句子实际最大长度和规定的长度上限取一个最小值，对于每个句子超过这个长度的部分不进行处理（即不纠错）
            batches = []
            for indexer in self.indexers:  # 遍历各模型对应的indexer（模型ensemble的需要，不同模型的indexer不相同，因为采用了不同的预训练模型）
                batch = []
                for sequence in token_batch:  # 遍历token_batch中的所有句子
                    tokens = sequence[:max_len]
                    tokens = [Token(token) for token in ['$START'] + tokens]  # 进行allennlp的Token化处理
                    batch.append(Instance({'tokens': TextField(tokens, indexer)}))
                batch = Batch(batch)  # 进行allennlp的Batch化处理
                batch.index_instances(self.vocab)  # 使用之前传入的tokens域的indexer，为当前batch所有instance的tokens域进行index操作
                batches.append(batch)  # 把当前模型的batch加入进去
            return batches

    def _convert(self, data):
        """
        转换seq2labels模型前向传播的输出，提取 probs, idx, error_probs三个部分
        :param data: seq2labels模型前向传播的输出
        :return:
        probs: 当前batch内，各源句子的各词，标注为所有编辑label的概率的最大值
        idx: 当前batch内，各源句子的各词，标注为所有编辑label的概率的最大值索引
        error_probs: 当前batch内，各源句子的出错概率
        """
        all_class_probs = torch.zeros_like(
            data[0]['class_probabilities_labels'])  # 当前batch内，各源句子的所有词标注为各label的概率，维度：[batch_size,seq_len,label_num]
        error_probs = torch.zeros_like(
            data[0]['max_error_probability'])  # 当前batch内，各源句子出错的概率（用在一个trick，当前句子如果错误概率过小，就不纠错了），维度为：[batch_size]
        d_tags_class_probs = torch.zeros_like(
            data[0]['class_probabilities_d_tags'])
        incorrect_prob = torch.zeros_like(data[0]['error_probability'])
        for output, weight in zip(data, self.model_weights):  # 用于模型ensemble，最后得到的概率是：各模型计算出来的概率的加权平均数
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            d_tags_class_probs += weight * output['class_probabilities_d_tags'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)
            incorrect_prob += weight * output['error_probability'] / sum(self.model_weights)

        d_tags_idx = torch.max(d_tags_class_probs, dim=-1)[1]
        max_vals = torch.max(all_class_probs, dim=-1)  # torch.max()函数会返回两个tensor，一个是最大值（这里是第3维的最大值），一个是最大值的索引，维度：[batch_size,seq_len]
        probs = max_vals[0].tolist()  # 获得各源句子的各词标注为所有编辑label的概率中最大值，维度:[batch_size,seq_len]
        idx = max_vals[1].tolist()  # 获得各源句子的各词标注为所有编辑label的概率中最大值的索引，维度:[batch_size,seq_len]
        return probs, idx, error_probs.tolist(), d_tags_idx.tolist(), incorrect_prob.tolist(), all_class_probs

    def update_final_batch(self, final_batch, pred_ids, pred_batch,
                           prev_preds_dict):
        """
        更新最终的纠错结果（需要有个迭代纠正的过程）
        :param final_batch: 最终的纠错结果列表
        :param pred_ids: 需要预测的句子id
        :param pred_batch: 当前迭代轮次的纠错结果列表
        :param prev_preds_dict: 之前迭代轮次的纠错结果列表
        :return:
        final_batch: 更新后的最终纠错结果列表
        new_pred_ids: 预测了多少个新句子
        total_updated: 当前轮次更新了多少句子
        """
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]  # 上一轮纠正结果
            pred = pred_batch[i]  # 本轮纠正结果
            prev_preds = prev_preds_dict[orig_id]  # 之前出现过的纠正结果
            if orig != pred and pred not in prev_preds:  # 纠错，并且纠错结果之前从未出现过
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:  # 纠错，但纠错结果之前出现过
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs,
                          error_probs, all_d_tags_idxs, all_correct_probs,
                          max_len=50, iter_num=0):
        """
        后处理，根据推理得到的结果，对源句子序列纠错，得到最终的句子
        :param batch: 当前批次要预测的句子（字符串列表，没有被instance化）
        :param all_probabilities: 每个句子的每个词处，预测的各编辑label的最大概率
        :param all_idxs: 每个句子的每个词处，预测的各编辑label的最大概率者的索引
        :param error_probs: 每个句子的出错概率
        :param max_len: 纠错的最大长度
        :param iter_num: 当前纠错的轮次，主要对第一轮次的概率进行干预
        :return: 完成纠错的句子
        """
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")  # 获得跳过操作的下标（即$KEEP编辑，保持当前词不变）
        for tokens, probabilities, idxs, error_prob, d_tags_idxs, correct_probs in zip(batch,
                                                                                       all_probabilities,
                                                                                       all_idxs,
                                                                                       error_probs,
                                                                                       all_d_tags_idxs,
                                                                                       all_correct_probs):
            length = min(len(tokens), max_len)  # 纠错长度（超过不纠错）
            edits = []  # 编辑操作，例子：[(1, 2, '$TRANSFORM_VERB_VBZ_VB', 0.9100876450538635)]

            # 跳过无需纠错的句子
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # 论文中提到的trick：模型会在源句子每个词处预测一个额外的tag：当前词是否出错。在所有词处预测出错的概率的最大值，可以看成是整个句子出错的概率。
            # 如果当前句子出错的概率过小，那么就不纠错（因为模型会尽可能纠错）
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue
            candidate_path, candidate_prob = [], []
            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # 假设当前词处预测的编辑是$KEEP，就不需要修改
                if idxs[i] == noop_index:
                    continue
                # 从vocab的labels域里，获取当前词处预测的编辑操作（除$KEEP外）
                cur_prob = correct_probs[i].cpu().numpy()
                cur_prob = list(enumerate(cur_prob))
                cur_prob = sorted(cur_prob, key=lambda x: x[1], reverse=True)
                label_idx = [x for x, y in cur_prob]
                label_prob = [y for x, y in cur_prob]
                candidate_sugg, can_prob = [], []
                for _idx, _prob in zip(label_idx[:5], label_prob[:5]):
                    _sugg = self.vocab.get_token_from_index(_idx, namespace='labels')
                    # if _prob > 0.1 and _sugg not in ['$KEEP', UNK, PAD]:
                    candidate_sugg.append(_sugg)
                    can_prob.append(_prob)
                sugg_token = self.vocab.get_token_from_index(idxs[i], namespace='labels')

                # 将编辑label转edit
                action = self.get_token_action(i, probabilities[i], sugg_token, iter_num, length)

                if not action: continue
                # 将当前edit暂存
                # candidate_edit.append(action[2])
                edits.append(action)
                candidate_path.append(candidate_sugg)
            # 对源句子使用edits序列，得到纠错后的句子
            print('origin_edits', edits)
            post_edits = self.action_postprocess(tokens, edits, correct_probs)
            print('post_edits', post_edits)
            all_results.append(get_target_sent_by_edits(tokens, post_edits))
        return all_results

    def get_probs(self, probs):
        cur_prob = probs.cpu().numpy()
        cur_prob = list(enumerate(cur_prob))
        cur_prob = sorted(cur_prob, key=lambda x: x[1], reverse=True)
        label_idx = [x for x, y in cur_prob]
        label_prob = [y for x, y in cur_prob]
        return label_idx, label_prob
    
    def action_postprocess(self, tokens, edits, correct_probs):
        post_edit = []
        idx = 0
        move_span = []
        ## todo： 修改后处理逻辑
        while idx < len(edits):
            if edits[idx][2] and edits[idx][2][-1].isdigit():
                move_span.append(edits[idx])
                pre_idx = edits[idx][0]
                idx += 1
                j = idx
                if j >= len(edits):
                    break
                while j < len(edits):
                    if edits[j][2] and edits[j][2][-1].isdigit() and edits[j][0] == pre_idx + 1:
                        pre_idx += 1
                        move_span.append(edits[j])
                    else:
                        break
                    j += 1
                if len(move_span) == 1:
                    move_span = move_span[0]
                    cur_prob = correct_probs[int(move_span[0]) + 1]
                    _idx, _prob = self.get_probs(cur_prob)
                    sugg = self.vocab.get_token_from_index(_idx[1], namespace='labels')
                    if sugg in [UNK, PAD, '$KEEP'] or sugg.startswith('$MOVE_'):
                        move_span = []
                        continue
                    sugg_edit = sugg.split('_')[-1] if sugg != '$DELETE' else ''
                    post_edit.append((move_span[0], move_span[1], sugg_edit, _prob[1]))
                else:
                    result = self.dilate_move_prob(tokens, correct_probs, move_span)
                    if result is not None:
                        post_edit.extend(result)
                idx = j
                move_span = []
            else:
                post_edit.append(edits[idx])
            idx += 1
        result_edit = []
        start_offset = [x[0] for x in post_edit]
        offset2edit = {}
        for _edit in post_edit:
            if _edit[0] not in offset2edit:
                offset2edit[_edit[0]] = [_edit]
            else:
                offset2edit[_edit[0]].append(_edit)
        for _offset in start_offset:
            _edit = offset2edit[_offset]
            if len(_edit) > 1:
                candidate_edit = []
                for item in _edit:
                    if item[2] and item[2][-1].isdigit():
                        candidate_edit.append(item)
                if len(candidate_edit) == 1:
                    result_edit.extend(candidate_edit)
                else:
                    return []
            else:
                result_edit.append(_edit[0])
        for _edit in result_edit:
            if abs(int(_edit[2])) > len(result_edit):
                return []
        return result_edit

    def dilate_move_prob(self, tokens, probs, move_span):
        start_idx = move_span[0][0]
        end_idx = move_span[-1][0]
        top_path, top_probs = [], []
        end_pos = end_idx + 5 if end_idx + 5 < len(probs) - 1 else len(probs) - 1
        for idx in range(start_idx-2, end_pos):
            _idx, _prob = self.get_probs(probs[idx+1])
            top_path.append([self.vocab.get_token_from_index(x, namespace='labels') for x in _idx[:5]])
            top_probs.append(_prob[:5])
        candidate_path, candidate_probs = [], []
        candidate_path.append([])
        candidate_probs.append([])
        for idx in range(len(top_path)):
            cur_path, tmp_path, cur_prob, tmp_prob = [], [], [], []
            for edit, prob in zip(top_path[idx], top_probs[idx]):
                if edit.startswith('$MOVE_'):
                    cur_path.append(int(edit.split('_')[1]))
                    cur_prob.append(prob)
                elif edit == '$KEEP':
                    cur_path.append(0)
                    cur_prob.append(prob)
            tmp_path = candidate_path[:]
            tmp_prob = candidate_probs[:]
            candidate_path, candidate_probs = [], []
            for _path, _prob in zip(cur_path, cur_prob):
                for value, prob in zip(tmp_path, tmp_prob):
                    if 0 in value[2:]:
                        first_zero = value[2:].index(0) + 2
                    if 0 not in value and len(set(value)) > 2:
                        continue
                    if len(value) > 2 and value[0] != 0 and value[1] == 0:
                        continue
                    if value and len(value) > 2 and 0 not in value[2:] \
                            and len(set(value[2:])) > 2:
                        continue
                    if 0 in value[2:] and sum(value[first_zero:]) != 0:
                        continue
                    candidate_path.append(value + [_path])
                    candidate_probs.append(prob + [_prob])
        candidate_path = np.asarray(candidate_path)
        # if
        sum_all = np.sum(candidate_path, axis=-1)
        last_path, last_start = [], []
        if isinstance(sum_all, numpy.float64):
            return None
        for _idx, _sum in enumerate(sum_all):
            if _sum != 0:
                continue
            _start, _end = -1, len(candidate_path[_idx]) + 1
            for idx, value in enumerate(candidate_path[_idx]):
                if value != 0 and _start == -1:
                    _start = idx
                    continue
                if 0 <= _start < idx and value == 0:
                    _end = idx
                    break
            if _end > len(candidate_path[_idx]):
                _end = len(candidate_path[_idx])
            if _start == -1:
                _start = 0
            _path = candidate_path[_idx][_start: _end]
            if 0 in _path:
                continue
            if sum(_path) != 0:
                continue
            pos_sidx = list(_path).index(int(_path[0]))
            pos_eidx = _path.shape[0] - list(_path)[::-1].index(int(_path[0]))
            if len(set(_path[pos_sidx: pos_eidx])) != 1:
                continue
            if len(set(_path)) == 2:
                last_path.append(candidate_path[_idx])
                last_start.append(_start)
        if not last_path:
            return None
        candidate_edits = []
        origin_start = start_idx
        for _last_path, _start in zip(last_path, last_start):
            result, edit_offset = [], []
            start_idx = origin_start
            for idx, _offset in enumerate(_last_path):
                if idx < _start:
                    continue
                if idx < len(move_span) and _offset == 0:
                    break
                if idx == 0 and _offset != 0:
                    edit_offset.append(_offset)
                    result.append((start_idx - 2, start_idx-1, str(_offset), 0.5))
                elif idx == 1 and _offset != 0:
                    edit_offset.append(_offset)
                    result.append((start_idx - 1, start_idx, str(_offset), 0.5))
                elif idx < len(move_span):
                    edit_offset.append(_offset)
                    result.append((start_idx, start_idx + 1, str(_offset), 0.5))
                    start_idx += 1
                elif _offset != 0:
                    edit_offset.append(_offset)
                    result.append((start_idx, start_idx + 1, str(_offset), 0.5))
                    start_idx += 1
            if len(set(edit_offset)) == 2:
                idx_count = 0
                for _edit in result:
                    if abs(int(_edit[2])) > len(result):
                        idx_count += 1

                if idx_count == 0:
                    candidate_edits.append(result)
        if not candidate_edits:
            return None
        candidate_tokens = [get_target_sent_by_edits(tokens, _edits) for _edits in candidate_edits]
        candidate_tokens.insert(0, tokens)
        ppl_score = []
        for tokens in candidate_tokens:
            _score = self.mask_bert.perplexity(' '.join(tokens), verbose=False,
                                               temperature=1.0, batch_size=16)
            ppl_score.append(_score)
        min_idx = np.argmin(ppl_score)
        if min_idx == 0:
            return None
        return candidate_edits[min_idx-1]

    def handle_batch(self, full_batch):
        """
        预测一个batch的句子的纠错结果
        :param full_batch: bacth化的句子数据
        :return:
        """
        intermediate_data = []
        final_batch = full_batch[:]  # 纠错后的结果列表
        batch_size = len(full_batch)  # batch规模
        prev_preds_dict = {i: [final_batch[i]] for i in
                           range(len(final_batch))}  # 构造一个{句子id->句子}的映射，用于存放某个句子的原始状态和在不同轮次纠正的所有结果
        short_ids = [i for i in range(len(full_batch))
                     if len(full_batch[i]) < self.min_len]  # 把长度小于min_len的句子的id放到short_ids中，不进行预测
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]  # 把长度大于等于min_len的句子的id放到pred_ids中，准备预测
        total_updates = 0  # 记录模型纠错次数

        # 迭代纠正（iterative refine），迭代轮数由self.iterations确定
        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]  # 当前迭代轮次中，需要预测的句子（来自于上一个迭代轮次or初始句子）

            sequences = self.preprocess(orig_batch)  # 把原始的句子预处理，每个模型一个batch，每个batch包含batch-size个句子的instance
            if not sequences:
                break
            probabilities, idxs, error_probs, d_tags_idxs, d_tags_prob, all_probs = self.predict(
                sequences)  # 预测，得到每个句子的每个词对应的编辑label的最大概率、最大概率label对应的索引、每个句子错误的概率

            pred_batch = self.postprocess_batch(orig_batch, probabilities,
                                                idxs, error_probs, d_tags_idxs, all_probs, self.max_len, n_iter)  # 预测后进行处理，得到完成纠错的句子
            for _org, _pred, _prob, _idx, _error, _d_tag_prob, _d_tag_idx in zip(orig_batch, pred_batch, probabilities, idxs, error_probs, d_tags_prob, d_tags_idxs):
                intermediate_data.append([n_iter, "".join(_org), "".join(_pred), _prob, _idx, _error, _d_tag_prob, _d_tag_idx])
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100 * len(pred_ids) / batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = \
                self.update_final_batch(final_batch, pred_ids, pred_batch,
                                        prev_preds_dict)  # 更新纠错后的句子列表，作为下一个迭代轮次的输入
            total_updates += cnt

            if not pred_ids:  # 如果本轮纠正，没有出现新的结果，那么就提前停止迭代纠正（early-stopping）
                break
        return final_batch, total_updates, intermediate_data
