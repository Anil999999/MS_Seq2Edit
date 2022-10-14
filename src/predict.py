# -*- coding: utf-8
import os
from transformers import BertModel
import torch
import sys
sys.path.append('../')
import tokenization
import argparse
from gector.gec_model import GecBERTModel
from gector.csc_model import CSCModel
import re
import pandas as pd
from openccpy import Opencc

vocab_path = os.getenv('vocab_path', 'vocab.txt')
# cc = OpenCC('t2s')
cc = Opencc()

def split_sentence(document: str, flag: str = "all", limit: int = 510):
    """
    Args:
        document:
        flag: Type:str, "all" 中英文标点分句，"zh" 中文标点分句，"en" 英文标点分句
        limit: 默认单句最大长度为510个字符
    Returns: Type:list
    """
    sent_list = []
    try:
        if flag == "zh":
            document = re.sub('(?P<quotation_mark>([。？！](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>([。？！])[”’"\'])', r'\g<quotation_mark>\n', document)  # 特殊引号
        elif flag == "en":
            document = re.sub('(?P<quotation_mark>([.?!](?![”’"\'])))', r'\g<quotation_mark>\n', document)  # 英文单字符断句符
            document = re.sub('(?P<quotation_mark>([?!.]["\']))', r'\g<quotation_mark>\n', document)  # 特殊引号
        else:
            document = re.sub('(?P<quotation_mark>([。？！….?!](?![”’"\'])))', r'\g<quotation_mark>\n',
                               document)  # 单字符断句符
            document = re.sub('(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’"\']))', r'\g<quotation_mark>\n',
                               document)  # 特殊引号

        sent_list_ori = document.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)
    except:
        sent_list.clear()
        sent_list.append(document)
    return sent_list


def predict_for_file(input_file, output_file, model, batch_size, log=True, intermediate_save_path=None, seg=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sents = [s.strip() for s in lines]
    subsents = []
    s_map = []
    for i, sent in enumerate(sents):  # 将篇章划分为子句，分句预测再合并
        if seg:
            subsent_list = split_sentence(sent, flag="zh")
        else:
            subsent_list = [sent]
        s_map.extend([i for _ in range(len(subsent_list))])
        subsents.extend(subsent_list)
    assert len(subsents) == len(s_map)
    predictions = []
    cnt_corrections = 0
    batch = []
    intermediate_data = []
    for sent in subsents:
        batch.append(sent.split())
        if len(batch) == batch_size:  # 如果数据够了一个batch的话，
            preds, cnt, interm_data = model.handle_batch(batch)
            assert len(preds) == len(batch)
            intermediate_data.extend(interm_data)
            predictions.extend(preds)
            cnt_corrections += cnt
            if log:
                for z in zip(batch, preds):
                    print("source： " + "".join(z[0]))
                    print("target： " + "".join(z[1]))
                    print()
            batch = []

    if batch:
        preds, cnt, interm_data = model.handle_batch(batch)
        assert len(preds) == len(batch)
        predictions.extend(preds)
        intermediate_data.extend(interm_data)
        cnt_corrections += cnt
        if log:
            for z in zip(batch, preds):
                print("source： " + "".join(z[0]))
                print("target： " + "".join(z[1]))
                print()
    intermediate_data = pd.DataFrame(intermediate_data, index=None)
    intermediate_data.to_csv(intermediate_save_path, index=False)
    assert len(subsents) == len(predictions)
    if output_file:
        with open(output_file, 'w') as f1:
            with open(output_file + ".char", 'w') as f2:
                results = ["" for _ in range(len(sents))]
                for i, ret in enumerate(predictions):
                    ret_new = [tok.lstrip("##") for tok in ret]
                    # ret = cc.convert("".join(ret_new))
                    ret = ''.join([cc.to_simple(x) for x in ret_new])
                    results[s_map[i]] += ret
                tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
                for ret in results:
                    f1.write(ret + "\n")
                    line = tokenization.convert_to_unicode(ret)
                    tokens = tokenizer.tokenize(line)
                    new_tokens = []
                    for idx, item in enumerate(tokens):
                        if '##' in item and idx == 0:
                            new_tokens.append(item.lstrip("##"))
                        elif '##' in item:
                            new_tokens[-1] = new_tokens[-1] + item.lstrip("##")
                        else:
                            new_tokens.append(item)
                    f2.write(" ".join(new_tokens) + "\n")
    return cnt_corrections


def csc_predict(model, input_file, output_path, seg=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sents = [s.strip() for s in lines]
    subsents = []
    s_map = []
    for i, sent in enumerate(sents):  # 将篇章划分为子句，分句预测再合并

        if seg:
            subsent_list = split_sentence(sent, flag="zh")
        else:
            subsent_list = [sent]
        s_map.extend([i for _ in range(len(subsent_list))])
        subsents.extend(subsent_list)
    assert len(subsents) == len(s_map)
    output = model.predict(subsents)
    cnt_corrections = 0
    for _origin, _output in zip(subsents, output):
        if _origin != _output:
            cnt_corrections += 1

    if output_path:
        with open(output_path, 'w') as f1:
            results = ["" for _ in range(len(sents))]
            for i, ret in enumerate(output):
                ret_new = [tok.lstrip("##") for tok in ret]
                # ret = cc.convert("".join(ret_new))
                ret = ''.join([cc.to_simple(x) for x in ret_new])
                results[s_map[i]] += ret
            for ret in results:
                f1.write(ret + "\n")
    return cnt_corrections


def main(args):
    # get all paths
    if args.csc_predict:
        model = CSCModel(args.csc_model_path)
        correct = csc_predict(model, args.input_file, args.output_file, args.seg)
        print('correct sentence', correct)

    else:
        model = GecBERTModel(vocab_path=args.vocab_path,
                             model_paths=args.model_path.split(','),
                             weights_names=args.weights_name.split(','),
                             max_len=args.max_len, min_len=args.min_len,
                             iterations=args.iteration_count,
                             min_error_probability=args.min_error_probability,
                             min_probability=args.min_probability,
                             min_replace_prob=args.min_replace_prob,
                             min_append_prob=args.min_append_prob,
                             min_delete_prob=args.min_delete_prob,
                             log=False,
                             confidence=args.additional_confidence,
                             is_ensemble=args.is_ensemble,
                             weigths=args.weights,
                             cuda_device=args.cuda_device)
        cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                           batch_size=args.batch_size, log=args.log, intermediate_save_path=args.intermediate_save_path, seg=args.seg)
        print(cnt_corrections)
        print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--csc_model_path',
                        default='../plm/bert')
    parser.add_argument('--csc_predict',
                        default=False,
                        type=bool)
    parser.add_argument('--model_path',
                        default="./exps/seq2edit_lang8/Best_Model_Stage_2.th",
                        help='Path to the model file'
                        )  # GECToR模型文件，多个模型集成的话，可以用逗号隔开
    parser.add_argument('--weights_name',
                        help='Path to the pre-trained language model',
                        default='./plm/chinese-struct-bert-large',
                        )  # 预训练语言模型文件，多个模型集成的话，每个模型对应一个PLM，可以用逗号隔开
    parser.add_argument('--vocab_path',
                        help='Path to the vocab file',
                        default='./data/output_vocabulary_chinese_char_hsk+lang8_5',
                        )  # 词表文件
    parser.add_argument('--input_file',
                        help='Path to the input file',
                        default="./data/cgec/valid.src.char"
                        )  # 输入文件，要求：预先分好词/字
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default="./exps/seq2edit_lang8/results/MuCGEC_test_plome.output")  # 输出结果文件
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length(all longer will be truncated)',
                        default=200)  # 最大输入长度（token数目），大于该长度的输入将被截断
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                              '(all longer will be returned w/o changes)',
                        default=0)  # 最小修改长度（token数目），小于该长度的输入将不被修改
    parser.add_argument('--batch_size',
                        type=int,
                        help='The number of sentences in a batch when predicting',
                        default=2)  # 预测时的batch大小（句子数目）
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model',
                        default=5)  # 迭代修改轮数
    parser.add_argument('--min_probability',
                        type=float,
                        help='The number of iterations of the model',
                        default=0)  # token级别最小修改阈值
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token',
                        default=0.0)  # Keep标签额外置信度
    parser.add_argument('--min_replace_prob',
                        type=float,
                        default=0)  #
    parser.add_argument('--min_delete_prob',
                        type=float,
                        default=0)
    parser.add_argument('--min_append_prob',
                        type=float,
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)  # 句子级别最小修改阈值
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)  # 是否进行模型融合
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)  # 不同模型的权重（加权集成）
    parser.add_argument('--cuda_device',
                        help='The number of GPU',
                        default=-1)  # 使用GPU编号
    parser.add_argument('--intermediate_save_path',
                        help='the model intermediate output path',
                        default='./exps/model_output.csv')
    parser.add_argument('--log',
                        action='store_true')  # 是否输出完整信息
    parser.add_argument('--seg',
                        action='store_true')  # 是否切分长句预测后再合并
    args = parser.parse_args()
    main(args)
