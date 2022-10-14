## CCL-2022 中文文本纠错 赛道二
### 本次比赛的主页 https://github.com/blcuicall/CCL2022-CLTC
### 比赛过程中使用的开源资源
cbert微调模型: https://github.com/xueyouluo/speller-bert
微信公众号语料: https://github.com/nonamestreet/weixin_public_corpus
流畅度计算方法: https://github.com/baojunshan/nlp-fluency

模型训练及推理过程中需要的资源：
cbert pytorch版本 链接: https://pan.baidu.com/s/17caa2BGit9QitRqQjKriXg 提取码: g810
chinese-struct-bert-large 链接: https://pan.baidu.com/s/1ha020XhoGN1j9f5ktekdgQ 提取码: rjbf
拼写纠错使用的数据： 链接: https://pan.baidu.com/s/13TMzn26eTpqWA84-wBWFXA 提取码: bqsj

###代码目录结构如下：
```
├── data
│       │── cgec: CGED历年数据
│       │── lang8: lang8数据
│       │── output_only_move: 乱序数据单模型字典
│       │── output_vocab: 常用字词典
│       │── output_vocab_more: move词典
│       │── output_vocabulary: 扩展词典
│       │── vocab: 词典
│       │── cged_test.txt: 评测数据
│       └── verb-form-vocab.txt 英文动词变换词典
├── gector: 主体语法纠错
│         ├── csc_model.py
│         ├── datareader.py
│         ├── fluency_models.py
│         ├── gec_model.py
│         ├── gec_model_w.py
│         ├── seq2labels_metric.py
│         └── seq2labels_model.py
├── helpers
│         ├── csc_projects
│         │         ├── config.py
│         │         ├── dataset.py
│         │         ├── decode.py
│         │         ├── decode.sh
│         │         ├── eval_char_level.py
│         │         ├── eval_sent_level.py
│         │         ├── models.py
│         │         ├── pipeline.sh
│         │         ├── train.py
│         │         └── utils.py
│         ├── data_augmentation
│         │         ├── README.md
│         │         └── data_generator.py
│         ├──fluent
│         │      ├──fluent.py
│         │      └──model.py
│         │
│         └──idiom_correct
│                  ├──data
│                  └──idiom_correct.py
├── metrics 
│         ├── README.md
│         ├── eval.sh
│         ├── evaluation.pl
│         ├── pair2edits_char.py
│         ├── pair2edits_word.py
│         ├── requirements.txt
│         ├── run_eval.sh
│         └── samples
├── plm 预训练模型
├── src
│         ├── pipeline.sh
│         ├── predict.py
│         ├── predict.sh
│         ├── predict_w.py
│         ├── predict_w.sh
│         ├── tokenization.py
│         └── train.py
├── tools
│         └── segment
│             ├── segment_bert.py
│             └── tokenization.py
└── utils 数据预处理
    ├── helpers.py
    └── preprocess_data.py
```
1. 安装依赖环境
```
pip install requirements.txt
```

2. 训练
标签体系加入move操作： sh pipeline_w.sh

3. 推理
通用模型的推理脚本: sh predict.sh
标签体系修改之后的推理脚本: sh predict_w.sh

### 模型推理的流程为：
1. 利用csc_project解决拼写错误
2. GECTor模型解决语法错误（模型集成）
3. 成语俗语后处理
4. 利用困惑度比较原始输入和不同模型的输出之间的困惑度值的大小，择优输出
