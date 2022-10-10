## CCL-2022 中文文本纠错 赛道二
### 本次比赛的主页 https://github.com/blcuicall/CCL2022-CLTC
### 比赛过程中使用的开源资源
cbert微调模型: https://github.com/xueyouluo/speller-bert
微信公众号语料: https://github.com/nonamestreet/weixin_public_corpus
流畅度计算方法: https://github.com/baojunshan/nlp-fluency

```
├── data
│       └── idioms.txt
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
│         └── idiom_correct.py
├── metrics
│         ├── README.md
│         ├── eval.sh
│         ├── evaluation.pl
│         ├── pair2edits_char.py
│         ├── pair2edits_word.py
│         ├── requirements.txt
│         ├── run_eval.sh
│         └── samples
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
└── utils
    ├── helpers.py
    └── preprocess_data.py
```
