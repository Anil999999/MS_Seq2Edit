## 拼写纠错模型

通过拼写纠错模型对数据中的形似音近错误进行纠正；该部分是数据是微信语料生成的数据，模型结构参考自[MDCSpell](https://aclanthology.org/2022.findings-acl.98) 。

实验过程： 首先，针对对于官方提供的数据进行用词错误数据抽取，然后利用混淆集对错误进行后处理；在实验对比过程中，因为同屋/室友等标注不一致的数据会引起模型的误召回量较多，
因此在该实验中，我们通过数据增强的方式，生成同音形似的数据。

预训练模型为bert-base-chinese

1. 安装依赖环境
```
pip install requirements.txt
```
2. 训练
```
sh pipeline.sh
```

3. 推理
```
sh decode.sh
```

