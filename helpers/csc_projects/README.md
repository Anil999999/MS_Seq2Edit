## 拼写纠错模型

通过拼写纠错模型对数据中的形似音近错误进行纠正；该部分是数据是微信语料生成的数据，模型结构参考自[MDCSpell](https://aclanthology.org/2022.findings-acl.98) 。

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

