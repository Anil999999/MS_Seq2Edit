## 数据生成策略

数据来源：微信公众号数据
         https://github.com/nonamestreet/weixin_public_corpus

本次比赛中拼写纠错任务和语法纠错任务都用到了数据增强策略，两次任务均使用微信语料作为种子，拼写任务数据生成策略只使用了音似和形似混淆集替换，
而语法纠错的数据生成包含所有的替换方法。

数据生成过程所需的数据文件说明：
```
├── data
│      ├── extra_candidate.json 冗余数据生成候选集
│      ├── pinyin_confusion_all.json 拼音混淆集
│      ├── shape_confusion_all.json  字形混淆集
│      ├── stop_words  停用词
│      ├── vocab.txt   字典
│      ├── word_confusion.json  词级别混淆集
│      ├── words.txt   
│      └── 生僻字.txt

```

拼写纠错任务数据生成参数设置：

|   参数     |  参数值   |    参数说明    |
|:---------:|:---------:|:------------:|
|  prob_all |  0.05   | 句子中字符替换的比例 | 
| pinyin_prob |  0.4       | 音似的替换比例 |
| shape_prob |   0.4       | 形似的替换比例 |
| random_prob | 0.2         | 随机的替换比例 |

除此之外，数据生成的过程中保留了一定比例的正确句子，以降低模型的误纠；这里保留的正确句子比例为0.3。

语法纠错任务中数据生成参数设置：

|   参数     |  参数值   |    参数说明    |
|:---------:|:---------:|:------------:|
|  prob_all |  0.05   | 句子中字符替换的比例 | 
| disorder_prob |  0.05      | 乱序的替换比例 |
| missing_prob   |    0.1      | 缺失的替换比例 |
| selection_prob |   0.7     | 用词错误的替换比例 |
| redundant_prob |   0.15    | 冗余的替换比例 | 

语法纠错的数据生成过程中同样保留了正确的句子，其设置与拼写纠错任务一致。

在数据生成的过程中，我们利用生僻字、停用词和非中文等策略保证生成过程中操作的字符为中文且随机替换策略不引入生僻字。

1. 安装环境
 pip install requirements.txt
2. 生成调用方法：
python data_generator.py --data_path DATA_PATH --save_path SAVE_PATH