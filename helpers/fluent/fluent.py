import jieba
import time
from models import MaskedBert
from tqdm import tqdm

start_time = time.time()

model = MaskedBert.from_pretrained("bert-base-chinese")

print(f"Loading ngrams model cost {time.time() - start_time:.3f} seconds.")


def read_data(path):
    data = []
    with open(path, "r", encoding="utf8") as fin:
        for line in fin.readlines():
            items = line.strip().split('\t')
            data.append(items)
    return data


if __name__ == '__main__':
    src_path = '../../data/cged_test.txt'
    struct_bert_path = "../../data/struct_bert_output.txt"

    src_data = read_data(src_path)
    struct_bert_data = read_data(struct_bert_path)

    count, n, boost = 0, 0, []
    threshold = 0.3

    f2 = open("../../data/merge_output.txt", "w")
    for s1, s2 in tqdm(zip(src_data, struct_bert_data)):
        ppl1 = model.perplexity(x=jieba.lcut(s1[1]),
                                verbose=False,
                                )
        ppl2 = model.perplexity(x=jieba.lcut(s2[1]),
                                verbose=False,
                                )
        if s2[1] != s1[1] and ppl2 - ppl1 > threshold:

            f2.write(s1[0] + '\t' + s1[1] + '\t' + s2[1] + '不纠' + '\n')
        else:
            f2.write(s1[0] + '\t' + s1[1] + '\t' + s2[1] + '\n')
