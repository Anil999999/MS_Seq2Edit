import sys
import tokenization
from tqdm import tqdm
import os 

vocab_path = os.getenv('vocab_path')
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(vocab_path), do_lower_case=False)

for line in tqdm(sys.stdin):
    line = line.strip()
    origin_line = line
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        continue
    tokens = tokenizer.tokenize(line)
    print(' '.join(tokens))
