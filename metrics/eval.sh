SRC_PATH=../data/cged_test.txt
#SRC_PATH=../exps/seq2edit_lang8/results/valid_data.txt
HYP_PATH=./samples/cged.pred
REF_PATH=./samples/truth_2021-mk.txt
OUTPUT_PATH=./samples/cged.pred.txt


python pair2edits_char.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH
#python pair2edits_word.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH

#perl evaluation.pl $OUTPUT_PATH ./samples/report.txt $REF_PATH
