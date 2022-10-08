SRC_PATH=./samples/src.txt
HYP_PATH=./samples/hyp.txt
REF_PATH=./samples/ref.txt
OUTPUT_PATH=./samples/output.txt

SRC_PATH=../data/cgec/valid_mk.src
#SRC_PATH=../exps/seq2edit_lang8/results/valid_data.txt
HYP_PATH=./samples/cged2021.pred
REF_PATH=./samples/truth_2021-mk.txt
OUTPUT_PATH=./samples/output.txt

#
#SRC_PATH=../data/lang8/source_test.txt
#HYP_PATH=../data/lang8/target.txt.char
#OUTPUT_PATH=./samples/lang8_test.txt

#python pair2edits_char.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH
python pair2edits_word.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH

perl evaluation.pl $OUTPUT_PATH ./samples/report.txt $REF_PATH
