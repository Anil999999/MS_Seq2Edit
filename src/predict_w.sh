#
CUDA_DEVICE=1
SEED=42
#

MODEL_DIR=../exps/seq2edit/model_only_move
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

#PRETRAIN_WEIGHTS_DIR=../plm/chinese-macbert-large
PRETRAIN_WEIGHTS_DIR=../plm/chinese-struct-bert-large

VOCAB_PATH=./data/output_only_move

# Step3. Inference
MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
RESULT_DIR=$MODEL_DIR"/results"

#INPUT_FILE=../data/cgec/valid.src
INPUT_FILE=../data/cged_test.txt

echo $INPUT_FILE
export vocab_path=../plm/chinese-struct-bert-large/vocab.txt

CSC_OUTPUT=$RESULT_DIR"/csc.output"

if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict_w.py --csc_predict True \
                    --input_file $INPUT_FILE \
                    --output_file $CSC_OUTPUT \

if [ ! -f $CSC_OUTPUT".char" ]; then
    python ./tools/segment/segment_bert.py < $CSC_OUTPUT > $CSC_OUTPUT".char"  # 分字
fi

if [ ! -f $INPUT_FILE".char" ]; then
    python ./tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi

if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi

OUTPUT_FILE=$RESULT_DIR"/seq2edit_cged_test.output"
ITERATION_COUNT=5
MIN_PROB=0.25
MIN_ERROR_PROB=0.35

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict_w.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --cuda_device 0\
                  --input_file $CSC_OUTPUT".char"\
                  --min_error_probability $MIN_ERROR_PROB\
                  --min_probability $MIN_PROB\
                  --iteration_count $ITERATION_COUNT\
                  --output_file $OUTPUT_FILE --log

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
