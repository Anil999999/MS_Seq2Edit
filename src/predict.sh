#
CUDA_DEVICE=1
SEED=42
#

MODEL_DIR=../exps/seq2edit/models
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

PRETRAIN_WEIGHTS_DIR=../plm/chinese-struct-bert-large

VOCAB_PATH=../data/output_vocabulary

# Step3. Inference
MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
RESULT_DIR=$MODEL_DIR"/results"

#INPUT_FILE=./data/cgec/valid.src
INPUT_FILE=../data/cged_test.txt

echo $INPUT_FILE
export vocab_path=../plm/chinese-struct-bert-large/vocab.txt

CSC_OUTPUT=$RESULT_DIR"/csc.output"

if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --csc_predict True \
                    --input_file $INPUT_FILE \
                    --output_file $CSC_OUTPUT \

if [ ! -f $CSC_OUTPUT".char" ]; then
    python ../tools/segment/segment_bert.py < $CSC_OUTPUT > $CSC_OUTPUT".char"  # 分字
fi

if [ ! -f $INPUT_FILE".char" ]; then
    python ../tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi

if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi


OUTPUT_FILE=$RESULT_DIR"/seq2edit_cged_test.output"

echo "Generating..."
# 0.4; 0.4; 0805
ITERATION_COUNT=5
MIN_PROB=0.2
MIN_ERROR_PROB=0.2
MIN_REPLACE_PROB=0
MIN_APPEND_PROB=0
MIN_DELETE_PROB=0.45
ADD_CONFIDENCE=0.0
WEIGHTS=118

SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --cuda_device 0\
                  --input_file $CSC_OUTPUT".char"\
                  --additional_confidence $ADD_CONFIDENCE\
                  --min_probability $MIN_PROB\
                  --min_error_probability $MIN_ERROR_PROB\
                  --min_delete_prob $MIN_DELETE_PROB\
                  --min_replace_prob $MIN_REPLACE_PROB\
                  --min_append_prob $MIN_APPEND_PROB\
                  --iteration_count $ITERATION_COUNT\
                  --weights $WEIGHTS\
                  --output_file $OUTPUT_FILE --log

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
