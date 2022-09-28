PRETRAIN_MODEL=../plm/bert-base-chinese
DATA_DIR=./exp_data/sighan

TEST_SRC_FILE=../data/dev/cged_test.txt
TAG=sighan15
export vocab_path=$PRETRAIN_MODEL"/vocab.txt"

python ./data_preprocess.py \
--source_dir $TEST_SRC_FILE \
--bert_path $PRETRAIN_MODEL \
--save_path $DATA_DIR"/test_"$TAG".pkl" \
--data_mode "lbl" \
--normalize "True"

MODEL_PATH=exps/sighan-epoch-3.pt
SAVE_PATH=exps/sighan/decode

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=1 python decode.py \
    --pretrained_model $PRETRAIN_MODEL \
    --test_path $DATA_DIR"/test_"$TAG".pkl" \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH"/"$TAG".lbl" ;

