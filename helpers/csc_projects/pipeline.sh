# Step 1. Data preprocessing
DATA_DIR=./exp_data/sighan
PRETRAIN_MODEL=../plm/bert-base-chinese
mkdir -p $DATA_DIR

export vocab_path=$PRETRAIN_MODEL"/vocab.txt"

# Step 2. Training
MODEL_DIR=./exps/sentence
CUDA_DEVICE=1
BATCH_SIZE=64
LEARNING_RATE=1e-3
NUM_EPOCH=30
MAX_SEQ_LEN=256

mkdir -p $MODEL_DIR/bak
cp ./pipeline.sh $MODEL_DIR/bak
cp train.py $MODEL_DIR/bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
    --pretrained_model $PRETRAIN_MODEL \
    --save_path $MODEL_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCH \
    --max_seq_len $MAX_SEQ_LEN \
    --lr $LEARNING_RATE \
    --tie_cls_weight True \
    --tag "sentence" \
    2>&1 | tee $MODEL_DIR"/_log.txt"


