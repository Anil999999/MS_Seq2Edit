# Step1. Data Preprocessing

## Download Structbert
if [ ! -f ./plm/chinese-struct-bert-large/pytorch_model.bin ]; then
    wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model
    mv ch_model ./plm/chinese-struct-bert-large/pytorch_model.bin
fi

export vocab_path=./plm/chinese-struct-bert-large/vocab.txt

## Tokenize
SRC_FILE=./data/cgec/source_20220818.txt  # 每行一个病句
TGT_FILE=./data/cgec/target_20220818.txt # 每行一个正确句子，和病句一一对应

if [ ! -f $SRC_FILE".char" ]; then
    python ./tools/segment/segment_bert.py < $SRC_FILE > $SRC_FILE".char"  # 分字
fi
if [ ! -f $TGT_FILE".char" ]; then
    python ./tools/segment/segment_bert.py < $TGT_FILE > $TGT_FILE".char"  # 分字
fi

## Generate label file
LABEL_FILE=./data/cgec/train.label  # 训练数据
if [ ! -f $LABEL_FILE ]; then
    python ./utils/preprocess_data.py -s $SRC_FILE".char" -t $TGT_FILE".char" -o $LABEL_FILE --worker_num 32
    shuf $LABEL_FILE > $LABEL_FILE".shuf"
fi

# lang8 tokenizer
LANG_SRC_FILE=./data/lang8/clean_data/source_20220817.txt
LANG_TGT_FILE=./data/lang8/clean_data/target_20220817.txt

if [ ! -f $LANG_SRC_FILE".char" ]; then
    python ./tools/segment/segment_bert.py < $LANG_SRC_FILE > $LANG_SRC_FILE".char"
fi

if [ ! -f $LANG_TGT_FILE".char" ]; then
    python ./tools/segment/segment_bert.py < $LANG_TGT_FILE > $LANG_TGT_FILE".char"
fi

LANG_LABEL_FILE=./data/lang8/clean_data/train_more.label  # 训练数据
if [ ! -f $LANG_LABEL_FILE ]; then
    python ./utils/preprocess_data.py -s $LANG_SRC_FILE".char" -t $LANG_TGT_FILE".char" -o $LANG_LABEL_FILE --worker_num 32
    shuf $LANG_LABEL_FILE > $LANG_LABEL_FILE".shuf"
fi

echo 'dev data process'
DEV_SET=./data/cgec/valid.label
DEV_SRC=./data/cgec/valid.src
DEV_TRG=./data/cgec/valid.trg

if [ ! -f $DEV_SRC".char" ]; then
    python ./tools/segment/segment_bert.py < $DEV_SRC > $DEV_SRC".char"  # 分字
fi
if [ ! -f $DEV_TRG".char" ]; then
    python ./tools/segment/segment_bert.py < $DEV_TRG > $DEV_TRG".char"  # 分字
fi

if [ ! -f $DEV_SET ]; then
    python ./utils/preprocess_data.py -s $DEV_SRC".char" -t $DEV_TRG".char" -o $DEV_SET --worker_num 32

fi

## Step2. Training
CUDA_DEVICE=1
SEED=42
#

MODEL_DIR=./exps/seq2edit/model
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

PRETRAIN_WEIGHTS_DIR=./plm/chinese-struct-bert-large
#
mkdir ${MODEL_DIR}/src_bak
cp ./pipeline.sh $MODEL_DIR
cp -r ./gector ${MODEL_DIR}/src_bak

VOCAB_PATH=./data/output_vocab_more

# Freeze encoder (Cold Step)
COLD_LR=1e-3
COLD_BATCH_SIZE=128
COLD_MODEL_NAME=Best_Model_Stage_1
COLD_EPOCH=5

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 0\
                --train_set $LANG_LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $COLD_MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $COLD_BATCH_SIZE\
                --n_epoch $COLD_EPOCH\
                --lr $COLD_LR\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --seed $SEED

# Unfreeze encoder
LR=1e-5
BATCH_SIZE=32
ACCUMULATION_SIZE=4
MODEL_NAME=Best_Model_Stage_2
EPOCH=10
PATIENCE=10
CUDA_DEVICE=1
SKIP_CORRECT=0
SKIP_COMPLEX=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 1\
                --train_set $LANG_LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $BATCH_SIZE\
                --n_epoch $EPOCH\
                --lr $LR\
                --accumulation_size $ACCUMULATION_SIZE\
                --patience $PATIENCE\
                --skip_correct $SKIP_CORRECT\
                --skip_complex $SKIP_COMPLEX\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --pretrain_folder $MODEL_DIR\
                --pretrain "Temp_Model"\
                --seed $SEED

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 1\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $BATCH_SIZE\
                --n_epoch $EPOCH\
                --lr $LR\
                --accumulation_size $ACCUMULATION_SIZE\
                --patience $PATIENCE\
                --skip_correct $SKIP_CORRECT\
                --skip_complex $SKIP_COMPLEX\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --pretrain_folder $MODEL_DIR\
                --pretrain "Temp_Model"\
                --seed $SEED


# Step3. Inference
MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
RESULT_DIR=$MODEL_DIR"/results"

INPUT_FILE=../data/valid.src # 输入文件
echo $INPUT_FILE

if [ ! -f $INPUT_FILE".char" ]; then
    python ../tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi

OUTPUT_FILE=$RESULT_DIR"/MuCGEC_test.output"

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --cuda_device $CUDA_DEVICE\
                  --input_file $INPUT_FILE".char"\
                  --output_file $OUTPUT_FILE --log

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
