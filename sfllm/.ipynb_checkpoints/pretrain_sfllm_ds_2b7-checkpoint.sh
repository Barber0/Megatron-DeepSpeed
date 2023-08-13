#!/bin/bash

# Runs the "1.3B" parameter model
script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
proj_home_dir=$script_dir/..
cd $proj_home_dir

export CUDA_DEVICE_MAX_CONNECTIONS=1

NAME=sfllm_2b7

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

SAVE_PATH=/root/autodl-tmp/sfllm-2b7-cl
LOAD_PATH=$SAVE_PATH

VOCAB_FILE=/root/gpt2_tokenizer/gpt2-vocab.json
MERGE_FILE=/root/gpt2_tokenizer/gpt2-merges.txt
DATA_PATH=/root/autodl-tmp/pile0204_text_document

BATCH_SIZE=16
GLOBAL_BATCH_SIZE=528
LOG_INTERVAL=500

SEQ_LEN=2048
HIDDEN_SIZE=2560
N_LAYERS=32
N_HEADS=20

BILLION=1000000000
MILLION=1000000

train_tokens_in_billion=26
train_tokens=$((${train_tokens_in_billion} * ${BILLION}))
train_samples=$(( 300 * ${BILLION} * 2 / ${SEQ_LEN} ))

lr_warmup_tokens_in_million=3000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * ${MILLION}))

lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * ${BILLION}))

CL_ENABLED="true"
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))

cl_tokens_in_billion=3
CL_TOKENS=$((${cl_tokens_in_billion} * ${BILLION}))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))

ZERO_STAGE=1

DATA_EFFICIENCY_SEED=8848

template_json="${script_dir}/ds_cfg_tpl.txt"
config_json="${script_dir}ds_cfg_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
        > ${config_json}

DS_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
"

MAX_LR=2e-4
MIN_LR=1e-5
WEIGHT_DECAY=1e-1

GPT_ARGS="
    --tensor-model-parallel-size ${NUM_GPUS} \
    --sequence-parallel \
    --num-layers ${N_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${N_HEADS} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --attention-dropout 0 \
    --hidden-dropout 0 \

    --use-rotary-position-embeddings \
    --micro-batch-size ${BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \

    --clip-grad 1.0 \
    --use-flash-attn \
    
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay ${WEIGHT_DECAY} \

    --lr ${MAX_LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-decay-style cosine \
    --lr-warmup-tokens ${lr_warmup_tokens} \

    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval ${LOG_INTERVAL} \
    --save-interval 50 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensorboard-dir /root/mega-logs-${NAME}
"

deepspeed \
    --master_port 3000 \
    pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $SAVE_PATH \
    --load $LOAD_PATH \
    --no-load-optim \
    --no-load-lr-state \
    $DS_ARGS