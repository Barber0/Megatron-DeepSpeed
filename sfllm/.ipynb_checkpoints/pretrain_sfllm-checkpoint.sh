#!/bin/bash

# Runs the "1.3B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/root/autodl-tmp/sfllm-1b4-alter
VOCAB_FILE=/root/gpt2_tokenizer/gpt2-vocab.json
MERGE_FILE=/root/gpt2_tokenizer/gpt2-merges.txt
DATA_PATH=/root/autodl-tmp/pile0204_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

seq_len=2048

train_tokens_in_billion=13
train_tokens=$((${train_tokens_in_billion} * 1000000000))
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

lr_warmup_tokens_in_million=3000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))

lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --use-rotary-position-embeddings \
    --micro-batch-size 22 \
    --global-batch-size 528 \
    --lr 2e-4 \
    --min-lr 1e-5 \
    --train-tokens ${train_tokens} \
    --train-samples ${train_samples} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --use-flash-attn \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
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
    --log-interval 200 \
    --save-interval 500 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --tensorboard-dir /root/mega-logs-1b3
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

