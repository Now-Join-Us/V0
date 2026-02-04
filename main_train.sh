#!/bin/bash
set -e
set -x

ulimit -n 65535
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

NNODES=1
NUM_GPUS_PER_NODE=2

NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=23326

# ================= 路径配置 =================
QWEN_EMBEDDING_MODEL_PATH="your_base_model_dir/Qwen3-Embedding-0.6B"
TABPFN_MODEL_PATH="your_base_model_dir/tabpfn_2_5/tabpfn-v2.5-classifier-v2.5_default.ckpt"

DATA_ROOT="./data"

PROMPT_DICT="${DATA_ROOT}/prompt_dict.json"
CONTEXT_DATA="${DATA_ROOT}/context.jsonl"
TRAIN_DATA="${DATA_ROOT}/train.jsonl"
EVAL_DATA="${DATA_ROOT}/test.jsonl"
VALIDITY_DATA="${DATA_ROOT}/validity.jsonl"

TIME_STR="run_temp"
BASE_DIR="./outputs/exp_temp"
LOG_DIR="${BASE_DIR}/logs/${TIME_STR}"
TASK_CHECKPOINT_DIR="${BASE_DIR}/checkpoints/${TIME_STR}"

WANDB_ID="love1314"

if [ "$NODE_RANK" -eq 0 ]; then
    mkdir -p "${LOG_DIR}"
    mkdir -p "${TASK_CHECKPOINT_DIR}"
fi

LOG_FILE="${LOG_DIR}/logs.jsonl"
METRIC_FILE="${LOG_DIR}/metrics.jsonl"
STDOUT_LOG_FILE="${LOG_DIR}/stdout_node_${NODE_RANK}.log"
WANDB_CACHE_DIR="${LOG_DIR}"
export WANDB_DIR="${WANDB_CACHE_DIR}"
export WANDB_MODE="offline"


# ================= 启动训练 =================
META_BATCH_SIZE=1 
GRAD_ACCUM_STEPS=2


torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    main_train.py \
    --qwen_path "${QWEN_EMBEDDING_MODEL_PATH}" \
    --tabpfn_checkpoint "${TABPFN_MODEL_PATH}" \
    --context_data_paths "${CONTEXT_DATA}" \
    --train_data_paths "${TRAIN_DATA}" \
    --eval_data_paths "${EVAL_DATA}" \
    --validity_data_paths "${VALIDITY_DATA}" \
    --prompt_dict_path "${PROMPT_DICT}" \
    --run_mode "train" \
    --pooling_strategy "dynamic_query" \
    --dynamic_query_generator_bottleneck_dim 128 \
    --dynamic_query_generator_dropout_rate 0.2 \
    --reduce_method "none" \
    --num_queries 168 \
    --embed_dim 6 \
    --num_heads 3 \
    --tabpfn_estimators 4 \
    --epochs 50 \
    --meta_batch_size ${META_BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --train_query_batch_size 8 \
    --eval_query_batch_size 4 \
    --support_size 256 \
    --lr_backbone 1e-5 \
    --lr_adapter 0.0002 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --train_embed_bs 4 \
    --eval_embed_bs 4 \
    --max_grad_norm 1.0 \
    --time_str "${TIME_STR}" \
    --log_path "${LOG_FILE}" \
    --metric_path "${METRIC_FILE}" \
    --wandb_project "generalist-v" \
    --wandb_id "${WANDB_ID}" \
    --checkpoint_dir "${TASK_CHECKPOINT_DIR}" \
    --save_interval 1 \
    --max_keep_checkpoints 50 \
    --resume \
    --label_strategy minmax_norm \
    --loss_type combined \
    --loss_alpha 0.25 \
    2>&1 | tee "${STDOUT_LOG_FILE}"

echo ">>> Node ${NODE_RANK} Finished."
