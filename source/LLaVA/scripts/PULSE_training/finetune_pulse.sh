#!/bin/bash

# wandb configurations
export WANDB__REQUIRE_LEGACY_SERVICE=TRUE
wandb login --relogin "wandb key"
export WANDB_ENTITY="wandb entity"
export WANDB_NAME="run name"
export WANDB_PROJECT="project name"

# distributed training configurations
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="1234"
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# huggingface configurations
# export TRANSFORMERS_CACHE=
# export HF_HOME=

model_path=${PULSE_MODEL_PATH:-PULSE-ECG/PULSE-7B}
version=llava_v1

data_path=${PTBXL_STAGE2_JSON:-/path/to/data/pulse_ptbxl_stage2/train.json}
image_folder=${PTBXL_STAGE2_IMAGE_FOLDER:-/path/to/data/pulse_ptbxl_stage2/images}
label_vocab_path=${PTBXL_STAGE2_LABEL_VOCAB:-/path/to/data/pulse_ptbxl_stage2/scp_71_vocab.json}
output_dir=${PULSE_STAGE2_OUTPUT_DIR:-/path/to/saved/pulse-ptbxl-stage2}

num_epochs=3
BATCH_PER_GPU=2
GLOBAL_BATCH_SIZE=128

TOTAL_BATCH_SIZE=$(($WORLD_SIZE * $BATCH_PER_GPU))
GRAD_ACC_STEP=$(($GLOBAL_BATCH_SIZE / $TOTAL_BATCH_SIZE))

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES \
    /path/to/PULSE/LLaVA/llava/train/train_mem.py \
    --deepspeed ../zero2.json \
    --model_name_or_path $model_path \
    --version $version \
    --data_path $data_path \
    --image_folder $image_folder \
    --label_vocab_path $label_vocab_path \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length False \
    --bits 4 \
    --quant_type nf4 \
    --double_quant True \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --freeze_mm_mlp_adapter True \
    --aslgen_enable True \
    --aslgen_lambda_0 0.3 \
    --aslgen_gamma_pos 1.0 \
    --aslgen_gamma_neg 4.0 \
    --aslgen_prob_shift_m 0.05 \
    --aslgen_positive_token " 1" \
    --aslgen_negative_token " 0" \
    --aslgen_probability_mode thesis \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_epochs \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb