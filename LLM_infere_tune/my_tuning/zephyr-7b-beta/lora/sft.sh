#!/bin/bash

nproc_per_node=1

gradient_accumulation_steps=$(expr 16 / $nproc_per_node)


    # --model_type zephyr_7b_beta \
    # --check_model_is_latest false \

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_cache_dir /home/css/models/zephyr-7b-beta \
    --model_id_or_path mistralai/Mistral-7B-v0.1 \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type llama \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --custom_train_dataset_path /home/qiuyang/llama-recipes/src/llama_recipes/lla_datasets/alpaca_data.json \
    --dataset_test_ratio 0 \
    --train_dataset_sample -1 \
    --val_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --lora_dtype AUTO \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --save_total_limit 2 \
    --logging_steps 10

