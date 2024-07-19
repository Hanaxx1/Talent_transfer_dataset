# Experimental environment: 2 * A10
# 2 * 16GB GPU memory
# --dtype fp16 \
nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_cache_dir /home/css/models/Llama-2-7b-chat-hf \
    --model_type llama2-7b-chat \
    --check_model_is_latest false \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type _llama \
    --dtype fp16 \
    --output_dir output \
    --ddp_backend nccl \
    --custom_train_dataset_path /home/qiuyang/llama-recipes/src/llama_recipes/lla_datasets/alpaca_data_siyuanzu_process.json \
    --dataset_test_ratio 0 \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 3700 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_dtype AUTO \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 10 
    # --resume_from_checkpoint /home/qiuyang/workplace/swift/examples/pytorch/llm/output/llama2-7b-chat/v2-20240112-145628/checkpoint-11900

