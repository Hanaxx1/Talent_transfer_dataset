# Experimental environment: A10
# If you want to merge LoRA weight and save it, you need to set `--merge_lora_and_save true`.
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=1,2 \
python llm_infer.py \
    --ckpt_dir output/openchat_3.5/without_info/20231218-07:43:24-split_type=8:2-train_ratio=1.0/checkpoint-609 \
    --load_args_from_ckpt_dir true \
    --eval_human false \
    --max_length 4096 \
    --use_flash_attn false \
    --max_new_tokens 4096 \
    --temperature 0.1 \
    --top_p 0.7 \
    --repetition_penalty 1.05 \
    --do_sample true \
    --merge_lora_and_save false
