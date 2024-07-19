import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from swift.tuners import Swift
import json


#微调目录
model_dir1 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/mistral-7b-chat-v2/v7-20231228-133749/checkpoint-17477'
model_dir2 = '/home/qiuyang/workplace/swift/examples/pytorch/llm/output/mistral-7b-chat-v2/v9-20240104-151231/checkpoint-250'

model_type1 = ModelType.mistral_7b_chat_v2
model_type2 = ModelType.zephyr_7b_beta_chat
template_type1 = get_default_template_type(model_type1)
template_type2 = get_default_template_type(model_type2)

model_mistral, tokenizer_mistral = get_model_tokenizer(model_type1, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/Mistral-7B-Instruct-v0.2")

model_zephyr, tokenizer_zephyr = get_model_tokenizer(model_type2, model_kwargs={'device_map': 'auto'}, model_dir="/home/css/models/zephyr-7b-beta")

model_mistral.generation_config.max_new_tokens = 4096
model_zephyr.generation_config.max_new_tokens = 4096


#mistral
# model_mistral = Swift.from_pretrained(model_mistral, model_dir1, inference_mode=True)
model_mistral = Swift.from_pretrained(model_mistral, model_dir2, inference_mode=True)

#zephyr
# model_zephyr = Swift.from_pretrained(model_zephyr, model_dir1, inference_mode=True)
model_zephyr = Swift.from_pretrained(model_zephyr, model_dir2, inference_mode=True)

template_mistral = get_template(template_type1, tokenizer_mistral)
template_zephyr = get_template(template_type2, tokenizer_zephyr)


max_lines = 100
current_line = 0

content_list = []

# 打开JSON文件
with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_tuning/data/c4_data.json', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 解析JSON数据
        data = json.loads(line)
        
        if len(tokenizer_mistral.tokenize(data['content'])) <3500:
        # 在这里可以对每一行的数据进行处理
            content_list.append(data['content'])

        # 更新行数计数器
        current_line += 1
        
        # 检查是否已经达到最大行数
        if current_line >= max_lines:
            break

print(len(content_list))
print('---------')
xt = 0

for i in range(0,len(content_list)):
    print(i)
    content = content_list[i]
    eval_prompt = f"### Instruction: According to the following information：<{content}> Answer the question。Whether anyone in the information was transferred between countries in text?if so,from which country to which country?What year and month did the transfer take place?only give the starting point and destination example [Yes or No,starting point,destination,year-month],just country name.\n### Response:\n"
    response_mistral, history_mistral = inference(model_mistral, template_mistral, eval_prompt)
    response_zephyr, history_zephyr = inference(model_zephyr, template_zephyr, eval_prompt)
    if response_mistral == response_zephyr:
        xt = xt + 1

print(xt)