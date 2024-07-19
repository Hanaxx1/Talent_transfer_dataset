#欢迎帅哥
#开发时间: 2024-02-01 13:08

def get_characters_before_third_comma(input_string):
    count_comma = 0
    index_third_comma = None

    for i, char in enumerate(input_string):
        if char == ',':
            count_comma += 1
            if count_comma == 3:
                index_third_comma = i
                break

    if index_third_comma is not None:
        return input_string[:index_third_comma]
    else:
        return "Third comma not found"

import json
import matplotlib.pyplot as plt
import linecache

# 定义三个列表存储数据
llama_list = []
mistral_list = []
openchat_list = []


with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_data/data_siyuanzu/mistral_list6.json', 'r') as file:
    data_mistral = json.load(file)
with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_data/data_siyuanzu/openchat_list6.json', 'r') as file:
    data_openchat = json.load(file)
with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_data/data_siyuanzu/solar_list6.json', 'r') as file:
    data_solar = json.load(file)


path_file = '/home/qiuyang/workplace/swift/examples/pytorch/llm/my_data/data/c4_data_length_3500/c4_data6_3500.json'



# 打印列表长度以验证数据加载情况
print("mistral_list length:", len(data_mistral))
print("openchat_list length:", len(data_openchat))
print("solar_list length:",len(data_solar))

count = 0

with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_data/data/c4_data_duiqi/c4_data.json', 'a') as output_file:
    for i in range(0,len(data_mistral)):
        # print(i)
        if data_mistral[i] == data_openchat[i] and data_mistral[i]!="[No,UNKNOWN,UNKNOWN,UNKNOWN]" and data_mistral[i] == data_solar[i]:
            count = count + 1
            # 指定要读取的行数
            line_number = i+1
            # 使用 linecache 模块读取指定行数的内容
            line = linecache.getline(path_file, line_number)
            # 解析JSON数据
            data = json.loads(line.strip())  # strip() 方法用于删除行尾的换行符
            origin = str(data_mistral[i]).split(',')[1]
            destination = str(data_mistral[i]).split(',')[2]
            timestamp = str(data_mistral[i]).split(',')[3].split(']')[0]
            # print(data_mistral[i],origin,destination,timestamp)
            data['origin'] = origin
            data['destination'] = destination
            data['timestamp'] = timestamp

            #写入
            json.dump(data, output_file)
            output_file.write('\n')

print("result_one consistent results:",count)
print('-------------------')
