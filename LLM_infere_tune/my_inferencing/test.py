import json
import linecache

with open('/home/qiuyang/workplace/swift/examples/pytorch/llm/my_data/data/c4_data_duiqi/c4_data.json', 'r') as input_file:

            for line in input_file:
                # 解析JSON数据
                data = json.loads(line)
                print(data)


