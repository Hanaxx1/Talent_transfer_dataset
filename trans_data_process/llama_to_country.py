import find_zhuyu as nn
import json
import re

def llama_to_alapaca(json_path):

    with open(json_path) as file:
        data = json.load(file)
    
    for index,item in enumerate(data):
        date = str(item.get('date'))
        content = str(item.get('content'))
        output = str(item.get('output'))

        if output.startswith("No"):
            continue
        else:
            origin = ''
            destination = ''
            # 使用逗号和句号一起来切分字符串
            split_result = re.split(r'[,.]', output)
            print(split_result)
            for sentence in split_result:
                if nn.has_from_to(sentence):
                    print(sentence)
                else:
                    print('no')
                    




if __name__ == '__main__':

    json_path = '/home/qiuyang/workplace/geo_to_ssh/geo/result_data/llama_result.json'
    llama_to_alapaca(json_path=json_path)