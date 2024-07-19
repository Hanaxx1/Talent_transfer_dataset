#欢迎帅哥
#开发时间: 2023-08-02 15:18

import json
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns

def add_data_to_json(file_path, title, text, label):
    # 读取现有的 JSON 数据
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    # 添加新数据
    new_entry = {
        'title': title,
        'text': text,
        'label': label
    }
    data.append(new_entry)

    # 将更新后的数据写回 JSON 文件
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    file_path = 'scrapy_json/test.json'
    title_list = []
    text_list = []
    label_list = []

    title_list.append('''Acclaimed Astronomer Explores Celestial Wonders, Observes Skies in Chile''')
    title_list.append('Master Sculptor Carves New Horizons, Creates Artistry in Italy')
    title_list.append('''Financial Pioneer Redefines Markets, Trades in London's Financial Hub''')
    title_list.append('Cross-Cultural Linguist Unveils Multilingual Stories, Narrates Tales in Morocco')
    title_list.append('''Acclaimed Environmentalist Joins Ocean Conservation Efforts, Dives into Australia's Waters''')
    label_list.append('true')
    label_list.append('true')
    label_list.append('true')
    label_list.append('true')
    label_list.append('true')


    print(len(title_list))
    for i in range(len(title_list)):
        add_data_to_json(file_path,title_list[i],text_list[i],label_list[i])
