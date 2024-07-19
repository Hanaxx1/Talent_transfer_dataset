#欢迎帅哥
#开发时间: 2023-08-02 9:25

import spacy
from geonamescache import GeonamesCache
import requests
import stanfordnlp
import torch
import json
import re
import numpy as np
import pandas as pd
import geonamescache
import pycountry
import csv
import numpy as np
import geo.find_zhuyu as un
import csv
import pandas as pd
from newsapi import NewsApiClient
import pymysql


import nltk
from nltk import pos_tag, word_tokenize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gc = GeonamesCache()
country_dic = ['England']

a = np.load('a.npy')
a = a.tolist()

from geopy.geocoders import Nominatim


def get_country_from_city(city_name):
    geolocator = Nominatim(user_agent="city_country_lookup")
    location = geolocator.geocode(city_name, exactly_one=True)

    if location:
        return location.address.split(",")[-1].strip()
    else:
        return None

def extract_subject(sentence):
    # 加载英文模型
    nlp = spacy.load("en_core_web_lg")

    # 对句子进行处理
    doc = nlp(sentence)

    # 分析句子中的从句和短语，找到每个从句或短语的主语
    subjects = []
    for sent in doc.sents:
        for token in sent:
            if "nsubj" in token.dep_:
                subjects.append(token.text)
                break

    if subjects:
        return subjects
    else:
        return "未找到主语。"


def has_list_b_words_after_list_a(sentence, listA, listB):
    for wordA in listA:
        if wordA in sentence:
            for wordB in listB:
                if wordB in sentence:
                    if sentence.find(wordB) > sentence.find(wordA):
                        return True
    return False

#获取介词
def get_prepositions(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    prepositions = []
    for token in doc:
        if token.pos_ == "ADP":
            preposition = token.text
            dependent_noun = [child.text for child in token.children if child.dep_ == "pobj"]
            if dependent_noun:
                prepositions.append((preposition, dependent_noun[0]))

    return prepositions


#是否有主代词
def has_subject_pronoun(sentence):
    # 分词和词性标注
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    # 查找主语位置并检查是否是主代词
    for word, pos in tagged_words:
        if pos.startswith('N') or pos.startswith('PRP'):
            if word.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they']:
                return True
            else:
                return False
    return False

def is_city(city_name):
    cities = gc.get_cities_by_name(city_name)
    return len(cities) > 0

def is_country(country_name):
    gc = geonamescache.GeonamesCache()
    countries = gc.get_countries()
    country_names = [country['name'] for country in countries.values()]
    return country_name in country_names

def prepositions_is_country(sentence):
    result = get_prepositions(sentence)
    print(result)
    test_re = []
    for tup in result:
        if tup[0] in ['to','of','for','in']:
            test_re.append(tup)
    if len(test_re) == 0:
        return True
    else:
        for tup in test_re:
            geo = tup[1]
            if (is_city(geo) or is_country(geo) or (geo in country_dic) or (geo in a)):
                return True
    return False

def insert_to_sql(titles,dates,urls,contents,abstracts):

    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="cg950523",
        database="geo"
    )
    # 创建游标对象
    cause = ''

    cursor = conn.cursor()
    for i in range(len(titles)):
        try:
            # origin, destination = un.origin_to_destination(abstracts[i])
            origin = 'uu'
            destination = 'uu'
            cause = 'uu'
            # 准备插入数据的SQL语句
            values =(
                titles[i],
                abstracts[i],
                cause,
                dates[i],
                urls[i],
                contents[i],
                origin,
                destination
            )
            sql = "INSERT INTO tran_data (title, abstract, cause, date, url, content, origin, destination) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

            # 插入数据
            cursor.execute(sql, values)
            # 提交更改
            conn.commit()

        except pymysql.Error as err:
            # 处理MySQL数据库异常
            print("MySQL错误:", err)

    cursor.close()
    conn.close()

def insert_to_sql_one(title,date,url,content,abstract):

    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="cg950523",
        database="geo"
    )
    # 创建游标对象
    cursor = conn.cursor()

    try:
        origin, destination = un.origin_to_destination(abstract)
        cause = 'unknow'
        # 准备插入数据的SQL语句
        values =(
            title,
            abstract,
            cause,
            date,
            url,
            content,
            origin,
            destination
        )
        sql = "INSERT INTO tran_data (title, abstract, cause, date, url, content, origin, destination) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

        # 插入数据
        cursor.execute(sql, values)
        # 提交更改
        conn.commit()

    except pymysql.Error as err:
        # 处理MySQL数据库异常
        print("MySQL错误:", err)

    cursor.close()
    conn.close()



def get_country_name_from_abbreviation(abbreviation):
    try:
        country = pycountry.countries.get(alpha_2=abbreviation)
        if country:
            return country.name
        else:
            return "Unknown abbreviation"
    except AttributeError:
        return "Invalid abbreviation"

def get_country_from_city(city_name):
    base_url = "http://api.geonames.org/searchJSON"
    params = {
        "q": city_name,
        "maxRows": 1,  # 限制结果数量
        "username": "hanaxx"  # 使用你的GeoNames用户名
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "geonames" in data and len(data["geonames"]) > 0:
        print(data)
        country_name =  data["geonames"][0].get("countryName")
        fcl = data["geonames"][0]["fcl"]
        return country_name
    return None







if __name__ == '__main__':
    # object_dic = ['It', 'it', 'he', 'He', 'she', 'She']
    # sentence =
    # print(get_country_name_from_abbreviation('USA'))
    # print(get_country_name_from_abbreviation('UK'))

    # 使用你的GeoNames用户名
    # city_name = "Liverpool"
    # country_name = get_country_from_city(city_name)
    # print(country_name)
    # list1 = [1,1,1]
    # set1 = set(list1)
    # print(len(set1))
    # if country_name:
    #     print(f"The city {city_name} is in {country_name}.")
    # else:
    #     print("Country not found for the given city.")

    # titles = []
    # dates = []
    # contents = []
    # csv_file_path = r'E:\kaggle\input\archive_nus_summary\news_summary.csv'
    # # 打开CSV文件并读取数据
    # result_data = pd.read_csv(csv_file_path,encoding='ISO-8859-1')
    #
    # # 遍历前10行的 'description' 列
    # sum = 0
    # for index, row in result_data.iterrows():
    #     title = row['text']
    #     title = str(title)
    #     print(index)
    #     if un.is_tran_two(title):
    #         sum = sum + 1
    #         date = row['date']
    #         content = row['text']
    #         origin, destination = un.origin_to_destination(title)
    #         print(title)
    #         print(origin,destination)
    #         print('----------')
    #         abstract = 'unknow'
    #         cause = 'unknow'
    # print(sum)
    #测试用
    # titles = ['a']
    # dates = ['2017-01-01']
    # contents = ['aa']
    # abstracts = ['aaa']
    # url = ['aa']
    # insert_to_sql(titles,dates,url,contents,abstracts)
    # 定义API请求的URL
    # text = '''WASHINGTON (Reuters) - President Donald Trump will travel to Texas to visit with the victims of the Santa Fe High School shooting on Thursday, White House spokeswoman Sarah Sanders said on Wednesday.'''
    # origin,destination = un.origin_to_destination(text)
    # print(origin,destination)
    # pass

    offset = 644500  #125700
    url_cc = f"https://datasets-server.huggingface.co/rows?dataset=cc_news&config=plain_text&split=train&offset={offset}&limit=100"
    # titles = []
    # dates = []
    # abstracts = []
    # contents = []
    # urls = []
    count = 0
    # while offset <= 708200:
    #先到100000

    while offset <= 708200:
        print(url_cc)
        # 写入内容
        with open(r'D:\scrapy\pubmed\geo\record.txt', 'w') as file:
            file.write(str(offset))
        # 发送GET请求
        response = requests.get(url_cc)
        offset = offset+100
        url_cc = f"https://datasets-server.huggingface.co/rows?dataset=cc_news&config=plain_text&split=train&offset={offset}&limit=100"
        # 检查请求是否成功
        if response.status_code == 200:
            # 获取响应的JSON数据

            data = response.json()
            # 在这里处理数据，根据数据格式进行解析
            article = data['rows']
            for i in range(len(article)):
                print(i)
                article_data = article[i]['row']
                title = article_data['title']
                content = article_data['text']
                date = article_data['date'].split(' ')[0]
                url = article_data['url']
                abstract = article_data['description']
                abstract = str(abstract)
                # print(type(article_data))
                # print(article_data)
                if un.is_tran_two(abstract):
                    print('~~')
                    insert_to_sql_one(title,date,url,content,abstract)
                    count = count + 1
        else:
            print("请求失败，状态码:", response.status_code)

    print(count)

    # np.save('a.npy',a)
    # a = np.load('a.npy')
    # a = a.tolist()
    # print('UK' in a)



