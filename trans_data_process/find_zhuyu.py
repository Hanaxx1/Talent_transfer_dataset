#欢迎帅哥
#开发时间: 2023-08-01 9:44
# -- coding: utf-8 --**

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
import nltk
from nltk import pos_tag, word_tokenize
import stanfordnlp
import random
from nltk.parse import DependencyGraph

user_name = ['Hanaxx4','Hanaxx5','Hanaxx6','miemie']

nlp = spacy.load("en_core_web_lg")
# nltk.download('punkt')
gc = GeonamesCache()

#国家二字，三字缩写
CountryCode = np.load('/home/qiuyang/workplace/geo_to_ssh/geo/a.npy').tolist()

trans_dic = ['visit','arrive','depart','travel','transfer','shift','relocate','migrate','transport',
             'walk','immigrate','emigrate','exit','transit','repatriate','journey','head',
             'return','drive','board','march','teleport','embark']
            #'flee','move','leave','carry','bring','fly','run','go','come','send','enter','cross','sail','launch','smuggle',
object_dic = ['It','it','he','He','she','She']
country_dic = ['England']


#请求geonames的API网址
base_url = "http://api.geonames.org/searchJSON"



#向A加入B中A没有的
def add_missing_elements(A, B):
    for element in B:
        if element not in A:
            A.append(element)
    return A

def find_person_entities(doc):
    person_entities = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_entities.append(ent)
    return person_entities

def complete_person_names(doc, person_entities):
    for ent in person_entities:
        tokens = [token for token in doc if token.text in ent.text]
        for token in tokens:
            modifier = [t for t in token.children if t.dep_ == "mod"]
            for mod in modifier:
                ent.text = f"{mod.text} {ent.text}"


#识别人名
def get_complete_person_names(sentence):
    doc = nlp(sentence)
    person_entities = find_person_entities(doc)
    complete_person_names(doc, person_entities)
    return [ent.text for ent in person_entities]

#不用改
def find_countries(doc):
    countries = []
    for ent in doc.ents:
        if ent.label_ == "GPE":  # 只筛选 GPE 标签，即地理政治实体
            countries.append(ent)
    return countries


#识别国家名,城市转国家
def get_countries_process(sentence):
    doc = nlp(sentence)
    countries = find_countries(doc)
    countries_list = [ent.text for ent in countries]
    result = []
    for token in countries_list:
        # print('输入的,',token)
        if is_city(token):
            # print('输出的,', city_to_country(token))
            result.append(city_to_country(token))
        else:
            # print('输出的,', token)
            result.append(token)
    # return [ent.text for ent in countries]

    return result

#识别地名
def get_countries(sentence):
    doc = nlp(sentence)
    countries = find_countries(doc)
    return [ent.text for ent in countries]

def is_city(city_name):
    # cities = gc.get_cities_by_name(city_name)
    # return len(cities) > 0

    max_retries = 3  # 设置最大重试次数
    retry_count = 0
    while retry_count < max_retries:
        try:
            username = str(random.choice(user_name))
            params = {
                "q": city_name,
                "maxRows": 1,  # 限制结果数量
                "username": username  # 使用你的GeoNames用户名
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            print(data)
            if "geonames" in data and len(data["geonames"]) > 0:
                # fclName = result_data["geonames"][0].get("fclName")
                # fclName = str(fclName).split(',')[0]
                # return fclName == 'city'
                fcl = data["geonames"][0].get("fcl")
                return fcl == 'P'
            return False
        except:
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying (attempt {retry_count})...")
            else:
                print("Max retries reached. Exiting.")
                break




#加了个trycatch
def is_country(city_name):
    # gc = geonamescache.GeonamesCache()pp
    # countries = gc.get_countries()
    # country_names = [country['name'] for country in countries.values()]
    # return country_name in country_names

    max_retries = 3  # 设置最大重试次数
    retry_count = 0
    while retry_count < max_retries:
        try:
            username = str(random.choice(user_name))
            params = {
                "q": city_name,
                "maxRows": 1,  # 限制结果数量
                "username": username  # 使用你的GeoNames用户名
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            print(data)
            if "geonames" in data and len(data["geonames"]) > 0:
                fcl = data["geonames"][0].get("fcl")
                return fcl == 'A'
            return False
        except:
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying (attempt {retry_count})...")
            else:
                print("Max retries reached. Exiting.")
                break
    return False


def city_to_country(city_name):
    # cities = gc.get_cities()
    # for city_id, city_info in cities.items():
    #     if city_info['name'] == city_name:
    #         country_code = city_info['countrycode']
    #         country_info = gc.get_countries()[country_code]
    #         country_name = country_info['name']
    #         return country_name
    # return None
    max_retries = 3  # 设置最大重试次数
    retry_count = 0
    while retry_count < max_retries:
        try:
            username = str(random.choice(user_name))
            params = {
                "q": city_name,
                "maxRows": 1,  # 限制结果数量
                "username": username  # 使用你的GeoNames用户名
            }
            response = requests.get(base_url, params=params)
            data = response.json()
            if "geonames" in data and len(data["geonames"]) > 0:
                country_name =  data["geonames"][0].get("countryName")
                if country_name is not None:
                    return country_name
            return city_name
        except:
            # 请求失败，记录错误信息
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying (attempt {retry_count})...")
            else:
                print("Max retries reached. Exiting.")
                break



# def find_countries(doc):
#     countries = []
#     for ent in doc.ents:
#         if ent.label_ == "GPE":
#             city_name = ent.text
#             country_name = get_country_by_city(city_name)
#             if country_name:
#                 countries.append(country_name)
#     return countries
#
# def get_country_by_city(city_name):
#     # 使用GeoNames API查询城市信息
#     url = f"http://api.geonames.org/searchJSON?q={city_name}&maxRows=1&username=demo"
#     response = requests.get(url)
#     result_data = response.json()
#     if "geonames" in result_data and len(result_data["geonames"]) > 0:
#         country_name = result_data["geonames"][0]["countryName"]
#         return country_name
#     return None
#
# def get_countries(sentence):
#     doc = nlp(sentence)
#     countries = find_countries(doc)
#     return countries


def find_verbs(doc):
    verbs = []
    for token in doc:
        if token.pos_ == "VERB":
            verbs.append(token)
    return verbs

def lemmatize_verbs(sentence):
    doc = nlp(sentence)
    lemmatized_verbs = [token.lemma_ if token.pos_ == "VERB" else token.text for token in doc]
    return " ".join(lemmatized_verbs)

#获得动词
def get_verbs(sentence):
    doc = nlp(sentence)
    verbs = find_verbs(doc)
    return verbs

#获得动词原型
def get_lemmatized_verbs(sentence):
    doc = nlp(sentence)
    verbs = find_verbs(doc)
    lemmatized_verbs = [verb.lemma_ for verb in verbs]
    return lemmatized_verbs

#句子拆分
def sentence_segmentation(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences



#依存句法树
def get_subject(sentence):
    # 运行依存句法分析
    doc = nlp(sentence)
    # 遍历句子中的每个词
    for token in doc:
        # 如果词的依存关系是"nsubj"（名词主语）或"nsubjpass"（被动语态的名词主语）
        if token.dep_ in ["nsubj", "nsubjpass"]:
            # 返回找到的主语
            return token.text
    # 如果没有找到主语，则返回None
    return None

#判断A中是否有元素在B中
def has_common_element(A,B):
    for strA in A:
        for strB in B:
            if strA == strB:
                return True

    return False

#判断A中是否有元素在B中
def get_has_common_element(A,B):
    result = []
    for strA in A:
        for strB in B:
            if strA == strB:
                result.append(strA)
    return result

def str_has_common(A,B):
    for strB in B:
        if A in strB:
            return True
    return False

#列表B中有词在列表A词后
def has_list_b_words_after_list_a(sentence, listA, listB):
    for wordA in listA:
        if wordA in sentence:
            for wordB in listB:
                if wordB in sentence:
                    if sentence.find(wordB) > sentence.find(wordA):
                        return True
    return False

#是否有主代词
def has_subject_pronoun(sentence):
    # 分词和词性标注
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    # 查找主语位置并检查是否是主代词
    for word, pos in tagged_words:
        if pos.startswith('N') or pos.startswith('PRP'):
            if word.lower() in ['i', 'you', 'he', 'she', 'we', 'they']: #'it',
                return True
            else:
                return False
    return False


#介词 to,of,for,in到达的是否是地名
def prepositions_is_country(sentence):
    result = get_prepositions(sentence)
    # print(result)
    test_re = []
    for tup in result:
        if tup[0] in ['to','of','for','in']:
            test_re.append(tup)
    if len(test_re) == 0:
        return True
    else:
        for tup in test_re:
            geo = tup[1]
            if (is_city(geo) or is_country(geo) or (geo in country_dic) or (geo in CountryCode)):
                return True
    return False


#判断是否发生转移
def is_tran(text):
    # sentences = sentence_segmentation(text)
    sentences = text.split(',')
    pr_subject = []
    pr_name = []
    country_var = []

    for sentence in sentences:
        #获取主语
        print('-----------')
        print(sentence)
        subject = get_subject(sentence)
        name = get_complete_person_names(sentence)
        ver = get_lemmatized_verbs(sentence)

        #统计全文提到的国家数
        geo = get_countries_process(sentence)
        country_var = add_missing_elements(country_var, geo)

        print(subject)
        print(name)
        print(ver)

        #动词中出现了转移名词
        if has_common_element(ver,trans_dic):
            #主语是宾语,继承上句
            if subject in object_dic:
                subject = pr_subject
                name = pr_name


            if str_has_common(subject,name):
                geo = get_countries(sentence)
                for token in geo:
                    if not is_city(token):
                        return True
                if len(country_var)>1:
                    return True
        else:
            if subject in object_dic:
                print(pr_subject)
                print(pr_name)
                continue

            else:
                pr_subject = subject
                pr_name = name

        print(pr_subject)
        print(pr_name)
        # #没主语就循环进下一句
        # if len(subject) == 0:
        #     continue
        # else:
        #     #如果主语是包含宾语的,则继承上一句的主语
        #     if has_common_element(subject,object_dic):
        #         pass
        #     #正常主语
        #     else:
        #         pass
    #循环中没有判断出转移
    return False

#获取介词
def get_prepositions(sentence):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(sentence)

    prepositions = []
    for token in doc:
        if token.pos_ == "ADP":
            preposition = token.text
            dependent_noun = [child.text for child in token.children if child.dep_ == "pobj"]
            if dependent_noun:
                prepositions.append((preposition, dependent_noun[0]))

    return prepositions

def has_from_to(sentence):
    geo = get_countries(sentence)
    pattern = r'from\s+([^,]+)\s+to\s+([^,.]+)'

    matches = re.search(pattern, sentence, re.IGNORECASE)

    if matches:
        from_location = matches.group(1)
        to_location = matches.group(2)
        cou1 = city_to_country(from_location)
        cou2 = city_to_country(to_location)
        if is_city(from_location) == False and  is_city(to_location)== False and from_location in geo and to_location in geo:
            return True
        elif cou1!=cou2 and from_location in geo and to_location in geo:
            return True
    return False


def is_tran_two(text):
    # sentences = sentence_segmentation(text)
    if len(text) > 800:
        return False
    sentences = text.split(',')
    country_ver = []

    for sentence in sentences:
        people = get_complete_person_names(sentence)
        if len(people)>0 or has_subject_pronoun(sentence):
            # print(people)
            #如果主语是国家，跳过，我需要个人转移
            subject = get_subject(sentence)
            # print(subject)

            if subject is not None:
                # print(subject)
                if is_country(subject) or is_city(subject) or (subject in country_dic):
                    # print(is_country(subject))
                    # print(is_city(subject))
                    # print('lala')
                    continue

            if has_from_to(sentence):
                # print('!')
                return True
            geo = get_countries_process(sentence)
            country_ver = add_missing_elements(country_ver,geo)
            ver = get_lemmatized_verbs(sentence)

            # print(sentence)
            # print(ver)
            # print(geo)
            # print('-----------')

            if has_common_element(ver,trans_dic):
                common_ver = get_has_common_element(ver,trans_dic)
                geo = get_countries(sentence)
                # print(geo)

                if has_list_b_words_after_list_a(sentence,common_ver,geo): #and prepositions_is_country(sentence)
                    for token in geo:
                        if not is_city(token) and is_country(token) or (token in country_dic) or (token in CountryCode):
                            # print('~~')
                            # print(common_ver)
                            # print(geo)
                            return True
                    if len(country_ver)>1:
                        # print('!!!!')
                        # print(common_ver)
                        # print(geo)
                        return True

            continue
        else:
            continue
    # print('xx')
    return False


#做转移的方法
def is_tran_two_test(text):
    # sentences = sentence_segmentation(text)
    sentences = text.split(',')
    country_ver = []

    for sentence in sentences:
        people = get_complete_person_names(sentence)
        if len(people)>0 or has_subject_pronoun(sentence):
            # print(people)
            #如果主语是国家，跳过，我需要个人转移
            subject = get_subject(sentence)
            if subject is not None:
                # print(subject)
                if is_country(subject) or is_city(subject) or (subject in country_dic):
                    # print('lala')
                    continue

            if has_from_to(sentence):
                return True
            geo = get_countries_process(sentence)
            country_ver = add_missing_elements(country_ver,geo)
            ver = get_lemmatized_verbs(sentence)

            # print(sentence)
            # print(ver)
            # print(geo)
            # print('-----------')

            if has_common_element(ver,trans_dic):
                common_ver = get_has_common_element(ver,trans_dic)
                geo = get_countries(sentence)

                if has_list_b_words_after_list_a(sentence,common_ver,geo):
                    for token in geo:
                        if not is_city(token) and is_country(token) or (token in country_dic):
                            # print('~~')
                            # print(common_ver)
                            # print(geo)
                                return True
                    if len(country_ver)>1:
                        # print('!!!!')
                        # print(common_ver)
                        # print(geo)
                        return True
        else:
            continue
    return False


#保存为json数据
def add_data_to_json(file_path,abstracts,urls,titles,dates):
    # 读取现有的 JSON 数据
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    for i in range(len(abstracts)):
        # 添加新数据
        new_entry = {
            'title': titles[i],
            'abstract': abstracts[i],
            'url':urls[i],
            'date':dates[i]
        }
        data.append(new_entry)

    # 将更新后的数据写回 JSON 文件
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

#找状语
def is_adverb(sentence,target_word):
    # 处理输入句子
    doc = nlp(sentence)

    # 遍历句子中的每个词汇标记
    word_list = target_word.split(' ')
    for word in word_list:
        for token in doc:
            # 判断词是否与目标词匹配且具有 "dobj" 依存标签
            if token.text == word and token.dep_ == "dobj" :  #or "pobj"
                return True

    # 如果没有找到匹配的词汇标记，返回 False
    return None


#成分标签
def show_pos_tags(sentence):
    # 处理输入句子
    doc = nlp(sentence)

    # 遍历句子中的每个词
    for token in doc:
        print(f"词汇：{token.text}, 成分标签：{token.pos_}")

def find_adp_propn_pairs(sentence):

    # 处理输入句子
    doc = nlp(sentence)

    adp_propn_pairs = []  # 用于存储ADP和其后的第一个PROPN词的键值对

    # 遍历句子中的每个词
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.pos_ == "ADP":
            adp_index = i
            propn_found = False

            # 从ADP的位置开始向后查找PROPN词
            for j in range(i + 1, len(doc)):
                if doc[j].pos_ == "ADP":
                    break
                elif doc[j].pos_ == "PROPN":
                    adp_propn_pairs.append((token.text, doc[j].text))
                    propn_found = True
                    break

            # 如果找到了PROPN词，跳过ADP和PROPN之间的词
            if propn_found:
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    return adp_propn_pairs


def show_dependency_labels(sentence):
    # 处理输入句子
    doc = nlp(sentence)

    # 遍历句子中的每个词
    for token in doc:
        print(f"词汇：{token.text}, dep标签：{token.dep_}")


def show_labels(sentence):
    # 处理输入句子
    doc = nlp(sentence)

    # 遍历句子中的每个词
    for ent in doc.ents:
        print(f"词汇：{ent.text}, dep标签：{ent.label_}")

def city_list_to_country(city_list):
    result_list = []

    for city_name in city_list:
        username = str(random.choice(user_name))
        params = {
            "q": city_name,
            "maxRows": 1,  # 限制结果数量
            "username": username  # 使用你的GeoNames用户名
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        print(data)
        if "geonames" in data and len(data["geonames"]) > 0:
            # fclName = result_data["geonames"][0].get("fclName")
            # fclName = str(fclName).split(',')[0]
            # return fclName == 'city'
            countryName = data["geonames"][0].get("countryName")
            result_list.append(countryName)
    result_list = list(set(result_list))
    return result_list


def origin_to_destination(sentence):
    origin = ''
    destination = ''
    city_list = get_countries(sentence)
    country_list = city_list_to_country(city_list)
    preposition = get_prepositions(sentence)
    print(preposition)
    adp_propn = find_adp_propn_pairs(sentence)
    to_preposition = []
    in_preposition = []
    of_preposition = []
    for token in preposition:
        if token[0] == 'to':
            to_preposition.append(token)
        elif token[0] == 'in':
            in_preposition.append(token)
        elif token[0] == 'of':
            of_preposition.append(token)
    for token in adp_propn:
        if token[0] == 'to':
            to_preposition.append(token)
        elif token[0] == 'in':
            in_preposition.append(token)
        elif token[0] == 'of':
            of_preposition.append(token)
    print(city_list)
    print(country_list)
    if len(city_list) >= 2 and len(country_list) == 1:
        return country_list[0],country_list[0]


    if len(country_list) == 1:
        destination = country_list[0]
        origin = 'unknow'
        return origin,destination
    elif len(country_list) > 1:
        for token in to_preposition:
            # if token[0] == 'to' :
            token_one_country = find_propn_phrase(sentence, token[1])
            if token_one_country is not None and token_one_country in country_list or token_one_country in city_list:
                if city_to_country(token_one_country) is not None:
                    destination = city_to_country(token_one_country)
                else:
                    destination = token_one_country
                break
        if len(destination) == 0:
            for token in in_preposition:
                token_one_country = find_propn_phrase(sentence, token[1])
                if token_one_country is not None and token_one_country in country_list or token_one_country in city_list:
                    if city_to_country(token_one_country) is not None:
                        destination = city_to_country(token_one_country)
                    else:
                        destination = token_one_country
                    break
        if len(destination) == 0:
            for token in of_preposition:
                token_one_country = find_propn_phrase(sentence, token[1])
                if token_one_country is not None and token_one_country in country_list or token_one_country in city_list:
                    if city_to_country(token_one_country) is not None :
                        destination = city_to_country(token_one_country)
                    else:
                        destination = token_one_country
                    break
                # if city_to_country(token[1]) is not None:
                #     destination = city_to_country(token[1])
                # else:
                #     destination = token[1]
                # break
    else:
        origin = 'unknow'
        destination = 'unknow'
        return origin,destination

    #介词找不到，就用状语找
    if len(destination) == 0:
        for country in city_list:
            if is_adverb(sentence,country):
                if city_to_country(country) is not None:
                    destination = city_to_country(country)
                else:
                    destination = country
                break


    #目的地找完 找到了就找出发地
    if len(destination) == 0:
        origin = 'unknow'
        destination = 'unknow'
        return origin,destination
    else:
        if len(country_list) == 1:
            origin = 'unknow'
        elif len(country_list) == 2:
            for country in city_list:
                if country != destination and city_to_country(country) != destination:
                    if city_to_country(country) is not None:
                        origin = city_to_country(country)
                    else:
                        origin = country
                    break
        #句子中有三个国家且找到了目的地，用介词和dep标签来找出发地

        #这一块还不好找
        elif len(country_list) > 2:
            for token in preposition:
                token_one_country = find_propn_phrase(sentence,token[1])
                if (token_one_country in country_list or token_one_country in city_list) and token_one_country != destination and city_to_country(token_one_country) != destination:
                    if city_to_country(token_one_country) is not None:
                        origin = city_to_country(token_one_country)
                    else:
                        origin = token_one_country
                    break

    if len(origin) == 0:
        origin = 'unknow'
    return origin,destination


def find_propn_phrase(sentence, word):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(sentence)

    for token in doc:
        if token.text == word and token.pos_ == "PROPN":
            propn_phrase = [token.text]
            left_idx = token.i - 1
            right_idx = token.i + 1

            # 查找相邻的PROPN词
            while left_idx >= 0 and doc[left_idx].pos_ == "PROPN":
                propn_phrase.insert(0, doc[left_idx].text)
                left_idx -= 1

            while right_idx < len(doc) and doc[right_idx].pos_ == "PROPN":
                propn_phrase.append(doc[right_idx].text)
                right_idx += 1

            return " ".join(propn_phrase)

    return None  # 如果没有找到满足条件的词语，返回 None

def find_propn_phrase_subject(sentence, word):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(sentence)

    for token in doc:
        if token.text == word and token.pos_ == "PROPN":
            propn_phrase = [token.text]
            left_idx = token.i - 1
            right_idx = token.i + 1

            # 查找相邻的PROPN词
            while left_idx >= 0 and doc[left_idx].pos_ == "PROPN" or "NOUN":
                propn_phrase.insert(0, doc[left_idx].text)
                left_idx -= 1

            while right_idx < len(doc) and doc[right_idx].pos_ == "PROPN" or "NOUN":
                propn_phrase.append(doc[right_idx].text)
                right_idx += 1

            return " ".join(propn_phrase)

    return None  # 如果没有找到满足条件的词语，返回 None


if __name__ == '__main__':

    # # 例子
    sentence = '''Novak Djokovic has arrived back in Serbia's capital Belgrade after his deportation from Australia ended the world No. 1 tennis player's hopes of playing in the Australian Open.'''
    # # sentences = sentence_segmentation(sentence)
    # print(get_countries(sentence))
    #
    # sentence = '''Captain Wendie Renard scores a late headed winner as France beat Brazil in a thrilling Group F game in Brisbane at the 2023 Fifa Women's World Cup.'''
    # #
    # subjects = get_subject(sentence2)
    #
    # print(get_prepositions(sentence))
    # print(is_tran_two(sentence))
    # print(is_country('East Yorkshire'))
    # print(is_city('China'))
    # print(get_verbs(sentence))
    # print(get_lemmatized_verbs(sentence))
    origin,destination = origin_to_destination(sentence)
    print(origin,destination)
    # total = 0
    # ace = 0
    #
    # print(is_tran_two('''A Ukrainian beauty blogger's trolls ordeal and sweets driven all the way from Somerset to Ukraine.'''))
    # pass
    # with open('test.json','r',encoding='utf8') as fp:
    #     json_data = json.load(fp)
    #     print(is_tran_two(json_data[20]['text']))
        # for item in json_data:
        #     print(is_tran_two(item['text']))
            # total = total + 1
            # if str(is_tran_two(item['text'])).lower() == item['label']:
            #     ace = ace + 1

        # print(ace)
        # print(total)

#     text = '''Artist George Butler has recently returned to the UK after a month spent in Ukraine.
#
# He travelled throughout the country, sketching events there as they unfolded.
#
# In the past the award-winning artist has documented life in war zones like Syria and Afghanistan.'''
#
#     print(is_city('Cairo'))


    # abstract = '''A look at Egypt-Russia relations as Putin visits Cairo'''
    # origin,destination = origin_to_destination(abstract)
    # print(origin,destination)
    # print(city_list_to_country(['Howard', 'Tennessee']))



    # print(is_city('Kathmandu'))
    # abstract = '''Prince Harry will travel to the UK but Meghan will stay in California with their children.'''

    # print(get_prepositions(abstract))
    # #
    # origin,destination = origin_to_destination(abstract)
    # print(origin,destination)

    # print(show_pos_tags(abstract))

    # if get_countries  count = 1  目的地为这个地方 出发地未知
    # if get_countries  count > 1  以介词来区分
    # 如果get_countries 不为空 有to跟着的国家 那么to的是抵达的 另一个是出发的   介词to>in

    # result1 = get_countries(text)
    # result2 = get_countries(abstract)
    # result3 = get_countries_process(text)
    # result = get_prepositions(abstract)
    # country_list = get_countries_process(abstract)
    # city_list = get_countries(abstract)
    # # # #
    # print(country_list)
    # print(city_list)
    # print(is_adverb(abstract,'New York'))
    #
    # print(city_to_country('New York'))
    #
    # propn = find_propn_phrase(abstract,'Huberman')
    # print(propn)
    # propn1 = find_propn_phrase(abstract,'path')
    # print(propn1)
    # print(show_dependency_labels(abstract))
    # print(show_labels(abstract))
    # print(propn)
    # print(get_prepositions(abstract))
    # print(city_to_country('San Francisco'))
    # print(is_city(abstract))
    # print(is_country(abstract))
    # origin,destination = origin_to_destination(abstract)
    # print(origin,destination)
    # print(get_prepositions(abstract))
    # print(get_countries(abstract))
    # print(show_dependency_labels(abstract))
    # print('------')
    # print(show_pos_tags(abstract))


    # 对句子进行处理   处理json新闻数据
    # abstracts = []
    # urls = []
    # titles = []
    # dates = []
    # with open(r'E:\kaggle\input\news_category\News_Category_Dataset_v3.json', 'r') as f:
    #     jdata = f.read()
    # count = 0
    # jincheng = 0
    #
    # jdata2 = [json.loads(line) for line in jdata.split('\n') if line]
    # df = pd.DataFrame.from_records(jdata2)
    # for index, row in df.iterrows():
    #     print(jincheng)
    #     jincheng = jincheng + 1
    #     short_description = row['short_description']
    #     # 在这里可以添加你的处理逻辑，例如打印short_description或进行其他操作
    #     if is_tran_two(short_description):
    #         count = count+1
    #         url = row['link']
    #         urls.append(url)
    #         title = row['headline']
    #         titles.append(title)
    #         abstracts.append(short_description)
    #         date = row['date']
    #         dates.append(date)
    #         # print(short_description)
    #
    # file_path = 'huffpost.json'
    # add_data_to_json(file_path, abstracts, urls, titles, dates)
    # print(count)


    # # 读取 CSV 文件
    # csv_file_path = r'E:\kaggle\input\archive_test_tran\bbc_news.csv'  # 替换为你的 CSV 文件路径
    # result_data = pd.read_csv(csv_file_path)
    # # title = []
    # abstract = []
    # url = []
    #
    #
    # # 遍历前10行的 'description' 列
    # max_rows = 20000  # 指定要遍历的最大行数
    # for index, row in result_data.iterrows():
    #     if index < max_rows:
    #         description = row['description']
    #
    #         # 在这里进行 description 的处理
    #         description = str(description)
    #         if is_tran_two(description):
    #             abstract.append(description)
    #             url.append(row['link'])
    #             # title.append(str([row['title']]))
    #             print(description)
    #             print(row['link'])
    #         # if is_tran_two_test(description):
    #         #     list_two.append(description)
    #     else:
    #         break




    # result = [item for item in list_two if item not in list_one]
    # print('------')
    # for x in result:
    #     print(x)

    # text = '''WASHINGTON (Reuters) - President Donald Trump will travel to Texas to visit with the victims of the Santa Fe High School shooting on Thursday, White House spokeswoman Sarah Sanders said on Wednesday.'''
    #
    # origin,destination = origin_to_destination(text)
    # print(origin,destination)

    # city = 'Texas'
    # print(is_city(city))
    # 对句子进行处理
    # abstracts = []
    # urls = []
    # # 打开 JSON 文件
    # with open('test.json', 'r') as file:
    #     data_list = json.load(file)
    #
    # # 遍历 JSON 数据列表
    # for result_data in data_list:
    #     # 从每个数据对象中提取 abstract 和 url
    #     text = result_data.get('text')
    #     url = result_data.get('url')
    #
    #     # 将 abstract 和 url 添加到对应的列表中
    #     abstracts.append(text)
    #     urls.append(url)
    #
    # for sentence in abstracts:
    #     origin,destination = origin_to_destination(sentence)
    #     print(f"sentence:{sentence}")
    #     print(f"origin:{origin},destination:{destination}")



