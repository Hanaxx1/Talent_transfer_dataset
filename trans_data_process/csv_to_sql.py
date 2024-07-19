#欢迎帅哥
#开发时间: 2023-09-12 14:11

import find_zhuyu as un
import csv
import pandas as pd
import chardet
import json
import ast

# def insert_to_sql_one(title,date,url,content,abstract):
#
#     conn = pymysql.connect(
#         host="localhost",
#         user="root",
#         password="cg950523",
#         database="geo"
#     )
#     # 创建游标对象
#     cursor = conn.cursor(echo $DISPLAY)
#
#     try:
#         origin, destination = un.origin_to_destination(abstract)
#         cause = 'unknow'
#         # 准备插入数据的SQL语句
#         values =(
#             title,
#             abstract,
#             cause,
#             date,
#             url,
#             content,
#             origin,
#             destination
#         )
#         sql = "INSERT INTO tran_data (title, abstract, cause, date, url, content, origin, destination) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
#
#         # 插入数据
#         cursor.execute(sql, values)
#         # 提交更改
#         conn.commit()
#
#     except pymysql.Error as err:
#         # 处理MySQL数据库异常
#         print("MySQL错误:", err)
#
#     cursor.close()
#     conn.close()

def data_to_local_json(json_file_path,title,date,url,content,abstract):
    origin, destination = un.origin_to_destination(abstract)

    cause = 'unknow'
    # 创建新的JSON数据对象
    new_data = {
        "title": title,
        "abstract" : abstract,
        "cause" : cause,
        "date": date,
        "url": url,
        "content": content,
        "origin": origin,
        "destination": destination
    }

    # 以追加模式打开文件，如果文件不存在则创建
    with open(json_file_path, 'a') as file:
        # 将新数据对象写入文件
        json.dump(new_data, file)
        file.write('\n')  # 添加换行符以分隔不同数据


def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


def data_to_local_json2(json_file_path,content,date):


    # 创建新的JSON数据对象
    new_data = {
        "content": content,
        "date": date
    }

    # 以追加模式打开文件，如果文件不存在则创建
    with open(json_file_path, 'a') as file:
        # 将新数据对象写入文件
        json.dump(new_data, file)
        file.write('\n')  # 添加换行符以分隔不同数据


def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']

if __name__ == '__main__':

    csv_file_path = r'workplace/geo_to_ssh/geo/news_data/nyt-metadata.csv'
    json_file_path = r'workplace/geo_to_ssh/geo/result_data/result_complete.json'


    # enc = get_encoding(csv_file_path)
    # print(enc)
    start_row = 0
    num_rows = 100000
    #一共2688878
    # 打开CSV文件并读取数据
    # data = pd.read_csv(csv_file_path,encoding='utf-8',skiprows=range(1,2041770),nrows=num_rows)
    data = pd.read_csv(csv_file_path,encoding='utf-8',skiprows=range(1,500000),nrows=num_rows)
    # ,low_memory=False
    #GB2312 gb18030
    # 遍历前10行的 'description' 列
    sum = 0
    count = 0
    for index, row in data.iterrows():
        print(index)
        count = count + 1
        if count == 200:
            count = 0
            try:
                with open(r'workplace/geo_to_ssh/geo/record.txt', 'w') as file:
                    file.write(index)
            except:
                pass
        abstract = str(row['abstract'])

        if un.is_tran_two(abstract):
            data_dict = ast.literal_eval(row['headline'])
            title = data_dict['main']
            date = row['pub_date'].split(' ')[0]
            content = row['lead_paragraph']
            url = row['web_url']
            print('~')
            sum = sum + 1
            # insert_to_sql_one(title,date,url,content,abstract)
            data_to_local_json(json_file_path,title,date,url,content,abstract)

    print(sum)






