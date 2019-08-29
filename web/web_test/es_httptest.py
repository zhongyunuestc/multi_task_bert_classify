#coding:utf-8
###################################################
# File Name: es_httptest.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年08月29日 星期四 18时21分57秒
#=============================================================
import requests
import codecs
import json


def read_queries(input_file):
    queries = []
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            if line == '':
                continue
            line_info = line.split('\t')
            query = line_info[0]
            queries.append(query)
    return queries
    

def write_candidates(output_file):
    pass



if __name__ == "__main__":
    queries = read_queries('errors.tsv')

    query = '介绍一下云桥'
    url = 'http://47.97.108.232:20015/findUserSentence/queryClassifyByES'
    #json = {'question': query, 'appId':'k55frqcd','pageSize': 50}
    json_data = {'question': query, 'appId':'k55frqcd','pageSize': 5}
    rs = requests.post(url=url, json=json_data)

    #print(rs.text)
    data = json.loads(rs.text)
    cands = data['data']['data']
    sims = [item['corpus'] for item in cands]
    print(sims)
    pass
