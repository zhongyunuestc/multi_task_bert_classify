#coding:utf-8
###################################################
# File Name: predict.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年08月23日 星期五 16时12分48秒
#=============================================================

import os
import sys
import csv
import json
import codecs
import logging
import datetime
import numpy as np


sys.path.append('../../')

from sklearn import metrics
from preprocess import bert_data_utils
from preprocess import dataloader
from preprocess import tokenization

from setting import *
from web.evaluator import Evaluator

def cosine(q, a):
    pooled_len_1 = np.sqrt(np.sum(np.multiply(q, q), 1))
    pooled_len_2 = np.sqrt(np.sum(np.multiply(a, a), 1))

    pooled_mul_12 = np.sum(np.multiply(q, a), 1)
    score = np.divide(pooled_mul_12, np.multiply(pooled_len_1, pooled_len_2) + 1e-8)
    return score


def read_feature_dict(input_file):
    cand2features = {}
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            text = line_info[0]
            features = json.loads(line_info[1])['features']
            cand2features[text] = features
    cand2features = {key: np.array(value).astype(float) for key, value in cand2features.items()}
    return cand2features

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


def read_similar_and_norm_map(input_file):
    sim2norm = {}
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            if line == '':
                continue
            line_info = line.split('\t')
            sim = line_info[0]
            norm = line_info[1]
            sim2norm[sim] = norm
    return sim2norm



def read_queries_and_answers(input_file):
    queries = []
    answers = []
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:     
            line = line.strip()
            if line == '':  
                continue    
            line_info = line.split('\t')
            query = line_info[0]
            answer = line_info[1]
            queries.append(query)
            answers.append(answer)
    return queries, answers     


def predict_queries(pred_instance, queries, candidates, cand2features):
    target_ids = []
    for query in queries:
        start_time = datetime.datetime.now()
        target_id = predict_single_query(pred_instance, query, candidates, cand2features)
        end_time = datetime.datetime.now()
        cost_time = (end_time - start_time).total_seconds() * 1000
        print('cost time:', cost_time)
        #exit()
        target_ids.append(target_id)
    return target_ids



def predict_single_query(pred_instance, query, candidates, cand2features):
    cur_features = pred_instance.extract_features(query)
    cand_features = [cand2features[cand] for cand in candidates]
    cur_features = np.reshape(cur_features, [1, -1])
    sim_scores = cosine(cur_features, cand_features)
    #print(sim_scores)
    #exit()
    return np.argmax(sim_scores)


def test_cosine_similarity():
    A = np.array([1, 0, 1])
    B = np.array([[1, 2, 3], [1, 0, 1]])
    A = np.reshape(A, [1, -1])
    #B = np.reshape(B, [1, -1])
    print(np.shape(A))
    print(np.shape(B))
    #rs = metrics.pairwise.cosine_similarity(A, B)
    rs = cosine(A, B)
    print(rs)

def write_result(output_file, queries, answers, candidates, target_ids, sim2norm):
    with codecs.open(output_file, 'w', 'utf8') as fw:
        fw.write('query\ttruth\tpred_sim\tpred_norm\n')
        for query, answer, target_id in zip(queries, answers, target_ids):
            fw.write(query + '\t' + answer + '\t' + candidates[target_id] + '\t' + sim2norm[candidates[target_id]] +'\n')


if __name__ == '__main__':
    sim2norm = read_similar_and_norm_map('map.tsv')

    cand2features = read_feature_dict('features.tsv')
    print(len(cand2features))

    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 32
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'

    test_cosine_similarity()
    pred_instance = Evaluator(config)
    
    #queries = read_queries('queries.tsv')
    queries, answers = read_queries_and_answers('queries.tsv')
    candidates = list(cand2features.keys())
    target_ids = predict_queries(pred_instance, queries, candidates, cand2features)
    write_result('output.tsv', queries, answers, candidates, target_ids, sim2norm)
    
    targets = [candidates[i] for i in target_ids]
    count = 0.0
    for left, right in zip(answers, targets):
        if left == right or left == sim2norm[right]:
            count += 1
    acc = count / len(answers)
    print('acc:', acc)



