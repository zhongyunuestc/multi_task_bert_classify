#coding:utf-8
###################################################
# File Name: extract_features.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年08月23日 星期五 15时12分04秒
#=============================================================
import os
import sys
import csv
import json
import codecs
import logging
import numpy as np


sys.path.append('../')



from setting import *
from web.evaluator import Evaluator

def read_candidates(input_file):
    cand_tuples = []
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            if len(line_info) < 2:
                continue
            cand = line_info[0]
            uuid = line_info[1]
            cand_tuples.append((cand, uuid))
    return cand_tuples


def build_features_dict(cand_tuples, extractor):
    feature_chunks = []
    for cand, uuid in cand_tuples:
        features = extractor.extract_features(cand)
        wrapper = {}
        wrapper['text'] = cand
        wrapper['features'] = [float(feature) for feature in features]
        wrapper['uuid_code'] = uuid
        feature_chunks.append(wrapper)
    return feature_chunks


def write_features(output_file, feature_chunks):
    with codecs.open(output_file, 'w', 'utf8') as fw:
        for chunk in feature_chunks:
            json_str = json.dumps(chunk, ensure_ascii=False)
            fw.write(json_str + '\n')






if __name__ == '__main__':
    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 128
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'

    extractor = Evaluator(config)
    
    cand_tuples = read_candidates(DATA_DIR + '/sentence_uuids.tsv')
    feature_chunks = build_features_dict(cand_tuples, extractor)

    output_file = MODEL_DIR + '/memory.tsv'
    write_features(output_file, feature_chunks)

