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


sys.path.append('../../')



from setting import *
from web.evaluator import Evaluator

def read_candidates(input_file):
    candidates = []
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            cand = line_info[0]
            candidates.append(cand)
    return candidates


def build_features_dict(texts, extractor):
    text2features = {}
    for text in texts:
        features = extractor.extract_features(text)
        text2features[text] = [str(feature) for feature in features]
    return text2features


def write_features(output_file, text2features):
    with codecs.open(output_file, 'w', 'utf8') as fw:
        for text in text2features:
            feature_wrapper = {}
            feature_wrapper['features'] = text2features[text]
            json_str = json.dumps(feature_wrapper)
            fw.write(text + '\t' + json_str + '\n')
        #json_str = json.dumps(text2features, ensure_ascii=False)
        #fw.write(json_str)





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
    
    candidates = read_candidates('candidates.tsv')
    text2features = build_features_dict(candidates, extractor)
    print(len(text2features))

    write_features('features.tsv', text2features)

