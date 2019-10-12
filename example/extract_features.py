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
from preprocess import bert_data_utils
from preprocess import dataloader
from preprocess import tokenization

class Extractor(object):
    def __init__(self, config):
        self.model_dir = config['model_dir']
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.label_map_file = config['label_map_file']
        self.model_pb_path = config['model_pb_path']
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

        label2idx, idx2label = bert_data_utils.read_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx


        #init stop set
        self.stop_set = dataloader.get_stopwords_set(STOPWORD_FILE)

        #use default graph
        self.graph = tf.get_default_graph()
        restore_graph_def = tf.GraphDef()
        restore_graph_def.ParseFromString(open(self.model_pb_path, 'rb').read())
        tf.import_graph_def(restore_graph_def, name='')

        session_conf = tf.ConfigProto()
        self.sess = tf.Session(config=session_conf)
        self.sess.as_default()
        self.sess.run(tf.global_variables_initializer())

        self.input_ids_tensor = self.graph.get_operation_by_name('input_ids').outputs[0]
        self.input_mask_tensor = self.graph.get_operation_by_name('input_mask').outputs[0]
        self.segment_ids_tensor = self.graph.get_operation_by_name('segment_ids').outputs[0]
        self.is_training_tensor = self.graph.get_operation_by_name('is_training').outputs[0]

        self.sentence_features_tensor = self.graph.get_operation_by_name('sentence_features').outputs[0]

    def extract_features(self, text):
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)
        feed_dict = {
                self.input_ids_tensor: input_ids,
                self.input_mask_tensor: input_mask,
                self.segment_ids_tensor: segment_ids,
                self.is_training_tensor: False}
        batch_sentence_features = self.sess.run(self.sentence_features_tensor, feed_dict)

        sentence_features = batch_sentence_features[0]
        return sentence_features

    def trans_text2ids(self, text):
        if text[-1] in self.stop_set:
            text = text[: -1]
        example = bert_data_utils.InputExample(guid='1', text_a=text)
        seq_length = min(self.max_seq_length, len(text) + 2)
        feature = bert_data_utils.convert_single_example(1, example, self.label2idx,
                                                seq_length, self.tokenizer)
        input_ids = [feature.input_ids]
        input_mask = [feature.input_mask]
        segment_ids = [feature.segment_ids]
        #print(input_ids)
        #print(input_mask)
        #print(segment_ids)
        return input_ids, input_mask, segment_ids


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
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'

    extractor = Extractor(config)
    
    cand_tuples = read_candidates(DATA_DIR + '/sentence_uuids.tsv')
    feature_chunks = build_features_dict(cand_tuples, extractor)

    output_file = MODEL_DIR + '/memory.tsv'
    write_features(output_file, feature_chunks)

