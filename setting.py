#coding:utf-8
###################################################################
# File Name: setting.py
# Author: Meng Zhao
# mail: @
# Created Time: Wed 21 Mar 2018 04:50:40 PM CST
#=============================================================
import os
import logging
import logging.handlers
import tensorflow as tf

#version
SERVER_NAME = 'search_person'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = BASE_DIR + '/data'
MODEL_DIR = BASE_DIR + '/example/runs/' + SERVER_NAME




#bert server
TF_SERVING_REST_PORT = 17122
TF_SERVING_CLIENT_PORT = 17123
TF_SERVING_SIGNATRUE_NAME = 'predict_text'


#test
ES_CLIENT_ADDRESS = 'http://172.16.159.164:20015/essential/findUserSentence/cleanAllFeature'

#release
#ES_CLIENT_ADDRESS = 'http://47.99.17.126:20016/essential/findUserSentence/cleanAllFeature'




#files path
#STOPWORD_FILE = DATA_DIR + '/stopword_data/stop_words'
STOPWORD_FILE = DATA_DIR + '/stopword_data/stop_symbol'
LABEL_FILE = MODEL_DIR + '/labels.txt'
LABEL_MAP_FILE = MODEL_DIR + '/label_map'
CODE_FILE = MODEL_DIR + '/labelcode'



LOG_DIR = BASE_DIR + '/log/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(lineno)s]%(filename)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
log_handler = logging.handlers.TimedRotatingFileHandler(filename=LOG_DIR+'searchperson.log', when='D', interval=1, backupCount=10)
log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger('').addHandler(log_handler)
