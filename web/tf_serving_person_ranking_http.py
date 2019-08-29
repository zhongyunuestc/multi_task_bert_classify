#coding:utf-8
###################################################
# File Name: httpserver.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年06月06日 星期三 15时24分44秒
#=============================================================

import os
import sys
#import ujson as json
import ujson as json
import tornado.web
import tornado.ioloop


import numpy as np
import datetime
import traceback


sys.path.append('../')


import tf_serving_evaluator

#from tornado.concurrent import run_on_executor
#from concurrent.futures import ThreadPoolExecutor


from multiprocessing import Pool, TimeoutError

from setting import *
from preprocess import dataloader
from common.segment.segment_client import SegClient


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PredictHttpServer(tornado.web.RequestHandler):
    #def __init__(self, application, request, **kwargs):
    #    super(tornado.web.RequestHandler, self).__init__()


    def initialize(self, pred_instance):
        self.pred_instance = pred_instance


    def check_arguments(self):
        if 'method' not in self.request.arguments:
            raise Exception('method is required (predict or getSupportIntentions)')
        
        req_method = self.get_argument('method')
        if 'text' not in self.request.arguments and req_method == 'predict':
            raise Exception('text is required')
        if req_method not in ['predict', 'getSupportIntentions']:
            raise Exception("method must be 'predict' or 'getSupportIntentions'")

    def get_intents_table(self):
        table = []
        idx2label = self.pred_instance.idx2label
        label2code = self.pred_instance.label2code

        for idx in idx2label:
            intent_dict = {}
            intent_dict['id'] = int(idx) + 1
            
            label = idx2label[idx]
            intent_dict['title'] = label
            intent_dict['name'] = label2code[label.lower()]
            table.append(intent_dict)

        table_json = json.dumps(table, ensure_ascii=False)
        return table_json


    def http_predict(self):
        try:
            text_uni = self.get_argument('text')
            logging.info(text_uni)
            response = []
            trunk = self.pred_instance.evaluate(text_uni)
            for cur_label, cur_code, cur_score in trunk:
                intent_item = {}
                intent_item['title'] = cur_label
                intent_item['name'] = cur_code
                intent_item['score'] = str(cur_score)
                response.append(intent_item)
        except Exception as err:
            raise err
        return response 

    @tornado.gen.coroutine
    def get(self):
        err_dict = {}
        try:
            self.check_arguments()     
            method_type = self.get_argument('method')
            if method_type == 'predict':
                response = self.http_predict()
            else:
                response = self.get_intents_table()
            response_json = json.dumps(response, ensure_ascii=False)
            self.write(response_json)

        except Exception as err:
            err_dict['errMsg'] = traceback.format_exc()
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(traceback.format_exc())
        
    @tornado.gen.coroutine
    def post(self):
        err_dict = {}
        try:
            self.check_arguments()
            method_type = self.get_argument('method')
            if method_type == 'predict':
                response = self.http_predict()
            else:
                response = self.get_intents_table()
            response_json = json.dumps(response, ensure_ascii=False)
            self.write(response_json)

        except Exception as err:
            err_dict['errMsg'] = traceback.format_exc()
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(traceback.format_exc())
    
    @tornado.gen.coroutine
    def head(self):
        self.write('OK')


class RankingHttpServer(tornado.web.RequestHandler):

    def initialize(self, pred_instance):
        self.pred_instance = pred_instance


    @tornado.gen.coroutine
    def head(self):
        self.write('OK')

    def online_extract_features(self):
        query = self.data['query']
        logging.info('extracted query:' + query)
        result = {}
        features = self.pred_instance.extract_features(query)
        result['features'] = [float(feature) for feature in features]
        return result

    def online_ranking(self):
        start_time = datetime.datetime.now()
        query = self.data['query']
        cand_pool = self.data['candidates']
        logging.info('ranking query:' + query)
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('candidates param cost' + str(cost))

        sorted_indices, sim_scores = self.pred_instance.ranking(query, cand_pool)

        top_sorted_indices = sorted_indices[: 3]
        logging.info(type(sim_scores))
        logging.info(type(sorted_indices))
        top_scores = list(sim_scores[top_sorted_indices])
        top_scores[0] = top_scores[0] * 10
        top_scores = softmax(top_scores)

        tmp_time = datetime.datetime.now()
        result = []
        for idx, score in zip(top_sorted_indices, top_scores):
            title = cand_pool[idx]['text']
            start_time = datetime.datetime.now()
            name = cand_pool[idx]['userId']
            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('get result name cost' + str(cost))

            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000 
            logging.info('get result name cost' + str(cost))

            start_time = datetime.datetime.now()
            item = {}
            item['title'] = title
            item['name'] = name
            item['score'] = score
            result.append(item)
            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('wrap one result cost' + str(cost))
        end_time = datetime.datetime.now()
        cost = (end_time - tmp_time).total_seconds() * 1000
        logging.info('wrap features cost' + str(cost))
        return result
    
    def prepare(self):
        start_time = datetime.datetime.now()
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.data = json.loads(self.request.body)
        else:
            self.data = {}
            for k in self.request.arguments:
                self.data[k] = self.get_argument(k)
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('json convert cost' + str(cost))

    @tornado.gen.coroutine
    def main_process(self):
        err_dict = {}
        try:
            start_time = datetime.datetime.now()
            logging.info(self.data.keys()) 
            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('get data keys cost' + str(cost))

            method_type = self.data['method'] 
            if method_type == 'extract':
                response = self.online_extract_features()
            else:
                response = self.online_ranking()
            
            start_time = datetime.datetime.now()
            response_json = json.dumps(response, ensure_ascii=False, double_precision=32)
            #response_json = {'result': response}
            self.write(response_json)
            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('wrap result cost' + str(cost))
        except Exception as err:
            err_dict['errMsg'] = traceback.format_exc()
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(traceback.format_exc())


    def get(self):
        self.main_process()

    def post(self):
        self.main_process()


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 32
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'
    config['tf_serving_url'] = 'http://localhost:'+str(TF_SERVING_REST_PORT)+'/v1/models/default:predict'
    config['signature_name'] = 'predict_text'

    pred_instance = tf_serving_evaluator.Evaluator(config)
    pred_instance.evaluate('班车报表')


    application = tornado.web.Application([
            (r"/searchperson", PredictHttpServer, 
            dict(pred_instance=pred_instance,)),
            (r"/personranking", RankingHttpServer,
            dict(pred_instance=pred_instance))
        ])
    application.listen(TF_SERVING_CLIENT_PORT)
   

    tornado.ioloop.IOLoop.instance().start()
