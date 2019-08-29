#coding:utf-8
###################################################
# File Name: eval.py
# Author: Meng Zhao
# mail: @
# Created Time: Fri 23 Mar 2018 09:27:09 AM CST
#=============================================================
import os
import sys
import csv
import codecs
import gensim
import numpy as np
import tensorflow as tf
sys.path.append('../')


from preprocess import tokenization
from preprocess import bert_data_utils
from preprocess import dataloader
from tensorflow.contrib import learn

from setting import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "" # not use GPU



flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("do_lower_case", True, "Whether to lower case the input text")
flags.DEFINE_string("vocab_file", MODEL_DIR + "/vocab.txt", "vocab file")
flags.DEFINE_string("model_dir", MODEL_DIR + '/checkpoints', "model file")
flags.DEFINE_string("label_map_file", MODEL_DIR + '/label_map', "label map file")
flags.DEFINE_string("label_file", DATA_DIR + '/labels.tsv', "label map file")


tf.flags.DEFINE_string("question_user_map_file", DATA_DIR + '/question_to_user_id.txt', "candidates data source.")
tf.flags.DEFINE_string("candidates_file", DATA_DIR + '/candidates.tsv', "candidates data source.")
tf.flags.DEFINE_string("test_data_file", DATA_DIR + '/test.tsv', "Test data source.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_sequence_length", 32, "max sequnce length")


tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

flags.DEFINE_string(
                "output_dir", 'output',
                "The output directory where the model checkpoints will be written.")

flags.DEFINE_string('model_method', 'sentence', 'experiments')


def cosine(q, a):
    pooled_len_1 = np.sqrt(np.sum(np.multiply(q, q), 1))
    pooled_len_2 = np.sqrt(np.sum(np.multiply(a, a), 1))

    pooled_mul_12 = np.sum(np.multiply(q, a), 1)
    score = np.divide(pooled_mul_12, np.multiply(pooled_len_1, pooled_len_2)+1e-8)
    #print score
    return score
    #return tf.clip_by_value(score, 1e-5, 0.99999)

def get_feed_data(features):
    feed_input_ids_a = [item.input_ids for item in features]
    feed_input_mask_a = [item.input_mask for item in features]
    feed_segment_ids_a = [item.segment_ids for item in features]
    feed_label_ids = [item.label_id for item in features]


    return (feed_input_ids_a, feed_input_mask_a, feed_segment_ids_a,
            feed_label_ids)


def get_candidates(file_name):
    candidates = []
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for item in fr:
            item = item.strip().lower()
            candidates.append(item)
    return candidates

def get_label_dict(file_name):
    label2idx = {}
    idx2label = {}
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for item in fr:
            item = item.strip()
            item_info = item.split('\t')
            idx = item_info[0]
            label = item_info[1]
            label2idx[label] = idx
            idx2label[idx] = label
    return label2idx, idx2label


def get_cand_input_features(candidates, tokenizer):
    pass



def get_cand_output_features(sess, 
                             batch_size, 
                             cand_input_features,
                             input_ids_a,
                             input_mask_a,
                             segment_ids_a,
                             is_training,
                             sentence_features_tensor):
    batches = dataloader.batch_iter(list(cand_input_features), batch_size, 1, shuffle=False)
    cand_output_features = []
    for batch in batches:
        feed_input_ids = [item.input_ids for item in batch]
        feed_input_mask = [item.input_mask for item in batch]
        feed_segment_ids = [item.segment_ids for item in batch]
        feed_dict = {input_ids_a: feed_input_ids,
                     input_mask_a: feed_input_mask,
                     segment_ids_a: feed_segment_ids,
                     is_training: False}


        sentence_features = sess.run(sentence_features_tensor, feed_dict)
        cand_output_features.extend(sentence_features)

    #cand_output_feature_dict = {}
    #for i, item in enumerate(cand_output_features, 1):
    #    cand_output_feature_dict[i] = item

    return cand_output_features



def gen_candidate_tfidf_model(candidates, tokenizer):
    #corpus 
    splited_candidates = [tokenizer.tokenize(item) for item in candidates]
    tfidf_dict = gensim.corpora.Dictionary(splited_candidates)
    corpus = [tfidf_dict.doc2bow(text) for text in splited_candidates]
    #tfidf
    tfidf_model = gensim.models.TfidfModel(corpus)

    #sim_indices
    corpus_tfidf = tfidf_model[corpus]
    tfidf_sim_indices= gensim.similarities.MatrixSimilarity(corpus_tfidf)
    #print(tfidf_model)

    #exit()
    return tfidf_dict, tfidf_model, tfidf_sim_indices

def build_dict(tags):
    tag2idx = {}
    idx2tag = {}
    for idx, tag in enumerate(tags):
        tag2idx[tag] = idx
        idx2tag[idx] = tag
    return tag2idx, idx2tag


def seq_predict(sess,):
    pass



def pooled_predict(sess,):
    pass


def ranking_pool_by_tfidf_model(pool, tokenizer, top=200):
    text_pool = [idx2cand[int(item)] for item in pool]
    tfidf_dict, tfidf_model, tfidf_sim_indices = gen_candidate_tfidf_model(text_pool, tokenizer)
    tokens = tokenizer.tokenize(feature.raw_text)
    cur_bow = tfidf_dict.doc2bow(tokens)       
    cur_tfidf = tfidf_model[cur_bow]           
    sims = tfidf_sim_indices[cur_tfidf]        
    sorted_ids = np.argsort(-sims)             
    #print('raw text:', feature.raw_text)       
    #for item in sorted_ids[:10]:               
    #    print(pool[item])                      
    #exit()
    pool = np.array(pool)
    absolute_ids = pool[sorted_ids[:top]]
    #for item in absolute_ids:
    #    print(item)
    #exit()
    return absolute_ids


def read_question_user_map(file_name):
    cand2user = {}
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line_info = line.split('\t')
            cand = line_info[0]
            user_name = line_info[1]
            user_id = line_info[2]
            cand2user[cand] = (user_name, user_id)
    return cand2user


def eval():
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    cand2user = read_question_user_map(FLAGS.question_user_map_file)

    candidates = get_candidates(FLAGS.candidates_file)
    cand2idx, idx2cand = build_dict(candidates)

    label_map = cand2idx

    #tfidf_dict, tfidf_model, tfidf_sim_indices = gen_candidate_tfidf_model(candidates, tokenizer)    

    features, cand_input_features = bert_data_utils.file_based_convert_examples_to_features_with_candidates(FLAGS.test_data_file,
                                                                                                                label_map,
                                                                                                                FLAGS.max_sequence_length,
                                                                                                                tokenizer,
                                                                                                                candidates) 


    print(np.shape(cand_input_features))
    print('\nEvaluating...\n')

    #Evaluation
    graph = tf.Graph()
    with graph.as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():

            #type 2
            checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            #tensors we feed
            input_ids_a = graph.get_operation_by_name('input_ids').outputs[0]
            input_mask_a = graph.get_operation_by_name('input_mask').outputs[0]
            segment_ids_a = graph.get_operation_by_name('segment_ids').outputs[0]

            is_training = graph.get_operation_by_name('is_training').outputs[0]
            

             
            #seq
            sentence_features_tensor = graph.get_operation_by_name('sentence_features').outputs[0]
            
           

            candidate_size = len(candidates)
            #batches = dataloader.batch_iter(list(features), candidate_size, 1, shuffle=False)
            batches = dataloader.batch_iter(list(features), 1, 1, shuffle=False)
           
            print('calc candidate features...')
            cand_output_features = get_cand_output_features(sess, 
                                                            256,
                                                            cand_input_features,
                                                            input_ids_a, 
                                                            input_mask_a,
                                                            segment_ids_a,
                                                            is_training, 
                                                            sentence_features_tensor)            

            print('calc is down...')
            #print(len(cand_output_feature_dict))
            #collect the predictions here
            corrects = 0.0
            user_corrects = 0.0
            count = 0.0
            predicts = []
            all_sentence_features = []
            all_truth_label_ids = []
            for batch in batches:
                (feed_input_ids_a, feed_input_mask_a, feed_segment_ids_a,
                 feed_label_ids) = get_feed_data(batch)
                
                pre_feed_dict = {input_ids_a: feed_input_ids_a,
                             input_mask_a: feed_input_mask_a,
                             segment_ids_a: feed_segment_ids_a,
                             is_training: False}


                batch_sentence_features = sess.run(sentence_features_tensor, pre_feed_dict)
                all_sentence_features.extend(batch_sentence_features)
    
            for sentence_feature, feature in zip(all_sentence_features, features):
                #absolute_ids = ranking_pool_by_tfidf_model(pool, tokenizer, top=200)
                #cand_ids = [cand_input_features[int(i)-1].input_ids_a for i in absolute_ids]
                #cand_num = len(cand_ids)                                   
                #print(candidates[absolute_ids[0]-1].split())               
                #print(tokenizer.convert_ids_to_tokens(cand_ids[0]))        
                #exit()


                expand_anchor_sentence_features = [sentence_feature] * candidate_size
                batch_probs_value = cosine(expand_anchor_sentence_features, cand_output_features)

                best_id = np.argmax(batch_probs_value)
                truth_id = feature.label_id
                #print(best_id)
                #print(truth_id)
                #print(batch_probs_value[best_id])
        
                predicts.append(candidates[best_id]) 
                if best_id == truth_id:
                    corrects += 1
                if cand2user[candidates[truth_id]] == cand2user[candidates[best_id]]:
                    user_corrects += 1
                count += 1
                #print('truth:', candidates[truth_id])
                #print('pred:', candidates[best_id])
                #print('truth id:', truth_id)
                #print('pred id:', best_id)
                #exit()
                
    print('count:', count)
    print('corrects:', corrects)
    print('acc:', corrects / count)
    print('user acc:', user_corrects / count)
    raw_examples = list(bert_data_utils.get_data_from_file(FLAGS.test_data_file))


    with codecs.open(MODEL_DIR + '/rs.txt', 'w', 'utf8') as fw:
        for left, right in zip(raw_examples, predicts):
            fw.write(left.text_a + '\t' + right + '\n')

    #truth_label_ids = np.array([item.label_id for item in features])
    #write predictions to file


if __name__ == '__main__':
    eval()
