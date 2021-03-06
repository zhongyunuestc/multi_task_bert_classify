# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import sys
import random
import datetime
import codecs
import shutil

import numpy as np
import tensorflow as tf


sys.path.append('../')

from models import modeling
from models import optimization
from preprocess import tokenization



flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
        "data_dir", None,
        "The input data dir. Should contain the .tsv files (or other data files) "
        "for the task.")

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                                        "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
        "do_predict", False,
        "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                                     "Total number of training epochs to perform.")

flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                                         "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                                         "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")

tf.flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")


#custom
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_string("stopword_file", "stopword_data/stop_symbol", "stopword file")


flags.DEFINE_string("candidates_file", 'candidates.tsv', "candidates data source.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                             input_ids,
                             input_mask,
                             segment_ids,
                             label_id,
                             is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class PointwiseInputFeatures(object):
    def __init__(self,
                input_ids,
                input_mask,
                segment_ids,
                answer_input_ids,
                answer_input_mask,
                answer_segment_ids,
                label_id,
                is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.answer_input_ids = answer_input_ids
        self.answer_input_mask = answer_input_mask
        self.answer_segment_ids = answer_segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines



class WeaverProcessor(DataProcessor):
    def __init__(self):
        self.stopword_set = None

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


    def get_labels(self, data_dir):
        with codecs.open(os.path.join(data_dir, 'labels.tsv'), 'r', 'utf8') as fr:
            labels = []
            for line in fr:
                line = line.strip().lower()
                labels.append(line)

        return set(labels)

    def set_stopwords(self, stopword_data_dir):
        with codecs.open(stopword_data_dir, 'r', 'utf8') as fr:
            words = []
            for line in fr:
                line = line.strip().lower()
                words.append(line)
        self.stopword_set = set(words)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if line[0] == '':
                continue
            guid = "%s-%d" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0]).lower()
            if text_a[-1] in self.stopword_set:
                text_a = text_a[: -1]
            if text_a == '':
                continue
            label = tokenization.convert_to_unicode(line[1]).lower()
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class PointwiseProcessor(DataProcessor):
    def __init__(self):
        self.stopword_set = None

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "pointwise_similar_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "pointwise_similar_dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "pointwise_similar_test.tsv")), "test")

    def set_stopwords(self, stopword_data_dir):
        with codecs.open(stopword_data_dir, 'r', 'utf8') as fr:
            words = []
            for line in fr:
                line = line.strip().lower()
                words.append(line)
        self.stopword_set = set(words)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            # Only the test set has a header
            guid = "%s-%d" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0]).lower()
            text_b = tokenization.convert_to_unicode(line[1]).lower()
            if text_a[-1] in self.stopword_set:
                text_a = text_a[: -1]
            if text_b[-1] in self.stopword_set:
                text_b = text_b[: -1]
            label = tokenization.convert_to_unicode(line[2]).lower()
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class MTDNN(object):
    
    def __init__(self,
                 bert_config,
                 num_labels,
                 use_one_hot_embeddings,
                 max_seq_length,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps):

        self.bert_config = bert_config
        self.num_labels = num_labels
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_seq_length = max_seq_length
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps        

        self._build_model()
        pass

    def _build_model(self):

        self._init_placeholders()
        self._init_bert_model()
        
        self.inference()
        #self.set_classify_loss()
        self.set_classify_train_op()
        
        self.similar_inference()
        #self.set_similar_loss()
        self.set_pointwise_similar_train_op()

        self.pairwise_similar_inference()
        self.set_pairwise_similar_train_op()

        
    
    def _init_placeholders(self):
        self.input_ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, None], name='segment_ids')

        self.answer_input_ids = tf.placeholder(tf.int32, [None, None], name='answer_input_ids')
        self.answer_input_mask = tf.placeholder(tf.int32, [None, None], name='answer_input_mask')
        self.answer_segment_ids = tf.placeholder(tf.int32, [None, None], name='answer_segment_ids')

        self.negative_input_ids = tf.placeholder(tf.int32, [None, None], name='negative_input_ids')
        self.negative_input_mask = tf.placeholder(tf.int32, [None, None], name='negative_input_mask')
        self.negative_segment_ids = tf.placeholder(tf.int32, [None, None], name='negative_segment_ids')

        self.label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.task_type_tensor =  tf.placeholder(tf.string, None, name='task_type')

    
    def _init_bert_model(self):
        self.model = modeling.BertModel(config=self.bert_config,
                               is_training=self.is_training,
                               input_ids=self.input_ids,
                               input_mask=self.input_mask,
                               token_type_ids=self.segment_ids,
                               use_one_hot_embeddings=self.use_one_hot_embeddings,
                               scope='bert')


        self.answer_model = modeling.BertModel(config=self.bert_config,
                               is_training=self.is_training,
                               input_ids=self.answer_input_ids,
                               input_mask=self.answer_input_mask,
                               token_type_ids=self.answer_segment_ids,
                               use_one_hot_embeddings=self.use_one_hot_embeddings,
                               scope='bert')        


        self.negative_model = modeling.BertModel(config=self.bert_config,
                               is_training=self.is_training,
                               input_ids=self.negative_input_ids,
                               input_mask=self.negative_input_mask,
                               token_type_ids=self.negative_segment_ids,
                               use_one_hot_embeddings=self.use_one_hot_embeddings,
                               scope='bert') 


        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names
                ) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string) 


    def inference(self):
        #output_layer = self.model.get_pooled_output()
        output_layer = self.model.get_sequence_output()
        output_layer = tf.reduce_mean(output_layer, 1)
        #output_layer = tf.identity(output_layer, 'sentence_features')

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        keep_prob = 1.0
        with tf.variable_scope("loss"):
            '''
            #original
            if is_training.value:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            '''
            def do_dropout():
                return 0.9
            def do_not_dropout():
                return 1.0
            keep_prob = tf.cond(self.is_training, do_dropout, do_not_dropout)
            output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)


            self.logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            self.logits = tf.nn.bias_add(self.logits, output_bias, name='logits')

            self.probs = tf.nn.softmax(self.logits, axis=-1, name='probs')
            log_probs = tf.nn.log_softmax(self.logits, axis=-1)
            one_hot_label_ids = tf.one_hot(self.label_ids, depth=self.num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_label_ids * log_probs, axis=-1)

            self.loss = tf.reduce_mean(per_example_loss)
            self.pred_labels = tf.argmax(self.logits, 1, name='pred_labels')

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(one_hot_label_ids, 1), self.pred_labels)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")



    def similar_inference(self):
        a_output_layer = self.model.get_sequence_output()
        a_output_layer = tf.reduce_mean(a_output_layer, 1)
        a_output_layer = tf.identity(a_output_layer, 'sentence_features')

        b_output_layer = self.answer_model.get_sequence_output()
        b_output_layer = tf.reduce_mean(b_output_layer, 1)

        hidden_size = a_output_layer.shape[-1].value

        pointwise_w = tf.get_variable(
                        "pointwise_output_weights", [2, hidden_size * 2],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))

        pointwise_bias = tf.get_variable(
                        "pointwise_output_bias", [2], initializer=tf.zeros_initializer())
    
        with tf.variable_scope("loss"):
            def do_dropout():
                return 0.9
            def do_not_dropout():
                return 1.0
            keep_prob = tf.cond(self.is_training, do_dropout, do_not_dropout)
            a_output_layer = tf.nn.dropout(a_output_layer, keep_prob=keep_prob)
            b_output_layer = tf.nn.dropout(b_output_layer, keep_prob=keep_prob)

            one_hot_label_ids = tf.one_hot(self.label_ids, depth=2, dtype=tf.float32)

            '''
            #method1
            pointwise_output_layer = tf.concat([a_output_layer, b_output_layer], -1)
            #pointwise_output_layer = tf.concat([tf.abs(a_output_layer - b_output_layer), a_output_layer * b_output_layer], -1)        
            #pointwise_output_layer = tf.concat([a_output_layer, b_output_layer, a_output_layer * b_output_layer], -1)        
            #pointwise_output_layer = tf.concat([a_output_layer, b_output_layer, tf.abs(a_output_layer - b_output_layer)], -1)        
            #pointwise_output_layer = tf.concat([a_output_layer, b_output_layer, tf.abs(a_output_layer - b_output_layer), a_output_layer * b_output_layer], -1)        
            #pointwise_output_layer = tf.concat([a_output_layer + b_output_layer, tf.abs(a_output_layer - b_output_layer)], -1)
            self.pointwise_logits = tf.matmul(pointwise_output_layer, pointwise_w, transpose_b=True)
            self.pointwise_logits = tf.nn.bias_add(self.pointwise_logits, pointwise_bias, name='pair_logits')

            pointwise_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.pointwise_logits, labels=one_hot_label_ids)
            #pointwise_losses = tf.losses.mean_squared_error(labels=one_hot_label_ids, predictions=tf.nn.sigmoid(pointwise_logits))
            #pointwise_losses = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=label_ids, embeddings=pointwise_logits, margin=0.5)
            #pointwise_losses = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels=label_ids, embeddings=pointwise_output_layer, margin=0.5)

            self.pointwise_loss = tf.reduce_mean(pointwise_losses)
            self.pointwise_pred_labels = tf.argmax(self.pointwise_logits, 1, name='pointwise_pred_labels')
            self.pointwise_probs = tf.nn.softmax(self.pointwise_logits, axis=-1, name='probs')
            '''

            ''' 
            #method 2
            cosine_sims = cosine(a_output_layer, b_output_layer)
            pointwise_logits = tf.concat([tf.expand_dims(1 - cosine_sims, -1), tf.expand_dims(cosine_sims, -1)], 1)
            pointwise_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label_ids, dtype=tf.float32), logits=cosine_sims)

            self.pointwise_loss = tf.reduce_mean(pointwise_losses) 

            pointwise_pred_labels = tf.argmax(pointwise_logits, 1, name='pointwise_pred_labels')
            pointwise_probs = tf.nn.softmax(pointwise_logits, axis=-1, name='probs')
            '''

            
            #method 3
            cosine_sims = cosine(a_output_layer, b_output_layer)
            pointwise_logits = tf.concat([tf.expand_dims(1 - cosine_sims, -1), tf.expand_dims(cosine_sims, -1)], 1)
            pointwise_losses = tf.losses.mean_squared_error(labels=self.label_ids, predictions=cosine_sims)
            self.pointwise_loss = tf.reduce_mean(pointwise_losses) 

            self.pointwise_pred_labels = tf.argmax(pointwise_logits, 1, name='pointwise_pred_labels')
            self.pointwise_probs = tf.nn.softmax(pointwise_logits, axis=-1, name='probs')


        with tf.name_scope("accuracy"):
            pointwise_correct_pred = tf.equal(tf.argmax(one_hot_label_ids, 1), self.pointwise_pred_labels)
            self.pointwise_acc = tf.reduce_mean(tf.cast(pointwise_correct_pred, tf.float32), name='pointwise_acc')


    def jointed_pointwise_inference(self):
        #output_layer = self.model.get_pooled_output()                  
        output_layer = self.model.get_sequence_output()
        output_layer = tf.reduce_mean(output_layer, 1)
        output_layer = tf.identity(output_layer, 'sentence_features')

        hidden_size = output_layer.shape[-1].value

        with tf.variable_scope('loss'):
            keep_prob = 1.0
            def do_dropout():
                return 0.9
            def do_not_dropout():
                return 1.0
            keep_prob = tf.cond(self.is_training, do_dropout, do_not_dropout)
            output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)

            self.pointwise_logits = tf.layers.dense(output_layer, 2, activation=None, name='jointed_pointwise_logits')
            self.pointwise_probs = tf.nn.softmax(self.pointwise_logits, axis=-1, name='jointed_pointwise_probs')
    
            one_hot_label_ids = tf.one_hot(self.label_ids, depth=2, dtype=tf.float32)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.pointwise_logits, labels=one_hot_label_ids)

            self.pointwise_loss = tf.reduce_mean(losses)
            self.pointwise_pred_labels = tf.argmax(logits, 1, name='jointed_pointwise_pred_labels')

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(one_hot_label_ids, 1), self.pointwise_pred_labels)
            self.pointwise_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="jointed_pointwise_acc")



    def pairwise_similar_inference(self):
        output_layer = self.model.get_sequence_output()
        output_layer = tf.reduce_mean(output_layer, 1)
        output_layer = tf.identity(output_layer, 'sentence_features')

        answer_output_layer = self.answer_model.get_sequence_output()
        answer_output_layer = tf.reduce_mean(answer_output_layer, 1)
        
        negative_output_layer = self.negative_model.get_sequence_output()
        negative_output_layer = tf.reduce_mean(negative_output_layer, 1)


        score12 = cosine(output_layer, answer_output_layer)
        score13 = cosine(output_layer, negative_output_layer)

        losses = tf.maximum(0.0, tf.subtract(0.05, tf.subtract(score12, score13)))
        self.pairwise_loss = tf.reduce_sum(losses)
        self.pairwise_probs = tf.identity(score12, 'pairwise_probs')
        
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(0.0, losses)
            self.pairwise_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="pairwise_acc")


    def set_classify_train_op(self):
        self.train_op = get_train_op(self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps)


    def set_pointwise_similar_train_op(self):
        self.pointwise_train_op = get_train_op(self.pointwise_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps)


    def set_pairwise_similar_train_op(self):
        self.pairwise_train_op = get_train_op(self.pairwise_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps)



def convert_single_example(ex_index, example, label_map, max_seq_length,
                                                     tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #    tokens:     [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #    type_ids: 0         0    0        0        0         0             0 0         1    1    1    1     1 1
    # (b) For single sequences:
    #    tokens:     [CLS] the dog is hairy . [SEP]
    #    type_ids: 0         0     0     0    0         0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
    return feature

def get_and_save_label_map(label_list, output_dir):
    label_map = {}
    with codecs.open(output_dir + '/label_map', 'w', 'utf8') as fw:
        for (i, label) in enumerate(label_list):
            label_map[label] = i
            fw.write(str(i) + '\t' + label + '\n')
    return label_map

def file_based_convert_examples_to_features(
        examples, label_map, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer)
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def read_data_from_tfrecord(input_file, max_seq_length, batch_size, is_training, epochs, drop_remainder=False):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    data = tf.data.TFRecordDataset(input_file)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    if is_training:
        data = data.shuffle(buffer_size=50000)
        data = data.repeat(epochs)

    data = data.apply(
        tf.contrib.data.map_and_batch(
                            lambda record: _decode_record(record, name_to_features),
                            batch_size=batch_size,
                            drop_remainder=drop_remainder))
    return data

def read_data_from_tfrecord_v2(input_file, max_seq_length, batch_size, is_training, epochs, drop_remainder=False):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "answer_input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "answer_input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "answer_segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    data = tf.data.TFRecordDataset(input_file)
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    if is_training:
        data = data.shuffle(buffer_size=50000)
        data = data.repeat(epochs)

    #data = data.apply(
    #    tf.data.experimental.map_and_batch(
    #                        lambda record: _decode_record(record, name_to_features),
    #                        batch_size=batch_size,
    #                        drop_remainder=drop_remainder))

    data = data.map(lambda record: _decode_record(record, name_to_features))
    data = data.batch(batch_size)
    
    return data






def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()






def get_train_op(loss, learning_rate, num_train_steps, num_warmup_steps):


    #type 1, bert default train ops
    with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
        train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, False)

   
    #type 2, adam
    #train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #type3
    #learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
    #train_op = tf.contrib.layers.optimize_loss(loss, global_step=global_step, learning_rate=learning_rate, optimizer="Adam")

    return train_op


def self_attention(inputs):
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        u = tf.layers.dense(inputs, 768, use_bias=False, name='u')
        uv = tf.layers.dense(tf.nn.tanh(u), 1, use_bias=False, name='uv')
        print(uv)
        alphas = tf.nn.softmax(uv, name='alphas')
        print(alphas)
        outputs = tf.reduce_sum(alphas*inputs, 1)
        print(outputs)
        #exit()

    return outputs


def attention_with_mask(queries, keys, values, masks, name='attention'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        queries_ = tf.layers.dense(queries, 768, use_bias=False)
        queries_ = tf.expand_dims(queries_, 1)
        keys_ = tf.layers.dense(keys, 768, use_bias=False)
        attention_scores = tf.nn.tanh(queries_ + keys_)
        if masks is not None:
            masks = tf.expand_dims(masks, -1)
            adder = (1.0 - tf.cast(masks, tf.float32)) * -10000.0
            attention_scores += adder
        #print(attention_scores)
        #exit()
        attention_probs = tf.nn.softmax(tf.layers.dense(attention_scores, 1, use_bias=False), name='attention_probs')
        outputs = tf.reduce_sum(attention_probs*values, 1)
    return outputs

def attention(queries, keys, values, name='attention'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        queries_ = tf.layers.dense(queries, 768, use_bias=False)
        queries_ = tf.expand_dims(queries_, 1)
        keys_ = tf.layers.dense(keys, 768, use_bias=False)
        alphas = tf.nn.softmax(tf.layers.dense(tf.nn.tanh(queries_ + keys_), 1, use_bias=False), name='attention_probs')
        outputs = tf.reduce_sum(alphas*values, 1)
    return outputs


def cosine(q, a, name=None):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))

    pooled_mul_12 = tf.reduce_sum(tf.multiply(q, a), 1)
    score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2) + 1e-8, name=name)
    return score




# This function is not used by this file but is still used by the Colab and
# people who depend on it.


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features



def write_record_file(examples, output_dir, label_map, tokenizer, max_seq_length, name='train'):
    input_file = os.path.join(output_dir, "%s.tf_record"%name)
    tf.logging.info('*****write %s examples to file*****'%name)
    file_based_convert_examples_to_features(examples, label_map,
                                        max_seq_length, tokenizer, input_file)
    tf.logging.info("*****         finished           *****")




def get_tf_record_iterator(output_dir, max_seq_length, batch_size, num_steps, name='train', epochs=1, shuffle=False):
    input_file = os.path.join(output_dir, "%s.tf_record"%name)
    tf.logging.info("***** Running %s step *****"%name)
    tf.logging.info("    Batch size = %d", batch_size)
    tf.logging.info("    Num steps = %d", num_steps)
    data = read_data_from_tfrecord(input_file, FLAGS.max_seq_length, batch_size, is_training=shuffle, epochs=epochs)
    iterator = data.make_one_shot_iterator()
    batch_data_op = iterator.get_next() 
    return batch_data_op


def convert_single_pointwise_example(ex_index, example, max_seq_length,
                                                     tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
                input_ids=[0] * max_seq_length,
                input_mask=[0] * max_seq_length,
                segment_ids=[0] * max_seq_length,
                label_id=0,
                is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    if len(tokens_b) > max_seq_length - 2:
        tokens_b = tokens_b[0: (max_seq_length - 2)]

    tokens_a.insert(0, "[CLS]")
    tokens_a.append("[SEP]")
    segment_ids = [0] * len(tokens_a)
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    tokens_b.insert(0, '[CLS]')
    tokens_b.append('[SEP]')
    answer_segment_ids = [0] * len(tokens_b)
    answer_input_ids = tokenizer.convert_tokens_to_ids(tokens_b)
    answer_input_mask = [1] * len(answer_input_ids)
    while len(answer_input_ids) < max_seq_length:
        answer_input_ids.append(0) 
        answer_input_mask.append(0)                   
        answer_segment_ids.append(0)
    assert len(answer_input_ids) == max_seq_length
    assert len(answer_input_mask) == max_seq_length
    assert len(answer_segment_ids) == max_seq_length    

    
    #print(example.label)
    label_id = int(example.label)
    if ex_index < 5:
        tf.logging.info("*** Pair Sent Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens_a: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens_a]))
        tf.logging.info("tokens_b: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens_b]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("answer_input_ids: %s" % " ".join([str(x) for x in answer_input_ids]))
        tf.logging.info("answer_input_mask: %s" % " ".join([str(x) for x in answer_input_mask]))
        tf.logging.info("answer_segment_ids: %s" % " ".join([str(x) for x in answer_segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = PointwiseInputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            answer_input_ids=answer_input_ids,
            answer_input_mask=answer_input_mask,
            answer_segment_ids=answer_segment_ids,
            label_id=label_id,
            is_real_example=True)
    return feature


def file_based_convert_examples_to_similar_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_pointwise_example(ex_index, example, max_seq_length, tokenizer)
        
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features['answer_input_ids'] = create_int_feature(feature.answer_input_ids)
        features["answer_input_mask"] = create_int_feature(feature.answer_input_mask)
        features["answer_segment_ids"] = create_int_feature(feature.answer_segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
                                [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()




def write_similar_record_file(examples, output_dir, tokenizer, max_seq_length, name='train'):
    input_file = os.path.join(output_dir, "%s.tf_record"%name)
    tf.logging.info('*****write train examples to file*****')
    file_based_convert_examples_to_similar_features(examples, 
                                        max_seq_length, tokenizer, input_file)
    tf.logging.info("*****         finished           *****")


def get_tf_record_iterator_v2(output_dir, max_seq_length, batch_size, num_steps, name='train', epochs=1, shuffle=False):
    input_file = os.path.join(output_dir, "%s.tf_record"%name)                                                                     
    tf.logging.info("***** Running %s step *****"%name)                                                                                   
    tf.logging.info("    Batch size = %d", batch_size)                                                                                    
    tf.logging.info("    Num steps = %d", num_steps)                                                                                      
    data = read_data_from_tfrecord_v2(input_file, max_seq_length, batch_size, is_training=shuffle, epochs=epochs)
    iterator = data.make_one_shot_iterator()
    batch_data_op = iterator.get_next()     
    return batch_data_op




def get_data_from_iterator(sess, num_steps, data_op, task_type='classify'):
    data = []
    for i in range(num_steps):
        batch_data = sess.run(data_op)
        data.append((task_type, batch_data))
    return data

def random_negative_sampling(batch_data, batch_size, candidate_dataset, tokenizer):
    batch_neg_input_ids = []
    batch_neg_input_mask = []
    batch_neg_segment_ids = []
    batch_neg_answer_ids = []
    batch_label_ids = batch_data['label_ids']
    #batch_raw_texts = batch_data['raw_texts']

    for i in range(batch_size):
        #tokens = tokenizer.tokenize(batch_raw_texts[i].decode())
        #cur_bow = tfidf_dict.doc2bow(tokens)
        #cur_tfidf = tfidf_model[cur_bow]
        #sims = tfidf_sim_indices[cur_tfidf]
        #sorted_ids = np.argsort(-sims) 

        #cur_label_ids = batch_label_ids[i].decode('utf8').split(' ')
        cur_label_id = batch_label_ids[i]
        choice = np.random.choice(len(candidate_dataset), size=1, replace=False)[0]
        #while choice in cur_label_ids:
        while choice == cur_label_id:
            choice = np.random.choice(len(candidate_dataset), size=1, replace=False)[0]

        neg_cand_feature = candidate_dataset[choice]
        batch_neg_input_ids.append(neg_cand_feature.input_ids)
        batch_neg_input_mask.append(neg_cand_feature.input_mask)
        batch_neg_segment_ids.append(neg_cand_feature.segment_ids)
        batch_neg_answer_ids.append(neg_cand_feature.label_id)

    neg_features = {'neg_input_ids': batch_neg_input_ids,
                    'neg_input_mask': batch_neg_input_mask,
                    'neg_segment_ids': batch_neg_segment_ids,
                    'neg_answer_ids': batch_neg_answer_ids}

    return neg_features


def train_process(sess,
                  mtdnn_model,
                  chunk,
                  tokenizer,
                  candidate_dataset):
    task_type, batch_data = chunk
    if task_type == 'splited_pointwise_similar':
        feed_dict = {mtdnn_model.input_ids: batch_data['input_ids'],
                     mtdnn_model.input_mask: batch_data['input_mask'],
                     mtdnn_model.segment_ids: batch_data['segment_ids'],
                     mtdnn_model.answer_input_ids: batch_data['answer_input_ids'],
                     mtdnn_model.answer_input_mask: batch_data['answer_input_mask'],
                     mtdnn_model.answer_segment_ids: batch_data['answer_segment_ids'],
                     mtdnn_model.label_ids: batch_data['label_ids'],
                     mtdnn_model.is_training: True,
                     mtdnn_model.task_type_tensor: task_type}
        _, loss_value, acc_value  = sess.run([mtdnn_model.pointwise_train_op, mtdnn_model.pointwise_loss, mtdnn_model.pointwise_acc], feed_dict)
    elif task_type == 'pairwise_similar':
        neg_features = random_negative_sampling(batch_data, len(batch_data['input_ids']),
                                        candidate_dataset, tokenizer)

        feed_dict = {mtdnn_model.input_ids: batch_data['input_ids'],
                     mtdnn_model.input_mask: batch_data['input_mask'],
                     mtdnn_model.segment_ids: batch_data['segment_ids'],
                     mtdnn_model.answer_input_ids: batch_data['answer_input_ids'],
                     mtdnn_model.answer_input_mask: batch_data['answer_input_mask'],
                     mtdnn_model.answer_segment_ids: batch_data['answer_segment_ids'],
                     mtdnn_model.negative_input_ids: neg_features['neg_input_ids'],
                     mtdnn_model.negative_input_mask: neg_features['neg_input_mask'],
                     mtdnn_model.negative_segment_ids: neg_features['neg_segment_ids'],
                     mtdnn_model.is_training: True,
                     mtdnn_model.task_type_tensor: task_type} 
        _, loss_value, acc_value = sess.run([mtdnn_model.pairwise_train_op, mtdnn_model.pairwise_loss, mtdnn_model.pairwise_acc], feed_dict)

    else:                                                   
        feed_dict = {mtdnn_model.input_ids: batch_data['input_ids'],
                     mtdnn_model.input_mask: batch_data['input_mask'],
                     mtdnn_model.segment_ids: batch_data['segment_ids'],
                     mtdnn_model.label_ids: batch_data['label_ids'],
                     mtdnn_model.is_training: True,             
                     mtdnn_model.task_type_tensor: task_type}   
        _, loss_value, acc_value  = sess.run([mtdnn_model.train_op, mtdnn_model.loss, mtdnn_model.acc], feed_dict)
    return loss_value, acc_value





def eval_process(sess,
                 mtdnn_model,
                 num_eval_classify_steps,
                 num_eval_similar_steps):

    dev_multi_task_data = []
    dev_classify_batch_data_op = get_tf_record_iterator(FLAGS.output_dir,
                                                  FLAGS.max_seq_length,
                                                  FLAGS.eval_batch_size,
                                                  num_eval_classify_steps,
                                                  name='eval_classify')
    
    dev_classify_data = get_data_from_iterator(sess, num_eval_classify_steps, dev_classify_batch_data_op, 'classify')


    dev_similar_batch_data_op = get_tf_record_iterator_v2(FLAGS.output_dir,
                                                  FLAGS.max_seq_length,
                                                  FLAGS.eval_batch_size,
                                                  num_eval_similar_steps,
                                                  name='eval_similar') 


    dev_multi_task_data.extend(dev_classify_data)




    print('* eval results:*')
    real_eval_label_ids = []
    all_predictions = []
    all_loss = 0.0
    for dev_step, chunk in enumerate(dev_multi_task_data):
        task_type, batch_data = chunk
        if task_type == 'similar':
            feed_dict = {mtdnn_model.input_ids: batch_data['input_ids'],
                     mtdnn_model.input_mask: batch_data['input_mask'],
                     mtdnn_model.segment_ids: batch_data['segment_ids'],
                     mtdnn_model.answer_input_ids: batch_data['answer_input_ids'],
                     mtdnn_model.answer_input_mask: batch_data['answer_input_mask'],
                     mtdnn_model.answer_segment_ids: batch_data['answer_segment_ids'],
                     mtdnn_model.label_ids: batch_data['label_ids'],
                     mtdnn_model.is_training: False, 
                     mtdnn_model.task_type_tensor: task_type}
        else:
            feed_dict = {mtdnn_model.input_ids: batch_data['input_ids'],
                         mtdnn_model.input_mask: batch_data['input_mask'],
                         mtdnn_model.segment_ids: batch_data['segment_ids'],
                         mtdnn_model.label_ids: batch_data['label_ids'],
                         mtdnn_model.is_training: True,
                         mtdnn_model.task_type_tensor: task_type}
        loss_value, acc_value, cur_preds = sess.run([mtdnn_model.loss, mtdnn_model.acc, mtdnn_model.pred_labels], feed_dict)
        real_eval_label_ids.extend(batch_data['label_ids'])
        all_predictions.extend(cur_preds)
        all_loss += loss_value
    correct_predictions = float(sum(np.array(all_predictions) == np.array(real_eval_label_ids)))
    mean_acc_value = correct_predictions / float(len(real_eval_label_ids))
    mean_loss_value =  all_loss / float(num_eval_classify_steps)
    print('Accuracy: {:g}'.format(mean_acc_value))
    return mean_loss_value, mean_acc_value



def convert_candidates_to_features(candidates, max_seq_length, tokenizer):
    idx2cand, cand2idx = {}, {}
    for idx, cand in enumerate(candidates):
        idx2cand[idx] = cand
        cand2idx[cand] = idx

    dataset = []
    for idx, cand in enumerate(candidates):
        example = InputExample(guid=str(idx), text_a=cand, label=cand)
        feature = convert_single_example(idx, example, cand2idx, max_seq_length, tokenizer)
        dataset.append(feature)
    return np.array(dataset)

def get_candidates_from_file(file_name):
    candidates = []
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for item in fr:
            item = item.strip()
            candidates.append(item)
    return candidates


def main(_):
    rng = random.Random(FLAGS.random_seed) 
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
            "test": WeaverProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
                "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    processor.set_stopwords(os.path.join(FLAGS.data_dir, FLAGS.stopword_file))

    addon_processor = PointwiseProcessor()
    addon_processor.set_stopwords(os.path.join(FLAGS.data_dir, FLAGS.stopword_file))


    label_list = processor.get_labels(FLAGS.data_dir)
    label_map = get_and_save_label_map(label_list, FLAGS.output_dir)


    candidates = get_candidates_from_file(os.path.join(FLAGS.data_dir, FLAGS.candidates_file))


    shutil.copy(FLAGS.vocab_file, FLAGS.output_dir)
    shutil.copy(FLAGS.data_dir + '/labelcode', FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host))

    #train data
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    rng.shuffle(train_examples)
    rng.shuffle(train_examples)

   
    addon_train_examples = addon_processor.get_train_examples(FLAGS.data_dir)

    

    #num_train_steps = int((len(train_examples) * FLAGS.num_train_epochs - 1) / FLAGS.train_batch_size + 1)


    #num_train_classify_steps = int((len(train_examples) * FLAGS.num_train_epochs - 1) / FLAGS.train_batch_size + 1)
    #num_train_similar_steps = int((len(addon_train_examples) * FLAGS.num_train_epochs - 1) / FLAGS.train_batch_size + 1)
    #num_train_similar_steps = num_train_classify_steps
    #num_train_steps = num_train_classify_steps + num_train_similar_steps

    
    num_train_classify_steps = int((len(train_examples) - 1) / FLAGS.train_batch_size + 1)
    num_train_similar_steps = int((len(addon_train_examples) - 1) / FLAGS.train_batch_size + 1)
    num_train_steps = (num_train_classify_steps + num_train_similar_steps) * FLAGS.num_train_epochs


    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    num_labels = len(label_list)

    


    #eval data
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_eval_classify_steps = int((len(eval_examples) - 1) / FLAGS.eval_batch_size) + 1

    addon_eval_examples = addon_processor.get_dev_examples(FLAGS.data_dir)
    num_eval_similar_steps = int((len(addon_eval_examples) - 1) / FLAGS.eval_batch_size) + 1



    candidate_dataset = convert_candidates_to_features(candidates, FLAGS.max_seq_length, tokenizer)
    


    mtdnn_model = MTDNN(bert_config, 
                        num_labels,
                        FLAGS.use_tpu,
                        FLAGS.max_seq_length,
                        FLAGS.init_checkpoint,
                        FLAGS.learning_rate,
                        num_train_steps,
                        num_warmup_steps)




    write_record_file(train_examples, FLAGS.output_dir, label_map, tokenizer, FLAGS.max_seq_length, name='train_classify')
    
    write_similar_record_file(addon_train_examples, FLAGS.output_dir, tokenizer, FLAGS.max_seq_length, name='train_similar')

    #similar_label_map = {"1": 1, "0": 0}
    #write_record_file(addon_train_examples, FLAGS.output_dir, similar_label_map, tokenizer, FLAGS.max_seq_length, name='train_similar')



    write_record_file(eval_examples, FLAGS.output_dir, label_map, tokenizer, FLAGS.max_seq_length, name='eval_classify')


    write_similar_record_file(addon_eval_examples, FLAGS.output_dir, tokenizer, FLAGS.max_seq_length, name='eval_similar')


    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=sess_config)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        train_multi_task_data = []

        train_classify_batch_data_op = get_tf_record_iterator(FLAGS.output_dir,
                                                     FLAGS.max_seq_length, 
                                                     FLAGS.train_batch_size, 
                                                     num_train_classify_steps,
                                                     name='train_classify',
                                                     epochs=1,
                                                     shuffle=True)
        train_classify_data = get_data_from_iterator(sess, num_train_classify_steps, train_classify_batch_data_op, 'classify')
        
        
        train_similar_batch_data_op = get_tf_record_iterator_v2(FLAGS.output_dir,
        #train_similar_batch_data_op = get_tf_record_iterator(FLAGS.output_dir,
                                                     FLAGS.max_seq_length,
                                                     FLAGS.train_batch_size,
                                                     num_train_similar_steps,
                                                     name='train_similar',
                                                     epochs=1,
                                                     shuffle=True)


        train_similar_data = get_data_from_iterator(sess, num_train_similar_steps, train_similar_batch_data_op, 'splited_pointwise_similar')
        #train_similar_data = get_data_from_iterator(sess, num_train_similar_steps, train_similar_batch_data_op, 'jointed_pointwise_similar')
        #train_similar_data = get_data_from_iterator(sess, num_train_similar_steps, train_similar_batch_data_op, 'pairwise_similar')

        train_multi_task_data.extend(train_classify_data)
        train_multi_task_data.extend(train_similar_data)
        
        
        best_acc_value = 0.0
        eval_acc_value = 0.0
        for epoch in range(int(FLAGS.num_train_epochs)):
            rng.shuffle(train_multi_task_data)
            for cur_train_step, chunk in enumerate(train_multi_task_data):
                train_step = cur_train_step + len(train_multi_task_data) * epoch
                loss_value, acc_value = train_process(sess, mtdnn_model, chunk, tokenizer, candidate_dataset)
            
                time_str = datetime.datetime.now().isoformat()
                #print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, train_step, loss_value, acc_value))
                print('{}: step {},\tloss {:g},\tacc {:g}'.format(time_str, train_step, loss_value, 
                                                                    acc_value))

                if (train_step % 1000 == 0 and train_step != 0) or \
                    (train_step == num_train_steps - 1):
                    eval_loss_value, eval_acc_value = eval_process(sess,
                                                               mtdnn_model,
                                                               num_eval_classify_steps,
                                                               num_eval_similar_steps)

                if eval_acc_value > best_acc_value:
                    best_acc_value = eval_acc_value
                    saver.save(sess, FLAGS.output_dir + '/checkpoints/model', global_step=train_step)

            if not FLAGS.do_eval:
                saver.save(sess, FLAGS.output_dir + '/checkpoints/model', global_step=train_step)
            


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
