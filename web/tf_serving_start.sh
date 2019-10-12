#coding:utf-8
###################################################
# File Name: tf_serving_start.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年04月30日 星期二 13时33分47秒
#=============================================================

export REST_API_PORT=17122
export MODEL_DIR=$PWD/../example/runs/search_person/checkpoints/
export MODEL_NAME=default

source activate tensorflow_new_3.6
nohup tensorflow_model_server --rest_api_port=$REST_API_PORT \
                              --model_name=$MODEL_NAME \
                              --model_base_path=$MODEL_DIR \
      >output.file 2>&1 &

nohup python tf_serving_person_ranking_http.py> output.file 2>&1 &

python sender.py
