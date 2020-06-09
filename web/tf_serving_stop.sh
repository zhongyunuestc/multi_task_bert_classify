#coding:utf-8
###################################################
# File Name: tf_serving_stop.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年09月05日 星期四 14时54分25秒
#=============================================================
CLIENT_NAME=tf_serving_person_ranking_http.py
ps -ef | grep "python $CLIENT_NAME" | grep -v 'grep'
ps -ef | grep "python $CLIENT_NAME" | grep -v 'grep' | awk '{print $2}' | xargs -n1 kill -9


SERVER_PORT=17122
ps -ef | grep "tensorflow_model_server --rest_api_port=$SERVER_PORT" | grep -v 'grep'
ps -ef | grep "tensorflow_model_server --rest_api_port=$SERVER_PORT" | grep -v 'grep' | awk '{print $2}' | xargs -n1 kill -9
