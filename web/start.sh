#coding:utf-8
###################################################
# File Name: start.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月13日 星期三 11时30分49秒
#=============================================================
source activate tensorflow_new_3.6
#nohup python person_rankning_http.py>/dev/null 2>&1 &
nohup python person_ranking_http.py>output.file 2>&1 &
