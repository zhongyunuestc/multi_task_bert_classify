#coding:utf-8
###################################################
# File Name: release.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年08月13日 星期二 13时26分02秒
#=============================================================
mkdir dist
cd dist
mkdir data
cp -r ../data/stopword_data data/


mkdir example
cp -r ../example/runs example/
cp -r ../example/__init__.py example/

cp -r ../web .
cp -r ../common .
cp -r ../setting.py .
cp -r ../preprocess .


