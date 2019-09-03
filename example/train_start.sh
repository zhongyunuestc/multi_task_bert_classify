###################################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月12日 星期二 16时19分57秒
#=============================================================
#!/bin/bash
source activate tensorflow_new_3.6

export BERT_BASE_DIR=/root/zhaomeng/google-BERT/chinese_L-12_H-768_A-12
#export BERT_BASE_DIR=/root/zhaomeng/baidu_ERNIE/pad_to_tf/checkpoints

python bert_train.py --task_name=test \
                     --output_dir=./runs/search_person \
                     --data_dir=../data \
                     --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
                     --bert_config_file=$BERT_BASE_DIR/bert_config.json \
                     --vocab_file=$BERT_BASE_DIR/vocab.txt \
                     --max_seq_length=32  \
                     --do_train=true \
                     --stopword_file=stopword_data/stop_symbol \
                     --num_train_epochs=20;
python ckpt_to_pb.py
python extract_features.py
