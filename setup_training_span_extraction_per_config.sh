#!/bin/bash

# as defined in experiments.conf
config_name=$1
prefix=$2
#either empty or "no_wb."

#it creates under ./data/{config_name}/ the train, dev, test directories
#python span_util.py $config_name train
#python span_util.py $config_name dev
#python span_util.py $config_name test


#extract the embeddings for all splits
#for split in train
#do
#    python span_util.py $config_name $split
#    python extract_span.py $config_name NP.${prefix}${split}.english.jsonlines data/${config_name}/${split} SPAN.
#    python extract_span_predict.py $config_name NP.${prefix}${split}.english.jsonlines data/${config_name}/${split}/SPAN_PR.
#done

train_data=data/${config_name}/train/
val_data=data/${config_name}/dev/SPAN_1.h5
test_data=data/${config_name}/test/SPAN_1.h5
exp_name=data/${config_name}/results_all_features

python train_probe.py --train_data $train_data --val_data $val_data --test_data $test_data --exp_name $exp_name

exp_name=data/${config_name}/results_ablate_boundary
option='--ablate_boundary'
python train_probe.py --train_data $train_data --val_data $val_data --test_data $test_data --exp_name ${exp_name} ${option}

exp_name=data/${config_name}/results_ablate_attention
option='--ablate_attention'
python train_probe.py --train_data $train_data --val_data $val_data --test_data $test_data --exp_name ${exp_name} ${option}

exp_name=data/${config_name}/results_ablate_span_width
option='--ablate_span_width'
python train_probe.py --train_data $train_data --val_data $val_data --test_data $test_data --exp_name ${exp_name} ${option}

exp_name=data/${config_name}/results_random
option='--random'
python train_probe.py --train_data $train_data --val_data $val_data --test_data $test_data --exp_name ${exp_name} ${option}


#python extract_span.py $config_name NP.${prefix}${split}.english.jsonlines data/$config_name/train SPAN.
#python extract_span.py $config_name NP.no_wb.test.english.jsonlines data/$config_name/test SPAN.
#python extract_span.py $config_name NP.no_wb.dev.english.jsonlines data/$config_name/dev SPAN.
