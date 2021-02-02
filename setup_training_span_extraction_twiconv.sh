#!/bin/bash

# as defined in experiments.conf
config_name=twiconv_eval
#either empty or "no_wb."
prefix="no_wb."


echo 'extract the embeddings for all splits'
for split in train test
do
    python span_util.py $config_name $split
    echo $config_name NP.${prefix}${split}.english.jsonlines data/${config_name}/${split} SPAN
    python extract_span.py $config_name NP.${prefix}${split}.english.jsonlines data/${config_name}/${split} SPAN
    python extract_span_predict.py $config_name NP.${prefix}${split}.english.jsonlines data/${config_name}/${split}/SPAN_PR
done

split=dev
python negative_positive_samples.py dev.english.ontonotes.jsonlines NP.dev.english.ontonotes.jsonlines
python span_util.py $config_name $split
python extract_span.py $config_name NP.${split}.english.ontonotes.jsonlines data/${config_name}/${split} SPAN
python extract_span_predict.py $config_name NP.${split}.english.ontonotes.jsonlines data/${config_name}/${split}/SPAN_PR

echo 'Run the probe training with all ablations'

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


echo 'Outputs the error pairs for qualitative analysis'
model_name=data/${config_name}/results_all_features.h5
test_data=data/${config_name}/test/SPAN_PR
exp_name=data/${config_name}/SPAN_PR_AN
python predict_joshi.py --model ${model_name} --test_data ${test_data} --exp_name ${exp_name}
python pred/pred_analyse.py ${exp_name}.jsonlines > ${exp_name}.txt

