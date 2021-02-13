#!/bin/bash

# as defined in experiments.conf
config_name=$1
#either empty or "no_wb."
prefix=$2



echo 'Run the probe training with all ablations'

train_data=data/${config_name}/train/
val_data=data/${config_name}/dev/SPAN_1.h5

if [ -f "$val_data" ]; then
    val_arg="--val_data $val_data"
else
    echo "No Validation Set available"
    val_arg=
fi

test_data=data/${config_name}/test/SPAN_1.h5
exp_name=data/${config_name}/results_all_features

python train_probe.py --train_data $train_data $val_arg --test_data $test_data --exp_name $exp_name

exp_name=data/${config_name}/results_ablate_boundary
option='--ablate_boundary'
python train_probe.py --train_data $train_data $val_arg --test_data $test_data --exp_name ${exp_name} ${option}

exp_name=data/${config_name}/results_ablate_attention
option='--ablate_attention'
python train_probe.py --train_data $train_data $val_arg --test_data $test_data --exp_name ${exp_name} ${option}

exp_name=data/${config_name}/results_ablate_span_width
option='--ablate_span_width'
python train_probe.py --train_data $train_data $val_arg --test_data $test_data --exp_name ${exp_name} ${option}

exp_name=data/${config_name}/results_random
option='--random'
python train_probe.py --train_data $train_data $val_arg --test_data $test_data --exp_name ${exp_name} ${option}


echo 'Outputs the error pairs for qualitative analysis'
model_name=data/${config_name}/results_all_features.h5
test_data=data/${config_name}/test/SPAN_PR
exp_name=data/${config_name}/SPAN_PR_AN
python predict_joshi.py --model ${model_name} --test_data ${test_data} --exp_name ${exp_name}
python pred/pred_analyse.py ${exp_name}.jsonlines > ${exp_name}.txt

