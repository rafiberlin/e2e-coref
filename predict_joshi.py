import h5py
import numpy as np
import argparse
import json

from keras.models import load_model
from keras_self_attention import SeqWeightedAttention

# python3 predict_joshi.py --model --test_data --exp_name

def get_args():
    parser = argparse.ArgumentParser(description='Run probing experiment for c2f-coref with BERT embeddings')

    parser.add_argument('--model', type=str, default=None)      # with extension
    parser.add_argument('--test_data', type=str, default=None)  # without extension
    parser.add_argument('--exp_name', type=str, default=None)   # without extension
    args = parser.parse_args()
    return args
#Prerequisite: train_probe.py was executed to get the model, extract_span_predict.py was executed to get the test data (.h5 and jsonline file)
#python3  predict_joshi,py --model ./data/ontonotes/e2e_c2f_probe.h5 --test_data ./data/ontonotes/test/span_pred_test --exp_name ./data/ontonotes/test/span_pred_test_analyse
if __name__ == "__main__":
    args = get_args()

    # test data name comes without the extention. same name for two data files
    test_data_json = args.test_data + ".jsonlines"
    test_data_h5 = args.test_data + ".h5"
    exp_name = args.exp_name + ".jsonlines"
    model = load_model(str(args.model), custom_objects=SeqWeightedAttention.get_custom_objects())

    # span representations: [parent_child_emb, men1_start, men1_end, men2_start,
    # men2_end, doc_key_arr, mention_dist, gold_label]
    with h5py.File(test_data_h5, 'r') as f:
        test_data = f.get('span_representations').value
        x_test = test_data[:, :-2]
        y_test = test_data[:, -1].astype(int)
        doc_key_arr = test_data[:, -2].astype(float)

    test_predict = (np.asarray(model.predict(x_test))).round()

    with open(exp_name, 'w') as output_file:
        with open(test_data_json, 'r') as input_file:
            for line in input_file.readlines():
                pred = []
                example = json.loads(line)
                # get the dockey of this example
                doc_key = example['doc_key']
                idxs = np.where(np.isclose(doc_key_arr, doc_key))
                idxs = list(idxs)
                for idx in idxs[0]:
                    pred.append(int(test_predict[int(idx)][0]))
                example['pred'] = list(pred)


                output_file.write(json.dumps(example))
                output_file.write("\n")


