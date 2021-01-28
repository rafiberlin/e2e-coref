import os
import sys
import math
import random


def main():

    if not os.path.isdir('train'):
        os.mkdir('train')
    if not os.path.isdir('dev'):
        os.mkdir('dev')
    if not os.path.isdir('test'):
        os.mkdir('test')
    parent_dir = sys.argv[1]
    parts = []
    test_split_parts = [6, 7, 14, 29, 98, 138, 144, 35, 64, 81, 91, 92, 95, 96, 117, 137, 140, 150, 153, 172]

    for conll in os.listdir(parent_dir):

        with open(os.path.join(parent_dir, conll), "r", encoding="utf-8") as c:
            conll2 = list(c)
            temp = []
            for e in conll2:
                temp.append(e)
                i = temp[0].split('; part')[-1].strip()
                int_i = int(i)
                if int_i not in test_split_parts:
                    parts.append(int_i)
                break

    cutoff = math.ceil(len(parts) * 0.1)
    dev_parts = random.sample(parts, cutoff)

    print("Random Dev Part Ids to use in split_train_dev.py:", dev_parts)

# NOT IN USE!
# The list in test split parts is the list of document being used for the evaluation
# in https://github.com/verosol/e2e-coref-to-Twitter
# this scripts just randomly selects a list of ids (10% of the original training documents) for evaluation
# The ids in the output are then copied in split_train_test.py to init the variable
# dev_split_parts
if __name__ == '__main__':
    main()
