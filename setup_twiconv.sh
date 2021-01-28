#!/bin/bash

# create the data split
python3 split_train_dev_test.py conll/

# Concatenates all documents in 1 file per split
cat train/*conll > train.english.v9_gold_conll
cat test/*conll > test.english.v9_gold_conll
cat dev/*conll > dev.english.v9_gold_conll

# Removes unnecessary columns
cp -f train.english.v9_gold_conll old.train.english.v9_gold_conll
cut -f1-11 old.train.english.v9_gold_conll > train.english.v9_gold_conll
rm old.train.english.v9_gold_conll
cp -f test.english.v9_gold_conll old.test.english.v9_gold_conll
cut -f1-11 old.test.english.v9_gold_conll > test.english.v9_gold_conll
rm old.test.english.v9_gold_conll
cp -f dev.english.v9_gold_conll old.dev.english.v9_gold_conll
cut -f1-11 old.dev.english.v9_gold_conll > dev.english.v9_gold_conll
rm old.dev.english.v9_gold_conll