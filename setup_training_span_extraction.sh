#!/bin/bash

echo 'Creates the positive and negative examples for all relevants sets...'
python negative_positive_samples.py train.english.jsonlines NP.train.english.jsonlines
python negative_positive_samples.py test.english.jsonlines NP.test.english.jsonlines
python negative_positive_samples.py dev.english.jsonlines NP.dev.english.jsonlines
python negative_positive_samples.py no_wb.train.english.jsonlines NP.no_wb.train.english.jsonlines
python negative_positive_samples.py no_wb.test.english.jsonlines NP.no_wb.test.english.jsonlines
python negative_positive_samples.py no_wb.dev.english.jsonlines NP.no_wb.dev.english.jsonlines
python negative_positive_samples.py train.english.ontonotes.jsonlines NP.train.english.ontonotes.jsonlines
python negative_positive_samples.py test.english.ontonotes.jsonlines NP.test.english.ontonotes.jsonlines
python negative_positive_samples.py dev.english.ontonotes.jsonlines NP.dev.english.ontonotes.jsonlines

