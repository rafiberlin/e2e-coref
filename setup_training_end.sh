# I don't remember if get_char_vocab.py and minimize really work with python 3. If not comment out lines 6-7
# and execute this script in conda with python 2 activated. Then, comment out 4-5 and remove the comment from 6-7,
# activate python 3 and run....

# creates all *.english.ontonotes.jsonlines files
python minimize.py
# creates all *.english.twiconv.jsonlines files
python minimize_twiconv.py

#merge twiconv and ontonotes jsonlines together as they have now the same structure
cat train.english.*.jsonlines > train.english.jsonlines
cat dev.english.*.jsonlines > dev.english.jsonlines
cat test.english.*.jsonlines > test.english.jsonlines

#python get_char_vocab.py

python filter_embeddings.py glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines
python cache_elmo.py train.english.jsonlines dev.english.jsonlines
#Comment out line 9 and put back line 11 to test ernie embeddings... Variable 'lm_path = ernie_cache.hdf5' in experiments.conf must be set.
#python cache_ernie.py train.english.jsonlines dev.english.jsonlines

#merge twiconv and ontonotes conll together. Even if they dont have the same structure, the scorer does not care (only checks last column of documents for coref clusters...)
cat train.english.v*_gold_conll > train.english.gold_conll
cat dev.english.v*_gold_conll > dev.english.gold_conll
cat test.english.v*_gold_conll > test.english.gold_conll