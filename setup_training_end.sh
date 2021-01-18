# I don't remember if get_char_vocab.py and minimize really work with python 3. If not comment out lines 6-7
# and execute this script in conda with python 2 activated. Then, comment out 4-5 and remove the comment from 6-7,
# activate python 3 and run....

python minimize.py
python get_char_vocab.py

python filter_embeddings.py glove.840B.300d.txt train.english.jsonlines dev.english.jsonlines
python cache_elmo.py train.english.jsonlines dev.english.jsonlines
