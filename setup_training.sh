#!/bin/bash

dlx() {
  wget $1/$2
  tar -xvzf $2
  rm $2
}

conll_url=http://conll.cemantix.org/2012/download
dlx $conll_url conll-2012-train.v4.tar.gz
dlx $conll_url conll-2012-development.v4.tar.gz
dlx $conll_url/test conll-2012-test-key.tar.gz
dlx $conll_url/test conll-2012-test-official.v9.tar.gz

dlx $conll_url conll-2012-scripts.v3.tar.gz

dlx http://conll.cemantix.org/download reference-coreference-scorers.v8.01.tar.gz
mv reference-coreference-scorers conll-2012/scorer
#change me!
ontonotes_path=./ontonotes-5.0
bash conll-2012/v3/scripts/skeleton2conll.sh -D $ontonotes_path/data/files/data conll-2012

function compile_partition() {
    rm -f $2.$5.$3$4
    cat conll-2012/$3/data/$1/data/$5/annotations/*/*/*/*.$3$4 >> $2.$5.$3$4
}

function compile_language() {
    compile_partition development dev v4 _gold_conll $1
    compile_partition train train v4 _gold_conll $1
    compile_partition test test v4 _gold_conll $1
}

# Remove all conll files for wb, as we create our own wb docs from twiconv
# But it seems to be a bad idea => Training only on Twiconv leads to poor results...
#find conll-2012/v4/data/train/data/english/annotations/wb/ -name '*.v4_gold_conll' -delete
#find conll-2012/v4/data/test/data/english/annotations/wb/ -name '*.v4_gold_conll' -delete
#find conll-2012/v4/data/development/data/english/annotations/wb/ -name '*.v4_gold_conll' -delete


compile_language english
#compile_language chinese
#compile_language arabic
