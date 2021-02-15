# Higher-order Coreference Resolution with Coarse-to-fine Inference

## Introduction
This repository contains the code for replicating results from

* [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392)
* [Kenton Lee](http://kentonl.com/), [Luheng He](https://homes.cs.washington.edu/~luheng), and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In NAACL 2018

## Dataset preparation : Ontonotes 5.0 and TwiConv

* The Ontonotes dataset must be available. 

* TwiConv
    * Run the scripts from `https://github.com/berfingit/TwiConv` to obtain the twitter dataset. Copy the  produced complete conll/ directory at the root of this directory.
    
    * The data is then transformed with `setup_twiconv.sh` in the section `Getting Started` as follow:
    
        * First creating a data split by executing `python3 split_train_dev_test.py conll/`. `split_train_dev_test.py` which is a modified version of `https://github.com/verosol/e2e-coref-to-Twitter/blob/master/split_train_test.py` creating a development / validation split.
        
        * Merging the contents of the folders into train and test files via
        
        ```bash
        cat train/*conll > train.english.v9_gold_conll
        cat test/*conll > test.english.v9_gold_conll
        cat dev/*conll > dev.english.v9_gold_conll
        ```
    
        * Removing all columns after column 10 by exeuting the following commands (only keep the all columns up to the column with coreference annotation).  At the end of the process, the v9_gold_conll should be at the 
        root of this github repository (at the same level as the setup_all.sh file for example).
    
        ```bash
        cp -f train.english.v9_gold_conll old.train.english.v9_gold_conll
        cut -f1-11 old.train.english.v9_gold_conll > train.english.v9_gold_conll
        rm old.train.english.v9_gold_conll
        cp -f test.english.v9_gold_conll old.test.english.v9_gold_conll
        cut -f1-11 old.test.english.v9_gold_conll > test.english.v9_gold_conll
        rm old.test.english.v9_gold_conll
        cp -f dev.english.v9_gold_conll old.dev.english.v9_gold_conll
        cut -f1-11 old.dev.english.v9_gold_conll > dev.english.v9_gold_conll
        rm old.dev.english.v9_gold_conll
        ```

## Getting Started

* Install python 3 requirements: `pip install -r requirements.txt` (You will need a second conda environment 
with Python 2 to run the scripts building the CoNLL files in `setup_training.sh`)
* Download pretrained models at https://drive.google.com/file/d/1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi
  * Move the downloaded file to the root of the repo and extract: `tar -xzvf e2e-coref.tgz`
* Download GloVe embeddings and build custom kernels by running `setup_all.sh` (in the conda environment with Python 3).
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* To setup the prerequisites to train your own models, run `setup_training.sh` in conda with Python 2 activated.
  * This assumes access to OntoNotes 5.0. Please edit the `ontonotes_path` variable in `setup_training.sh`.
  * This script downloads Ontonotes annotations and scripts creating the CoNLL files needed.
  * If you are unlucky, you will not be able to download the files per script (computer flagged by the data provider as a bot). You will need to download and unzip the files manually.
  * To be able to train the model Twiconv without WB genre but with other genres conbination, you can remove the 
  conll-2012/v4/data/wb/train/wb , conll-2012/v4/data/wb/test/wb, conll-2012/v4/data/wb/development/wb before re-executing
  `setup_training.sh`. The resulting train.english.v4_gold_conll, test.english.v4_gold_conll, dev.english.v4_gold_conll
  will not contain any document of the WB genre. Then proceed with next steps without changes. You might want to rename 
  the produced files though, to be able to work with a dataset with and without WB documents. We assume that all files without the 
  WB genre have the prefix `no_wb.` to run the probe experiments for example.  
* To transform the Twiconv dataset as needed in this project, run `setup_twiconv.sh` in conda with Python 3 activated.
* To install the last prerequisites to train your own models, run `setup_training_end.sh` in conda with Python 3 activated.
  * This will transform all CoNLL files (for TwiConv and Ontonotes) into jsonlines and merge all ConLL files together,
  * Ernie embedding can be used but the model must be retrained. See comments in `setup_training_end.sh`


## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best` or `twiconv_allspoken`
* Training: `python train.py <experiment>`
* Please note: if for some reason, your training were to be interrupted, replace the variable `max_f1` in experiments.conf by the highest F1 average observed on the dev set (it's being printed at each evaluation of the validation (dev) set)
* Results are stored in the `logs` directory, in a directory of the same name as the experiment and can be viewed via TensorBoard.
* Evaluation: `python evaluate.py <experiment>` , e.g. `python evaluate.py twiconv_allspoken_eval`

## Running probe experiments

* Run `setup_training_span_extraction.sh` only once to create the files with positive and negative examples.
* Execute `setup_training_span_extraction_per_config.sh <experiment_eval> <prefix> <output_dim> <batch_size>` to run a probe, where `<experiment_eval>`
  is an evaluation configuration (lm_path = "" in experiments.conf) and `<prefix>` is the prefix of the dataset to use,
  `<output_dim>` is the output dimension of the probe network `<batch_size>` the number of elemts to train on per iteration.
  The results will be stored under data/<experiment_eval>
  - Examples:
    * bash setup_training_span_extraction_per_config.sh twiconv_wb_eval '' 450 256
    * bash setup_training_span_extraction_per_config.sh twiconv_tc_eval 'no_wb.' 450 256

TODO: We can define some distance bins in pred/pred_analyse.py to investigate the effect of longer distance between 2 mentions.
## Demo Instructions

* Command-line demo: `python demo.py final`
* To run the demo with other experiments, replace `final` with your configuration name.

## Batched Prediction Instructions

* Create a file where each line is in the following json format (make sure to strip the newlines so each line is well-formed json):
```
{
  "clusters": [],
  "doc_key": "nw",
  "sentences": [["This", "is", "the", "first", "sentence", "."], ["This", "is", "the", "second", "."]],
  "speakers": [["spk1", "spk1", "spk1", "spk1", "spk1", "spk1"], ["spk2", "spk2", "spk2", "spk2", "spk2"]]
}
```
  * `clusters` should be left empty and is only used for evaluation purposes.
  * `doc_key` indicates the genre, which can be one of the following: `"bc", "bn", "mz", "nw", "pt", "tc", "wb"`
  * `speakers` indicates the speaker of each word. These can be all empty strings if there is only one known speaker.
* Run `python predict.py <experiment> <input_file> <output_file>`, which outputs the input jsonlines with predicted clusters.

## Other Quirks

* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* The training runs indefinitely and needs to be terminated manually. The model generally converges at about 400k steps.
