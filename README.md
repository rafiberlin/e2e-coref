# Higher-order Coreference Resolution with Coarse-to-fine Inference

## Introduction
This repository contains the code for replicating results from

* [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392)
* [Kenton Lee](http://kentonl.com/), [Luheng He](https://homes.cs.washington.edu/~luheng), and [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz)
* In NAACL 2018

## Getting Started

* Install python 3 requirements: `pip install -r requirements.txt` (You will need a second conda environment 
with Python 2 to run the scripts building the CoNLL files in `setup_training.sh`)
* Download pretrained models at https://drive.google.com/file/d/1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi
  * Move the downloaded file to the root of the repo and extract: `tar -xzvf e2e-coref.tgz`
* Download GloVe embeddings and build custom kernels by running `setup_all.sh` (in the conda environment with Python 3).
  * There are 3 platform-dependent ways to build custom TensorFlow kernels. Please comment/uncomment the appropriate lines in the script.
* To setup the prerequisites to train your own models, run `setup_training.sh` in conda with Python 2 activated.
  * This assumes access to OntoNotes 5.0. Please edit the `ontonotes_path` variable.
* To install the last prerequisites to train your own models, run `setup_training_end.sh` in conda with Python 3 activated.
  * Ernie embedding can be used but the model must be retrained. See comments in `setup_training_end.sh`

## Converting TwiConv

* Run the scripts from `https://github.com/berfingit/TwiConv` to obtain the twitter dataset
* Download the `split_train_test.py` script from `https://github.com/verosol/e2e-coref-to-Twitter/blob/master/split_train_test.py`
* Run `python3 split_train_test.py conll/`

Merge the contents of the folders into train and test files via

```bash
cat train/*conll > train.english.v9_gold_conll
cat test/*conll > test.english.v9_gold_conll
```

* Convert the files into jsonlines using

```
python3 minimize_twiconv.py
```

This script assumes the train and test v9 gold conll files are in the current working directory.

* `minimize_twicon` also automatically splits the train set into 80% train and 20% dev sets

## Training Instructions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* Training: `python train.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* Evaluation: `python evaluate.py <experiment>`

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
