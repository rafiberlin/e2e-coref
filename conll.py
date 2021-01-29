from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import tempfile
import subprocess
import operator
import collections

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

def get_doc_key(doc_id, part):
  return "{}_{}".format(doc_id, int(part))

def output_conll_with_filter(input_file, output_file, predictions, genres):
  prediction_map = {}
  for doc_key, clusters in predictions.items():
    start_map = collections.defaultdict(list)
    end_map = collections.defaultdict(list)
    word_map = collections.defaultdict(list)
    for cluster_id, mentions in enumerate(clusters):
      for start, end in mentions:
        if start == end:
          word_map[start].append(cluster_id)
        else:
          start_map[start].append((cluster_id, end))
          end_map[end].append((cluster_id, start))
    for k,v in start_map.items():
      start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
    for k,v in end_map.items():
      end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
    prediction_map[doc_key] = (start_map, end_map, word_map)

  word_index = 0
  allowed_genre = False
  doc_started = False
  for line in input_file.readlines():
    row = line.split()
    if len(row) == 0 and doc_started:
      output_file.write("\n")
    elif row and row[0].startswith("#"):
      begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
      if begin_match:
        doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
        genre = doc_key[:2]
        doc_started = True
        allowed_genre = genre in genres.keys()
        if allowed_genre:
          start_map, end_map, word_map = prediction_map[doc_key]
          word_index = 0
          output_file.write(line)
      else:
        doc_started = False
        if allowed_genre:
          output_file.write(line)
          allowed_genre = False
    else:
      if allowed_genre:
        assert get_doc_key(row[0], row[1]) == doc_key
        coref_list = []
        if word_index in end_map:
          for cluster_id in end_map[word_index]:
            coref_list.append("{})".format(cluster_id))
        if word_index in word_map:
          for cluster_id in word_map[word_index]:
            coref_list.append("({})".format(cluster_id))
        if word_index in start_map:
          for cluster_id in start_map[word_index]:
            coref_list.append("({}".format(cluster_id))

        if len(coref_list) == 0:
          row[-1] = "-"
        else:
          row[-1] = "|".join(coref_list)

        output_file.write("   ".join(row))
        output_file.write("\n")
        word_index += 1

def output_conll(input_file, output_file, predictions):
  prediction_map = {}
  for doc_key, clusters in predictions.items():
    start_map = collections.defaultdict(list)
    end_map = collections.defaultdict(list)
    word_map = collections.defaultdict(list)
    for cluster_id, mentions in enumerate(clusters):
      for start, end in mentions:
        if start == end:
          word_map[start].append(cluster_id)
        else:
          start_map[start].append((cluster_id, end))
          end_map[end].append((cluster_id, start))
    for k,v in start_map.items():
      start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
    for k,v in end_map.items():
      end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
    prediction_map[doc_key] = (start_map, end_map, word_map)

  word_index = 0
  for line in input_file.readlines():
    row = line.split()
    if len(row) == 0:
      output_file.write("\n")
    elif row[0].startswith("#"):
      begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
      if begin_match:
        doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
        start_map, end_map, word_map = prediction_map[doc_key]
        word_index = 0
      output_file.write(line)
      output_file.write("\n")
    else:
      assert get_doc_key(row[0], row[1]) == doc_key
      coref_list = []
      if word_index in end_map:
        for cluster_id in end_map[word_index]:
          coref_list.append("{})".format(cluster_id))
      if word_index in word_map:
        for cluster_id in word_map[word_index]:
          coref_list.append("({})".format(cluster_id))
      if word_index in start_map:
        for cluster_id in start_map[word_index]:
          coref_list.append("({}".format(cluster_id))

      if len(coref_list) == 0:
        row[-1] = "-"
      else:
        row[-1] = "|".join(coref_list)

      output_file.write("   ".join(row))
      output_file.write("\n")
      word_index += 1

def official_conll_eval(gold_path, predicted_path, metric, official_stdout=False, doc_name="none"):
  cmd = ["conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, doc_name]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  stdout, stderr = process.communicate()
  process.wait()

  stdout = stdout.decode("utf-8")
  if stderr is not None:
    print(stderr)

  if official_stdout:
    print("Official result for {}".format(metric))
    print(stdout)

  coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
  recall = float(coref_results_match.group(1))
  precision = float(coref_results_match.group(2))
  f1 = float(coref_results_match.group(3))
  return { "r": recall, "p": precision, "f": f1 }

def evaluate_conll(gold_path, predictions, official_stdout=False, genres=None):
  # Creates a temporary gold files where all keys not in genres are filtrered out.
  with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_gold:
    create_temp_gold_file(gold_path, temp_gold, genres)
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as prediction_file:
      with open(temp_gold.name, "r") as gold_file:
        if genres is not None:
          output_conll_with_filter(gold_file, prediction_file, predictions, genres)
        else:
          output_conll(gold_file, prediction_file, predictions)
        print("Predicted conll file: {}".format(prediction_file.name))
    return { m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }


def create_temp_gold_file(gold_path, output_file, genres):
  with open(gold_path, "r") as input_file:
    doc_started = False
    allowed_genre = False
    for i,line in enumerate(input_file.readlines()):
      row = line.split()
      if len(row) == 0 and not doc_started:
        output_file.write("\n")
      elif row and row[0].startswith("#"):
        begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
        if begin_match:
          doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
          genre = doc_key[:2]
          allowed_genre = genre in genres.keys()
          if allowed_genre:
            output_file.write(line)
          doc_started = True
        else:
          if allowed_genre:
            output_file.write(line)
            allowed_genre = False
          doc_started = False
      else:
        if allowed_genre:
          output_file.write(line)

