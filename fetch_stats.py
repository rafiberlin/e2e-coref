#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import util

# Usage: python3 fetch_stats twiconv_bn

config = util.initialize_from_env()
train_path = Path(config["train_path"])
genres = { g:i for i, g in enumerate(config["genres"]) }

with train_path.open("r") as f:
    num_documents = 0
    num_clusters = 0
    num_tokens = 0
    num_mentions = 0
    for jsonline in f.readlines():
        obj = json.loads(jsonline)
        genre = obj["doc_key"][:2]
        if genre in genres:
            clusters = obj["clusters"]
            sents = obj["sentences"]
            words = sum([len(sent) for sent in sents])
            num_tokens += words
            num_clusters += len([x for x in clusters if len(x) > 1])
            num_mentions += sum([len(x) for x in clusters if len(x) > 1])
            num_documents += 1
    print("Num tokens:", num_tokens)
    print("Num chains:", num_clusters)
    print("Num mentions:", num_mentions)
    print("Num documents:", num_documents)
