#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
root = Path("data")

def main():
    df = pd.Series(root.glob("*._with_boundaries_gold_conll"))
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_dev = df[mask]
    test = df[~mask]

    mask = np.random.rand(len(train_dev)) < 0.8
    train = train_dev[mask]
    dev = train_dev[~mask]

    with (root / "train.english.v4_gold_conll").open("a") as outfile:
        for t in train:
            with open(t, "r") as f:
                outfile.write(f.read())


    with (root / "dev.english.v4_gold_conll").open("a") as outfile:
        for t in dev:
            with open(t, "r") as f:
                outfile.write(f.read())

    with (root / "test.english.v4_gold_conll").open("a") as outfile:
        for t in test:
            with open(t, "r") as f:
                outfile.write(f.read())

if __name__ == "__main__":
    main()
