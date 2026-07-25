#!/usr/bin/env python3
"""Print raw and shuffled-training Ala2 box metadata."""

import sys
from pathlib import Path

import numpy as np
from chemtrain.data import preprocessing

path = Path(sys.argv[1])
with np.load(path, allow_pickle=False) as archive:
    data = {key: np.asarray(archive[key]) for key in archive.files}
training, _, _ = preprocessing.train_val_test_split(
    data,
    train_ratio=0.9,
    val_ratio=0.1,
    shuffle=True,
    shuffle_seed=0,
)
print("raw_first", data["box"][0])
print("training_first", training["box"][0])
print("max_abs", np.max(np.abs(data["box"][0] - training["box"][0])))
