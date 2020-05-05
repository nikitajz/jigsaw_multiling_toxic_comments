import csv
import os
import dill
import random
import numpy as np
from pathlib import Path

import torch
from torchtext.data import Dataset

import logging

logger = logging.getLogger(__name__)

def set_seed_everywhere(seed, cuda=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_scores_to_file(scores, filepath, precision=5):
    """Save dict with scores in csv file.

    Parameters
    ----------
    scores : [dict]
        Dictinary with scores
    filepath : [str]
        File path
    precision : int, optional
        Precision to round score to, by default 5
    """    
    with open(filepath, 'w') as f:
        w = csv.writer(f, delimiter='\t')
        for key, val in scores.items():
            val = round(val, precision) if precision else val
            w.writerow([key, val])
    logger.debug(f'Saved scores to the file: {filepath}')


def load_scores_from_file(filepath):
    """Load scores from csv file into dictionary.

    Parameters
    ----------
    filepath : [str]
        File in csv format

    Returns
    -------
    [dict]
        Scores from the file
    """    
    scores = {}
    with open(filepath, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for key, val in csv_reader:
            scores[key] = float(val)
    return scores
    
def save_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f'Saving dataset to path {path}')
    torch.save(dataset.examples, path/"examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path/"fields.pkl", pickle_module=dill)

def load_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    logger.debug(f'Loading preprocessed dataset from path: {path}')
    examples = torch.load(path/"examples.pkl", pickle_module=dill)
    fields = torch.load(path/"fields.pkl", pickle_module=dill)
    return Dataset(examples, fields)