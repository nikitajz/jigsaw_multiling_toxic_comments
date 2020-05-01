import csv
import os
import random
import numpy as np
import torch

import logging
logger = logging.getLogger()

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
    
        