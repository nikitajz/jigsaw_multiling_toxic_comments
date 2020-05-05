import logging
import os
import sys
from pathlib import Path

import spacy

import torch
import torch.nn as nn
import torch.optim as optim

from src.cnn import CNNMultiling
from src.pipe import MultilangIter, evaluate_performance, get_predictions, load_data_iters, train_model
from src.utils import set_seed_everywhere, save_scores_to_file


class Config:
    SEED = 42
    device = 'cuda:0' # 'cpu' #

    learning_rate = 0.005
    weight_decay = 0.0005
    batch_size_train = 64
    batch_size_test = 256
    n_epochs = 1

    kernel_sizes = [3, 4, 5]
    dropout = 0.5
    emb_dim = 300
    num_channels = emb_dim
    hidden_dim = 100
    n_classes = 1
    # pad_idx = TEXT.vocab.stoi['<pad>'] # 1

    vocab_size = 100000
    min_freq = 5
    # https://github.com/pytorch/text/blob/v0.2.1/torchtext/vocab.py#L379-L393
    vectors_cache = ".vector_cache" # symlinked to ~/.vector_cache
    vectors = "wiki.{}.align.vec"

    data_path = Path('data')
    # data from first competition. English comments from Wikipedia’s talk page edits.
    train_txc_path = data_path/'jigsaw-toxic-comment-train.csv'
    # data from second competition. Civil Comments dataset with additional labels.
    train_ub_path = data_path/'jigsaw-unintended-bias-train.csv'
    train_j_path = data_path/'jigsaw-joint-train.csv'
    val_path = data_path/'validation.csv'
    test_path = data_path/'test.csv'
    val_scores_path = 'val_scores.tsv'

    val_langs = ['es', 'it', 'tr']
    test_langs = ['tr', 'ru', 'it', 'fr', 'pt', 'es']
    tokenizer_exception_langs = ['ru', 'tr']


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level='DEBUG')
    
    cfg = Config()
    set_seed_everywhere(cfg.SEED, True)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    torch.Tensor.__repr__ = torch.Tensor.__str__ = lambda self: f'Shape:{self.shape.__str__()[11:-1]}'

    logger.info(f'Using device: [{device}]')

    train_iter, val_iter, test_iter, text_vocabs = load_data_iters(cfg, cache=False)

    ## Sanity check
    ex_train = next(iter(train_iter))

    ex_val_comb = next(iter(val_iter))    

    ex_test_comb = next(iter(test_iter))

    ## Load vectors & init model

    logger.debug(f'Extracting pre-trained vectors')
    emb_vectors_dict = {lang: vocab.vectors for lang, vocab in text_vocabs.items()}

    model = CNNMultiling(emb_vectors_dict=emb_vectors_dict, 
                         kernel_sizes=cfg.kernel_sizes,
                         num_channels=cfg.num_channels, 
                         hidden_size=cfg.hidden_dim, 
                         dropout_p=cfg.dropout#, 
                        #  pad_idx=TEXT_LNG["en"].vocab.stoi['<pad>']
                         )

    opt = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), 
                     lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    loss_func = nn.BCEWithLogitsLoss()
    model.to(device)

    # debug
    # ex_pred = model(ex_val.comment_text, ex_val.lang)
    logger.info(f'Training the model for {cfg.n_epochs} epochs')
    model, train_loss, valid_loss = train_model(model, loss_func, opt, train_iter, val_iter, patience=5, n_epochs=cfg.n_epochs)

    logger.info(f'Evaluating score on validation set')
    val_scores = evaluate_performance(model, val_iter, loss_func, print_metrics=True)

    save_scores_to_file(val_scores, cfg.val_scores_path)

    model.to(device)
    pred_df = get_predictions(model, test_iter)
    pred_df.to_csv("submit.csv", index=False)
