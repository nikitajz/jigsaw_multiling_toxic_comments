import logging
import os
from pathlib import Path

import spacy

import torch
import torch.nn as nn
import torch.optim as optim

# from torchtext import datasets
from torchtext.data import Field, LabelField
from torchtext.data import BucketIterator, Iterator
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors

from src.cnn import CNNModel
from src.pipe import train_model, evaluate_performance
from src.utils import set_seed_everywhere, save_scores_to_file


class Config:
    SEED = 42
    device = 'cuda:0'

    learning_rate = 0.005
    weight_decay = 0.0005
    batch_size_train = 256
    batch_size_test = 256
    n_epochs = 20

    kernel_sizes = [3, 4, 5]
    dropout = 0.5
    emb_dim = 300
    num_channels = emb_dim
    hidden_dim = 100
    n_classes = 1
    # pad_idx = TEXT.vocab.stoi['<pad>'] # 1

    vocab_size = 400000
    min_freq = 20
    # https://github.com/pytorch/text/blob/v0.2.1/torchtext/vocab.py#L379-L393
    vectors_cache = os.path.expanduser("~/.vector_cache")
    vectors = "wiki.align.comb.vec"

    data_path = Path('data')
    # data from first competition. English comments from Wikipedia’s talk page edits.
    train_txc_path = data_path/'jigsaw-toxic-comment-train.csv'
    # data from second competition. Civil Comments dataset with additional labels.
    train_ub_path = data_path/'jigsaw-unintended-bias-train.csv'
    val_path = data_path/'validation.csv'
    test_path = data_path/'test.csv'
    val_scores_path = 'val_scores.tsv'


def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch
    return pad_to_min_len


if __name__ == '__main__':
    cfg = Config()
    set_seed_everywhere(cfg.SEED, True)

    logger = logging.getLogger()
    logging.basicConfig(level='DEBUG')

    torch.Tensor.__repr__ = torch.Tensor.__str__ = lambda self: f'Shape:{self.shape.__str__()[11:-1]}'

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: [{device}]')

    min_len_padding = get_pad_to_min_len_fn(min_length=max(cfg.kernel_sizes))

    TEXT = Field(sequential=True, use_vocab=True, lower=True,
                 batch_first=True, postprocessing=min_len_padding)
    LABEL = LabelField(batch_first=True, use_vocab=False)
    TARGET = LabelField(batch_first=True, use_vocab=False, is_target=True)

    txc_datafields = [
        ("id", None),  # ignore field
        ("comment_text", TEXT),
        ("toxic", TARGET),
        ("severe_toxic", None), ("threat", None),
        ("obscene", None), ("insult", None),
        ("identity_hate", None)
        # ("severe_toxic", LABEL), ("threat", LABEL), # ("obscene", LABEL), ("insult", LABEL), # ("identity_hate", LABEL)
    ]

    train_txc = TabularDataset(
        path=cfg.train_txc_path, format='csv', fields=txc_datafields, skip_header=True)

    val_datafields = [
        ("id", None),
        ("comment_text", TEXT),
        ("lang", None),
        ("toxic", TARGET)
    ]

    val = TabularDataset(path=cfg.val_path, format='csv',
                         fields=val_datafields, skip_header=True)

    test_datafields = [
        ("id", None),
        ("comment_text", TEXT),  # "content" -> "comment_text"
        ("lang", None)
    ]

    test = TabularDataset(path=cfg.test_path, format='csv',
                          fields=test_datafields, skip_header=True)

    logger.info(f'Loading pre-trained vector: {cfg.vectors}')
    vectors = Vectors(name=cfg.vectors, cache=cfg.vectors_cache)

    logger.info('Building vocabulary')
    TEXT.build_vocab(
        train_txc, val, test,
        max_size=cfg.vocab_size,
        min_freq=cfg.min_freq,
        vectors=vectors
    )

    train_iter, val_iter = BucketIterator.splits((train_txc, val),
                                                 batch_size=cfg.batch_size_train,
                                                 sort_key=lambda x: len(x.comment_text),
                                                 sort=False,
                                                 sort_within_batch=True,
                                                 repeat=False,
                                                 device=device)

    test_iter = Iterator(test, batch_size=cfg.batch_size_test, device=device,
                         sort=False, sort_within_batch=False, repeat=False, train=False)

    model = CNNModel(emb_vectors=TEXT.vocab.vectors, kernel_sizes=cfg.kernel_sizes,
                     num_channels=cfg.num_channels, hidden_size=cfg.hidden_dim, dropout_p=cfg.dropout, pad_idx=TEXT.vocab.stoi['<pad>'])
    # load pre-trained vectors
    model.emb.weight.data.copy_(TEXT.vocab.vectors)


    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                     lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_func = nn.BCEWithLogitsLoss()
    model.to(device)

    model, train_loss, valid_loss = train_model(model, loss_func, opt, train_iter, val_iter, patience=5, n_epochs=cfg.n_epochs)

    val_scores = evaluate_performance(model, val_iter, loss_func, print_metrics=True)

    save_scores_to_file(val_scores, cfg.val_scores_path)
