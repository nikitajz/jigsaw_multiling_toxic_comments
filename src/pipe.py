import logging

import dill
import pandas as pd
import numpy as np
import spacy
import torch

from torchtext.data import Field, LabelField, RawField
from torchtext.data import BucketIterator, Iterator
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
       Original: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def train_model(model, criterion, optimizer, train_loader, valid_loader, patience, n_epochs, avg_loss=True):
    """Training loop with EarlyStopping criterion.

    Arguments:
        model {nn.Module} -- PyTorch model
        criterion {[type]} -- Loss function
        optimizer {[type]} -- optimizer
        train_loader {[type]} -- train DataLoader
        valid_loader {[type]} -- validation DataLoader
        patience {[type]} -- number of epochs to train before stopping since the last best model found
        n_epochs {[type]} -- total number of epoch to train the model (if early stopping won't break before this)

    Returns:
        (model, train_loss, valid_loss)
    """

    train_losses = []  # the average training loss per epoch
    valid_losses = []  # the average validation loss per epoch

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        logger.debug(f'Epoch {epoch}')
        train_losses_epoch = []
        valid_losses_epoch = []
        val_acc = 0

        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            preds = model(data, "en")
            loss = criterion(preds, target.float())
            loss.backward()
            optimizer.step()
            train_losses_epoch.append(loss.item())

        model.eval()
        for data, lang, target in valid_loader:
            preds = model(data, lang)
            loss = criterion(preds, target.float())
            valid_losses_epoch.append(loss.item())
            val_acc += torch.eq(torch.round(torch.sigmoid(preds)
                                            ).long(), target).sum().item()

        train_loss_epoch_avg = np.mean(train_losses_epoch).item()
        valid_loss_epoch_avg = np.mean(valid_losses_epoch).item()

        epoch_len = len(str(n_epochs))

        logger.info(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss_epoch_avg:.5f} ' +
                    f'valid_loss: {valid_loss_epoch_avg:.5f}')

        if avg_loss:
            train_losses.append(train_loss_epoch_avg)
            valid_losses.append(valid_loss_epoch_avg)
        else:
            train_losses.extend(train_losses_epoch)
            valid_losses.extend(valid_losses_epoch)

        early_stopping(valid_loss_epoch_avg, model)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, train_losses_epoch, valid_losses_epoch


def evaluate_performance(model, test_iter, loss_fn, print_metrics=False):
    """
    Given the trained model and test dataloader, evaluate model performance. Namely:
    - accuracy
    - precision
    - recall
    - F1
    """

    acc = 0
    test_loss = 0.0
    test_size = len(test_iter.dataset)
    y_true_l = []
    y_pred_l = []
    model.eval()
    for batch in test_iter:
        data = batch.comment_text
        lang = batch.lang
        target = batch.toxic

        preds = model(data, lang)
        loss = loss_fn(preds, target.float())
        test_loss += loss.item()
        preds = torch.round(torch.sigmoid(preds)).long()
        acc += torch.eq(preds, target).sum().item()
        y_true_l.append(target)
        y_pred_l.append(preds)

    y_true = torch.cat(y_true_l, 0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred_l, 0).cpu().detach().numpy()
    test_loss /= test_size
    acc /= float(test_size)

    scores = {}
    # metrics['loss'] = test_loss
    scores['auc'] = roc_auc_score(y_true, y_pred)
    scores['accuracy'] = acc
    scores['f1'] = f1_score(y_true, y_pred)
    scores['precision'] = precision_score(y_true, y_pred)
    scores['recall'] = recall_score(y_true, y_pred)

    if print_metrics:
        for k, v in scores.items():
            logger.info(f'{k.capitalize()}: {v:.3f}')

    return scores


def get_predictions(model, train_iter):
    ids = []
    toxic = []
    for idx, text, lang in train_iter:
        idx.append(idx)
        pred = model(text, lang)
        toxic.append(pred)

    return pd.DataFrame(list(zip(ids, toxic)), columns=["id", "toxic"]).sort_values(by="id")


class MultilangIter:
    def __init__(self, iter_list):
        """Metaiterator to chain multiple per-language iterators.

        Parameters
        ----------
        iter_list : [list]
            Sequence of torchtext iterators.
        """
        self.iters = iter_list

    def __iter__(self):
        for it in self.iters:
            for batch in it:
                yield batch

    def __len__(self):
        return sum(len(it) for it in self.iters)


def load_data_iters(conf, cache=False, load_cached=False):
    cache_dir = Path(conf.data_path)/"cache"
    cache_dir.mkdir(exist_ok=True)

    if not load_cached:
        device = torch.device(
            conf.device if torch.cuda.is_available() else 'cpu')

        min_len_padding = get_pad_to_min_len_fn(
            min_length=max(conf.kernel_sizes))

        spacy.load('en_core_web_sm')
        stopwords_en = spacy.lang.en.stop_words.STOP_WORDS

        TEXT_LNG = {}
        vectors_dict = {}

        ### train ###
        lang = "en"
        TEXT_LNG[lang] = Field(sequential=True, use_vocab=True, 
                               lower=True, 
                               tokenize='spacy',
                               tokenizer_language=lang,
                               stop_words=stopwords_en,
                               batch_first=True, 
                               #  preprocessing=preprocess,
                               postprocessing=min_len_padding
                               )
        TARGET = LabelField(batch_first=True, use_vocab=False,
                            is_target=True, dtype=torch.float)

        train_j_datafields = [
            ("id", None),  # ignore field
            ("comment_text", TEXT_LNG[lang]),
            ("toxic", TARGET),
            ("severe_toxic", None), ("threat", None),
            ("obscene", None), ("insult", None),
            ("identity_hate", None)
        ]

        logger.debug('Loading train dataset')
        train_joint_ds = TabularDataset(path=conf.train_txc_path, format='csv', skip_header=True,
                                        fields=train_j_datafields)

        train_iter = Iterator(train_joint_ds,
                              batch_size=conf.batch_size_train,
                              sort=False,
                              sort_within_batch=True,
                              sort_key=lambda x: len(x.comment_text),
                              repeat=True, train=True,
                              device=device)

        logger.info('Building train (en) vocabulary')
        vectors_dict[lang] = Vectors(name=conf.vectors.format(lang), cache=conf.vectors_cache)
        
        TEXT_LNG[lang].build_vocab(
            train_joint_ds,
            max_size=conf.vocab_size,
            min_freq=conf.min_freq,
            vectors=vectors_dict[lang]
        )

        ### validation ###

        ID = Field(sequential=False, use_vocab=False, batch_first=True, is_target=False)
        LANG = RawField()

        val_fields_dict = {}
        val_ds_dict = {}
        val_iter_dict = {}

        ### test ###

        test_fields_dict = {}
        test_ds_dict = {}
        test_iter_dict = {}
        for lang in conf.test_langs:
            tokenize_str = "spacy" if lang not in conf.tokenizer_exception_langs else "toktok"
            TEXT_LNG[lang] = Field(sequential=True, use_vocab=True, lower=True, batch_first=True,
                                   tokenize=tokenize_str, tokenizer_language=lang)

            test_fields_dict[lang] = [
                ("idx", ID),
                ("comment_text", TEXT_LNG[lang]),  # content -> comment_text
                ("lang", LANG)
            ]
            logger.info(f'Creating test dataset for language: [{lang}]')
            test_ds_dict[lang] = TabularDataset(path=conf.test_path, format='csv',
                                                fields=test_fields_dict[lang], skip_header=True,
                                                filter_pred=lambda ex: ex.lang == lang)
            logger.debug(f'Dataset for language [{lang}] size: {len(test_ds_dict[lang])}')

            test_iter_dict[lang] = Iterator(test_ds_dict[lang],
                                            batch_size=conf.batch_size_test,
                                            sort=False,
                                            sort_within_batch=False,
                                            repeat=False,
                                            train=False,
                                            device=device)

            if lang in conf.val_langs:
                val_fields_dict[lang] = [
                    ("idx", None),
                    ("comment_text", TEXT_LNG[lang]),
                    ("lang", LANG),
                    ("toxic", TARGET)
                ]
                logger.info(f'Creating validation dataset for language: [{lang}]')
                val_ds_dict[lang] = TabularDataset(path=conf.val_path, format='csv',
                                                   fields=val_fields_dict[lang], skip_header=True,
                                                   filter_pred=lambda ex: ex.lang == lang
                                                   )
                logger.debug(f'Dataset for language [{lang}] has size: {len(val_ds_dict[lang])}')

                val_iter_dict[lang] = Iterator(val_ds_dict[lang],
                                               batch_size=conf.batch_size_train,
                                               sort=False,
                                               sort_within_batch=False,
                                               repeat=False,
                                               train=False,
                                               device=device)

            logger.info(f'Building vocabulary for language [{lang}]')
            vectors_dict[lang] = Vectors(
                name=conf.vectors.format(lang), cache=conf.vectors_cache)
            if lang in conf.val_langs:
                TEXT_LNG[lang].build_vocab(
                    val_ds_dict[lang], test_ds_dict[lang],
                    max_size=conf.vocab_size,
                    min_freq=conf.min_freq,
                    vectors=vectors_dict[lang]
                )
            else:
                TEXT_LNG[lang].build_vocab(
                    test_ds_dict[lang],
                    max_size=conf.vocab_size,
                    min_freq=conf.min_freq,
                    vectors=vectors_dict[lang]
                )

        if cache:
            logger.info("Saving iterators to cache")
            torch.save(train_iter, cache_dir/"train_iter.pkl", pickle_module=dill)
            for lang in conf.val_langs:
                torch.save(val_iter_dict[lang],   cache_dir/f"val_{lang}_iter.pkl", pickle_module=dill)
            for lang in conf.test_langs:
                torch.save(test_iter_dict,  cache_dir/f"test_{lang}_iter.pkl", pickle_module=dill)
            torch.save(TEXT_LNG,   cache_dir/"TEXT_LNG.pkl", pickle_module=dill)

    else:
        logger.info("Loading iterators from cache")
        train_iter = torch.load(cache_dir/"train_iter.pkl", pickle_module=dill)
        for lang in conf.val_langs:
            val_iter_dict[lang] = torch.load(cache_dir/f"val_{lang}_iter.pkl", pickle_module=dill)
        for lang in conf.test_langs:
            test_iter_dict = torch.load(cache_dir/f"test_{lang}_iter.pkl", pickle_module=dill)
        TEXT_LNG = torch.load(cache_dir/"TEXT_LNG.pkl", pickle_module=dill)

    val_iter = MultilangIter(val_iter_dict.values())
    test_iter = MultilangIter(test_iter_dict.values())

    vocabs = {lang: TEXT_LNG[lang].vocab for lang in TEXT_LNG.keys()}

    return train_iter, val_iter, test_iter, vocabs


def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch
    return pad_to_min_len


def preprocess(tokens):
    pass
    return tokens
