import logging

import dill
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn # "cuda" attribute appears only after importing torch.nn #283

from torchtext.data import Field, LabelField, RawField
from torchtext.data import BucketIterator, Iterator
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.utils import format_time

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
       Original: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, path=None):
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
        if path is None:
            path = Path('.checkpoints/early_stopping_checkpoint.pt')
            path.mkdir(exist_ok=True)
        self.path = path

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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, n_epochs, args):
        """Training loop (with optional EarlyStopping criterion).
        Arguments:
            model {nn.Module} -- PyTorch model
            optimizer {[type]} -- optimizer
            scheduler {[type]} -- Loss function
            train_loader {[type]} -- train DataLoader
            valid_loader {[type]} -- validation DataLoader
            args {[type]} -- arguments/parameters for training
            n_epochs {[type]} -- total number of epoch to train the model (if early stopping won't break before this)
        Returns:
            (model, train_loss, valid_loss)
        """
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_epochs = self.args.n_epochs
        self.device = self.args.device

        model_save_dir = Path('.model_checkpoint')
        model_save_dir.mkdir(exist_ok=True)

        self._setup_tensorboard(self.args.tensorboard_enable, self.args.tb_log_dir)

        logger.info(f"**** Running model training for {self.n_epochs} epochs")

    def _setup_tensorboard(self, tb_writer, tb_log_dir):
            if is_tensorboard_available() and self.args.tensorboard_enable:
                tb_log_dir = Path(tb_log_dir)
                tb_log_dir.mkdir(exists_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            elif not is_tensorboard_available():
                    logger.warning(
                        "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
                    )

    def train_model(self, model_path):
        
        epoch_train_loss = []  # the average training loss per epoch
        epoch_valid_loss = []  # the average validation loss per epoch

        early_stopping_enabled = self.args.patience is not None
        if early_stopping_enabled:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, path=self.args.early_stopping_checkpoint_path)

        for epoch in range(1, self.n_epochs+1):
            logger.debug(f'Epoch {epoch}')
            train_losses_epoch = []
            valid_losses_epoch = []
            val_acc = 0
            t0 = time.time()

            self.model.train()
            for step, batch in enumerate(self.train_loader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                if self.args.log_step is not None and step % self.args.log_step == 0:
                    elapsed = format_time(time.time() - t0)
                    logger.debug(' Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_loader), elapsed))

                self.optimizer.zero_grad()
                loss, logits = self.model(b_input_ids, 
                                          token_type_ids=None, 
                                          attention_mask=b_input_mask, 
                                          labels=b_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                train_losses_epoch.append(loss.cpu().detach().item())

            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.valid_loader):
                    b_input_ids = batch[0].to(self.device)
                    b_input_mask = batch[1].to(self.device)
                    b_labels = batch[2].to(self.device)

                    self.optimizer.zero_grad()
                    loss, logits = self.model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask, 
                                        labels=b_labels)
                    
                    valid_losses_epoch.append(loss.cpu().detach().item())

            train_loss_epoch_avg = np.mean(train_losses_epoch)
            valid_loss_epoch_avg = np.mean(valid_losses_epoch)

            epoch_len = len(str(self.args.n_epochs))

            logger.info(f'[{epoch:>{epoch_len}}/{self.args.n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss_epoch_avg:.5f} ' +
                        f'valid_loss: {valid_loss_epoch_avg:.5f}')

            epoch_train_loss.append(train_loss_epoch_avg)
            epoch_valid_loss.append(valid_loss_epoch_avg)

            if early_stopping_enabled:
                early_stopping(valid_loss_epoch_avg, self.model)

                if early_stopping.early_stop:
                    logger.warning("Early stopping")
                    break

        self.model.save_pretrained(model_path)
        logger.info(f'Saved finetuned model to {model_path}')

        return train_losses_epoch, valid_losses_epoch


def evaluate_performance(model, test_iter, device, print_metrics=False):
    """
    Given the trained model and test dataloader, evaluate model performance. Namely:
    - accuracy
    - precision
    - recall
    - F1
    """

    acc = 0
    test_loss = 0.0
    test_size = len(test_iter)
    y_true_l = []
    y_pred_l = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_iter):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            loss, logits = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)

            test_loss += loss.cpu().detach().item()
            logits = torch.argmax(logits, dim=1)
            acc += torch.eq(logits, b_labels).sum().cpu().detach().item()
            y_true_l.append(b_labels)
            y_pred_l.append(logits)

        y_true = torch.cat(y_true_l, 0).cpu().detach().numpy()
        y_pred = torch.cat(y_pred_l, 0).cpu().detach().numpy()
        test_loss /= test_size
        acc /= float(test_size)

    scores = {}
    scores['loss'] = test_loss
    scores['auc'] = roc_auc_score(y_true, y_pred)
    scores['accuracy'] = acc

    if print_metrics:
        for k, v in scores.items():
            logger.info(f'{k.capitalize()}: {v:.3f}')

    return scores
