import logging

import pandas as pd
import numpy as np
import torch
from torchtext.data import Iterator
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score       

logger = logging.getLogger()

# Original: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
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
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
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
    
    train_losses = [] # the average training loss per epoch
    valid_losses = [] # the average validation loss per epoch
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
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
            val_acc += torch.eq(torch.round(torch.sigmoid(preds)).long(), target).sum().item()


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
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, train_losses_epoch, valid_losses_epoch

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
        for k,v in scores.items():
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
