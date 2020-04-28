import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


DEBUG=True
if DEBUG:
    pd.DataFrame.__repr__ = lambda self: f'Shape:{self.shape.__str__()}'

def oof_validation(model, train_x, train_y, n_splits=5, seed=42):
    """
    Calculate out of fold prediction using 5-fold stratified cross-validation.
    """
    ytr_oof = np.zeros(train_y.shape)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr_idx, oof_idx) in enumerate(kf.split(train_x, train_y)):
        # model = model
        tr_x = train_x.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        oof_x = train_x.iloc[oof_idx]
        model.fit(tr_x, tr_y)
        ytr_oof[oof_idx] = model.predict(oof_x)
    return ytr_oof


def print_metrics(y_true, y_pred, name='', report=True):
    print(f'{name}')
    print(f'AUC: {metrics.roc_auc_score(y_true, y_pred):.3f} '
          f'F1: {metrics.f1_score(y_true, y_pred):.3f} '
          f'Accuracy: {metrics.accuracy_score(y_true, y_pred):.3f}')
    if report:
        print('Report: \n', metrics.classification_report(y_true, y_pred))