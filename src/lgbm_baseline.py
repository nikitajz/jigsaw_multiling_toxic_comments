import numpy as np
import pandas as pd

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


import logging
import warnings

from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import oof_validation, print_metrics
from src.pipeline import TextStats

class Config:
    SEED = 42
    use_preprocessed_data = False
    additional_features = False # TODO: fix
    predict_submission = False
    # data from first competition. English comments from Wikipedia’s talk page edits.
    train_wiki_path = 'data/jigsaw-toxic-comment-train.csv'
    # data from second competition. Civil Comments dataset with additional labels.
    train_cc_path = 'data/jigsaw-unintended-bias-train.csv'
    train_joint_path = 'data/temp/jigsaw-joint-train.csv'
    val_path = 'data/validation.csv'
    test_path = 'data/test.csv'
    submission_path = 'data/sample_submission.csv'

warnings.filterwarnings(action='ignore', category=UserWarning, module='joblib')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

cfg = Config()
if not cfg.use_preprocessed_data:
    train_wiki = pd.read_csv(cfg.train_wiki_path, usecols=['comment_text', 'toxic'])
    train_cc = pd.read_csv(cfg.train_cc_path, usecols=['comment_text', 'toxic'])
    train = pd.concat([train_wiki, train_cc])

    if cfg.additional_features:
        text_stats = TextStats()
        feat_df = text_stats.transform(train['comment_text'])
        train = pd.concat([train, feat_df], axis=1)

    train.to_csv(cfg.train_joint_path, index=False)
else:
    print('Loading pre-computed dataset')
    train = pd.read_csv(cfg.train_joint_path)

# print(f'Train shape: {train.shape}')

train['toxic'] = (train['toxic'] > 0.5).astype(int)
# if DEBUG:
#     train = train.head(1000)
print(f'Train columns: {train.columns.tolist()}')
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['toxic']), 
                                                    train['toxic'],
                                                    test_size=0.2, random_state=cfg.SEED,
                                                    shuffle=True, stratify=train['toxic'])

val_df = pd.read_csv(cfg.val_path)
# select only the same columns as in train dataset
val_df = val_df[train.columns]
print(f'Validation columns: {train.columns.tolist()}')
X_val, y_val = val_df.drop(columns=['toxic']), val_df['toxic']

params_vect = {
    'stop_words': 'english', 
    'min_df':3,
    'max_df':0.9,
    'ngram_range':(1,2)
}

text_stats_pipe = Pipeline([('stats', TextStats()), 
                            ('scaler', StandardScaler())
                           ], memory='./.run_cache', verbose=True)

pipe = Pipeline([
    ('vect', ColumnTransformer([
        ('vect_comment', TfidfVectorizer(**params_vect), 'comment_text'),
        ('text_stats', text_stats_pipe, 'comment_text')
        ],n_jobs=-1)),
    ('clf', LGBMClassifier(n_jobs=-1, random_state=cfg.SEED))
], memory='./.run_cache', verbose=True)

print('Fitting pipeline')
pipe.fit(X_train, y_train)

# print('OOF validation')
# oof_pred = oof_validation(pipe, X_train, y_train, n_splits=5)
# print_metrics(y_train, oof_pred, name='Out-of-fold scores')

print('Predicting holdout')
threshold = 0.5
pred_holdout = (pipe.predict_proba(X_test) > threshold)[:, 1].astype(int)  # pipe.predict(X_test)
print_metrics(y_test, pred_holdout, name='Holdout scores')

print('Predicting validation dataset')
pred_val = (pipe.predict_proba(X_val) > threshold)[:, 1].astype(int)  # pipe.predict(X_test)
print_metrics(y_val, pred_val, name='Validation scores')

if cfg.predict_submission:
    test = pd.read_csv(cfg.test_path)
    test = test.rename(columns={'content': 'comment_text'})
    sub = pd.read_csv(cfg.submission_path)
    sub['toxic'] = pipe.predict(test)
    sub.to_csv('submission.csv', index=False)