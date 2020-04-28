from pandas import DataFrame

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.features import text_stats


class TextStats(TransformerMixin, BaseEstimator):
    """Extract numerical features from each document. See function `text_stats` for more details."""

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        return DataFrame.from_dict(text_stats(docs))