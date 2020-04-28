import pytest
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from src.features import text_stats


def test_text_stats():
    cats = ['rec.autos', 'misc.forsale', 'rec.motorcycles']
    news = fetch_20newsgroups(subset='train', categories=cats, random_state=42)
    data = news.data
    feats = text_stats(data)
    assert isinstance(feats, dict)
    assert len(feats) == 4
