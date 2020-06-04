# jigsaw-toxic-comments
https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

There were few approaches tried (each in a separate [branch]):
- [basic_models] LightGBM and Logistic Regression on TFIDF features. This usually works as a good baseline for monoligual models (test/pred is the same as train). But not in this case. Here just for completeness.
- [conv_multiling] The main idea was to use [MUSE](https://github.com/facebookresearch/MUSE) (aligned pretrained multilingual vectors from FAIR). Approach has complicated Embeddings structure inside the model with separate embedding layer per language. It didn't work, gave up the idea because there were more modern BERT-based architectures.
- [xlm-roberta] Uses XLM-Roberta from HF Transformers. Public score ~0.79 which is far from top, though plenty of training tricks are yet to be added.

Overall it's still WIP and above approaches more aimed for learning rather than getting a high score.