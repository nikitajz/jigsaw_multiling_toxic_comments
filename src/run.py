import logging
import os
import sys
from pathlib import Path

import spacy

import torch
import torch.nn as nn
import torch.optim as optim

# from torchtext import datasets
from torchtext.data import Field, LabelField, RawField
from torchtext.data import BucketIterator, Iterator
from torchtext.data import TabularDataset
from torchtext.vocab import Vectors

from src.cnn import CNNMultiling
from src.pipe import MultilangIter, evaluate_performance, get_predictions, train_model
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


if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level='DEBUG')
    
    cfg = Config()
    set_seed_everywhere(cfg.SEED, True)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    torch.Tensor.__repr__ = torch.Tensor.__str__ = lambda self: f'Shape:{self.shape.__str__()[11:-1]}'

    logger.info(f'Using device: [{device}]')

    min_len_padding = get_pad_to_min_len_fn(min_length=max(cfg.kernel_sizes))

    spacy.load('en_core_web_sm')
    stopwords_en = spacy.lang.en.stop_words.STOP_WORDS

    TEXT_LNG = {}
    vectors_dict = {}

    ## train ##
    lang = "en"
    TEXT_LNG[lang] = Field(sequential=True, use_vocab=True, lower=True, tokenize='spacy',
                #  preprocessing=preprocess,
                 batch_first=True, postprocessing=min_len_padding,
                 stop_words=stopwords_en
                 )
    # LabelField doesn't work because is_target=True is hardcoded there
    # LABEL = Field(sequential=False, use_vocab=False, batch_first=True, is_target=False,  dtype=torch.float)
    TARGET = LabelField(batch_first=True, use_vocab=False, is_target=True, dtype=torch.float)

    train_j_datafields = [
        ("id", None),  # ignore field
        ("comment_text", TEXT_LNG[lang]),
        ("toxic", TARGET),
        ("severe_toxic", None), ("threat", None),
        ("obscene", None), ("insult", None),
        ("identity_hate", None)
        # ("severe_toxic", LABEL), ("threat", LABEL), ("obscene", LABEL), ("insult", LABEL), ("identity_hate", LABEL)
    ]

    logger.debug('Loading train dataset')
    train_joint_ds = TabularDataset(path=cfg.train_txc_path, format='csv', skip_header=True,
                                    fields=train_j_datafields)

    train_iter = Iterator(train_joint_ds, 
                          batch_size=cfg.batch_size_train, 
                          sort=False, 
                          sort_within_batch=True, 
                          sort_key=lambda x: len(x.comment_text),
                          repeat=True, train=True,
                          device=device)


    logger.info('Building train (en) vocabulary')
    vectors_dict[lang] = Vectors(name=cfg.vectors.format(lang), cache=cfg.vectors_cache)
    TEXT_LNG[lang].build_vocab(
        train_joint_ds,
        max_size=cfg.vocab_size,
        min_freq=cfg.min_freq,
        vectors=vectors_dict[lang]
    )

    ## validation ##

    ID = Field(sequential=False, use_vocab=False, batch_first=True, is_target=False) #dtype=torch.long, 
    LANG = RawField()

    val_fields_dict = {}
    val_ds_dict = {}
    val_iter_dict = {}
    # for lang in cfg.val_langs:
    #     tokenize_str = "spacy" if lang not in cfg.tokenizer_exception_langs else "toktok"
    #     TEXT_LNG[lang] = Field(sequential=True, use_vocab=True, lower=True, batch_first=True,
    #                         tokenize=tokenize_str, tokenizer_language=lang)

        
    #     val_fields_dict[lang] = [
    #             ("idx", None),
    #             ("comment_text", TEXT_LNG[lang]),
    #             ("lang", LANG),
    #             ("toxic", TARGET)
    #     ]
    #     logger.info(f'Creating validation dataset for language: [{lang}]')
    #     val_ds_dict[lang] = TabularDataset(path=cfg.val_path, format='csv',
    #                             fields=val_fields_dict[lang], skip_header=True,
    #                             filter_pred=lambda ex: ex.lang == lang
    #                             )
    #     logger.debug(f'Dataset for language [{lang}] has size: {len(val_ds_dict[lang])}')
        
    #     val_iter_dict[lang] = Iterator(val_ds_dict[lang], batch_size=16, device=device,
    #                         sort=False, sort_within_batch=False, repeat=False, train=False)

    ## test ##

    test_fields_dict = {}
    test_ds_dict = {}
    test_iter_dict = {}
    for lang in cfg.test_langs:
        # if lang not in TEXT_LNG.keys():
        tokenize_str = "spacy" if lang not in cfg.tokenizer_exception_langs else "toktok"
        TEXT_LNG[lang] = Field(sequential=True, use_vocab=True, lower=True, batch_first=True,
                                tokenize=tokenize_str, tokenizer_language=lang)

        test_fields_dict[lang] = [
                ("idx", ID),
                ("comment_text", TEXT_LNG[lang]), # content -> comment_text
                ("lang", LANG)
        ]
        logger.info(f'Creating test dataset for language: [{lang}]')
        test_ds_dict[lang] = TabularDataset(path=cfg.test_path, format='csv',
                                fields=test_fields_dict[lang], skip_header=True,
                                filter_pred=lambda ex: ex.lang == lang)
        test_iter_dict[lang] = Iterator(test_ds_dict[lang], 
                                        batch_size=cfg.batch_size_test,
                                        sort=False, 
                                        sort_within_batch=False, 
                                        repeat=False, 
                                        train=False,
                                        device=device)
        logger.debug(f'Dataset for language [{lang}] size: {len(test_ds_dict[lang])}')

        if lang in cfg.val_langs:
            val_fields_dict[lang] = [
                ("idx", None),
                ("comment_text", TEXT_LNG[lang]),
                ("lang", LANG),
                ("toxic", TARGET)
            ]
            logger.info(f'Creating validation dataset for language: [{lang}]')
            val_ds_dict[lang] = TabularDataset(path=cfg.val_path, format='csv',
                                    fields=val_fields_dict[lang], skip_header=True,
                                    filter_pred=lambda ex: ex.lang == lang
                                    )
            logger.debug(f'Dataset for language [{lang}] has size: {len(val_ds_dict[lang])}')
            
            val_iter_dict[lang] = Iterator(val_ds_dict[lang], 
                                           batch_size=cfg.batch_size_train,
                                           sort=False, 
                                           sort_within_batch=False, 
                                           repeat=False, 
                                           train=False,
                                           device=device)

        logger.info(f'Building vocabulary for language [{lang}]')
        vectors_dict[lang] = Vectors(name=cfg.vectors.format(lang), cache=cfg.vectors_cache)
        if lang in cfg.val_langs:
            TEXT_LNG[lang].build_vocab(
                val_ds_dict[lang], test_ds_dict[lang],
                max_size=cfg.vocab_size,
                min_freq=cfg.min_freq,
                vectors=vectors_dict[lang]
            )
        else:
            TEXT_LNG[lang].build_vocab(
                test_ds_dict[lang],
                max_size=cfg.vocab_size,
                min_freq=cfg.min_freq,
                vectors=vectors_dict[lang]
            )

    val_iter = MultilangIter(val_iter_dict.values())
    test_iter = MultilangIter(test_iter_dict.values())


    ## Sanity check
    ex_train = next(iter(train_iter))

    ex_val_comb = next(iter(val_iter))    

    ex_test_comb = next(iter(test_iter))

    ## Load vectors & init model

    logger.info(f'Loading pre-trained vectors')
    emb_vectors_dict = {lang: TEXT_LNG[lang].vocab.vectors for lang in TEXT_LNG.keys()}

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