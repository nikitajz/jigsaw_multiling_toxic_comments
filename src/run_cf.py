""" Finetuning the XLM-RoBERTa model for sequence classification."""

import logging
from pprint import pformat
import os
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (XLMRobertaTokenizer, 
                          XLMRobertaForSequenceClassification, 
                          XLMRobertaConfig, 
                          XLMRobertaModel, 
                          AdamW,
                          set_seed)
from transformers import get_linear_schedule_with_warmup

from src.config_base import ModelArgs, TrainingArgs
from src.trainer import Trainer, evaluate_performance
from src.utils import load_or_parse_args


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.DEBUG
        )

    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)

    logger.info(f'Fixing the seed: [{training_args.seed}]')
    set_seed(training_args.seed)

    logger.info('Creating model and tokenizer')

    if model_args.load_checkpoint:
        model_name_or_path = model_args.model_checkpoint_path 
        logger.info('Loading fune-tuned model from checkpoint')
    else:
        logger.info('Using pretrained HF model (without finetuning)')
        model_name_or_path = model_args.model_name

    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name_or_path, 
        num_labels = 2,    
        output_attentions = False, 
        output_hidden_states = False, 
    )

    model.to(training_args.device)

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_args.tokenizer_name)

    logger.info('Loading datasets')
    cols_to_use = ['comment_text', 'toxic']
    # data from first competition. English comments from Wikipedia’s talk page edits.
    train_wiki = pd.read_csv('data/jigsaw-toxic-comment-train.csv', usecols=cols_to_use)
    logger.debug(f'Training wiki shape: {train_wiki.shape}')
    # data from second competition. Civil Comments dataset with additional labels.
    train_cc = pd.read_csv('data/jigsaw-unintended-bias-train.csv', usecols=cols_to_use)
    logger.debug(f'Training cc shape: {train_cc.shape}')

    train = pd.concat([train_wiki, train_cc], axis=0)
    assert train.shape[1] == train_wiki.shape[1]
    logger.debug(f'Combined training shape: {train.shape}')

    # TODO: consider using score, but this requires different loss
    train['toxic'] = (train['toxic'] >= 0.5).astype('int')

    # train = train.sample(1000, random_state=training_args.seed)
    # logger.warning(f'Sampled training dataset shape: {train.shape}')

    sentences = train['comment_text'].values
    labels = train['toxic'].values

    logger.info('Applying tokenizer to train dataset')
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = model_args.max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.9 * len(dataset))
    holdout_size = len(dataset) - train_size

    # # Divide the dataset by randomly selecting samples.
    train_dataset, holdout_dataset = random_split(dataset, [train_size, holdout_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} holdout samples'.format(holdout_size))

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = training_args.batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    holdout_dataloader = DataLoader(
                holdout_dataset, # The validation samples.
                sampler = SequentialSampler(holdout_dataset), # Pull out batches sequentially.
                batch_size = training_args.batch_size # Evaluate with this batch size.
            )

    optimizer = AdamW(model.parameters(),
                    lr = training_args.learning_rate, 
                    eps = training_args.adam_epsilon
                    )

    total_steps = len(train_dataloader) * training_args.n_epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    trainer = Trainer(model, optimizer, scheduler, train_dataloader, holdout_dataloader, n_epochs=training_args.n_epochs, args=training_args)

    trainer.train_model(model_args.model_checkpoint_path)
