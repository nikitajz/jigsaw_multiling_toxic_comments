""" Finetuning the XLM-RoBERTa model for sequence classification."""

import logging
import os
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import (XLMRobertaTokenizer, 
                          XLMRobertaForSequenceClassification, 
                          XLMRobertaConfig, 
                          XLMRobertaModel,
                          AdamW,
                          HfArgumentParser,
                          set_seed)

sys.path.append(os.getcwd())
from src.config_base import ModelArgs, TrainingArgs
from src.trainer import predict_toxic
from src.utils import load_or_parse_args

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.DEBUG
        )

    logger.info('Fixed the seed')
    set_seed(42)

    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_args.model_checkpoint_path,
        num_labels = 2,   
        output_attentions = False,
        output_hidden_states = False, 
    )

    model.to(training_args.device)

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_args.tokenizer_name)

    sample_input_pt = tokenizer.encode_plus(
        'This is a sample input to demonstrate performance of distiled models especially inference time', 
        return_tensors="pt"
    )

    sample_input_pt.keys()

    logger.debug(f"Tokens ids: {sample_input_pt['input_ids'].tolist()[0]}")
    logger.debug(f"Tokens: {tokenizer.convert_ids_to_tokens(sample_input_pt['input_ids'].tolist()[0])}")
    logger.debug(f"Attention mask: {sample_input_pt['attention_mask'].tolist()[0]}")

    logger.info('Loading datasets')
    import pandas as pd
    cols_to_use = ['id', 'content']
    test_df = pd.read_csv('data/test.csv', usecols=cols_to_use)


    sentences = test_df['content'].values

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

    test_dataset = TensorDataset(input_ids, attention_masks)

    test_dataloader = DataLoader(
                test_dataset, 
                sampler = SequentialSampler(test_dataset), 
                batch_size = training_args.batch_size  
            )

    logger.info('Predicting test set')
    preds = predict_toxic(model, test_dataloader, training_args.device)

    submit = pd.read_csv('data/sample_submission.csv')
    submit['toxic'] = preds
    logger.info(f"There are {(preds == 1).sum()}/{preds.shape[0]} positive samples")
    submit.to_csv('data/submit.csv', index=False)
