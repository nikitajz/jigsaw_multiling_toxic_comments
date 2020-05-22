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
from src.trainer import evaluate_performance
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
    cols_to_use = ['comment_text', 'toxic']
    val_df = pd.read_csv('data/validation.csv', usecols=cols_to_use)

    sentences = val_df['comment_text'].values
    labels = val_df['toxic'].values

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
    val_dataset = TensorDataset(input_ids, attention_masks, labels)

    val_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = training_args.batch_size  
            )

    evaluate_performance(model, val_dataloader, training_args.device, print_metrics=True)
