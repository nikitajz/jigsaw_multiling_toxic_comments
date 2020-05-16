""" Finetuning the XLM-RoBERTa model for sequence classification."""

import logging
import os
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split

from transformers import (XLMRobertaTokenizer, 
                          XLMRobertaForSequenceClassification, 
                          XLMRobertaConfig, 
                          XLMRobertaModel, 
                          AdamW,
                          set_seed)

from src.trainer import train_model, evaluate_performance

logger = logging.getLogger(__name__)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG
    )

logger.info('Fixing the seed')
set_seed(42)

MAX_LEN = 128
MODEL_NAME = "xlm-roberta-base"
DEVICE = 'cuda:0'

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

logger.info(f'Using device: {device}')

model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.to(device)

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

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
# data from first competition. English comments from Wikipedia’s talk page edits.
train_wiki = pd.read_csv('data/jigsaw-toxic-comment-train.csv', usecols=cols_to_use)
# data from second competition. Civil Comments dataset with additional labels.
train_cc = pd.read_csv('data/jigsaw-unintended-bias-train.csv', usecols=cols_to_use)


sentences = train_wiki['comment_text'].values
labels = train_wiki['toxic'].values

logger.info('Applying tokenizer to train dataset')
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = MAX_LEN,           # Pad & truncate all sentences.
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


from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
holdout_size = len(dataset) - train_size

# # Divide the dataset by randomly selecting samples.
train_dataset, holdout_dataset = random_split(dataset, [train_size, holdout_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} holdout samples'.format(holdout_size))


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
holdout_dataloader = DataLoader(
            holdout_dataset, # The validation samples.
            sampler = SequentialSampler(holdout_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 2

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

train_model(model, optimizer, scheduler, train_dataloader, holdout_dataloader, n_epochs=2, device=device)
