from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn # for cuda attribute

@dataclass
class ModelArgs:
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_len: int = field(
        default=256, metadata={"help": "Max number of tokens for model input"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_checkpoint_path: Optional[str] = field(
        default='.model_checkpoint',
        metadata={"help": "Path to save to and load from fune-tuned model (checkpoint)"}
    )
    load_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to load pretrained model from last checkpoint"}
    )

@dataclass
class TrainingArgs:  
    dataset: str = field(
        default='en', metadata={"help": "Which dataset to use for training. Possible options are: `en`, `google-translated`, `wiki`, `cc`"}
    )
    resample: bool = field(
        default=False, metadata={"help": "Resample to have equal samples per class"}
    )
    seed: int = field(
        default=42, metadata={"help": "Random number"}
    )
    device: str = field(
        default='cuda:0', metadata={"help": "Device to train model on"}
    )
    freeze_backbone: bool = field(
        default=False, metadata={"help": "Freeze Roberta model and train only classifier"}
    )    
    n_epochs: int = field(
        default=2, metadata={"help": "Number of epochs to train"}
    )    
    batch_size: int = field(
        default=16, metadata={"help": "Batch size"}
    )
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for Adam."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay if we apply some."}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."}
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    log_step: Optional[int] = field(
        default=40, metadata={"help": "Log training metrics every $step"}
    )
    eval_on_log_step: bool = field(
        default=False, metadata={"help": "Evaluate on validation dataset on each log step"}
    )    
    tensorboard_enable: bool = field(
        default=False, metadata={"help": "Whether to use tensorboard"}
    )
    tb_log_dir: str = field(
        default=None, metadata={"help": "Directory to save Tensorboard logs"}
    )
    early_stopping_checkpoint_path: str = field(
        default="early_stopping_checkpoint.pt", metadata={"help": "Checkpoint path."}
    )
    patience: int = field(
        default=5, metadata={"help": "Early stopping patience"}
    )

    def __post_init__(self):
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")