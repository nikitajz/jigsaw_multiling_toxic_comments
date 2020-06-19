from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArgs:
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    model_checkpoint_path: Optional[str] = field(
        default='.model_checkpoint',
        metadata={"help": "Path to save to and load from fine-tuned model (checkpoint)"})
    freeze_backbone: bool = field(
        default=True, metadata={"help": "Freeze Roberta model and train only classifier"})
    load_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to load pretrained model from last checkpoint"})
    max_len: int = field(
        default=256, metadata={"help": "Max number of tokens for model input"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})


@dataclass
class TrainingArgs:
    data_path: str = field(
        default="data/", metadata={"help": "Folder where data located"})
    data_train: List[str] = field(
        default="jigsaw-toxic-comment-train.csv",
        metadata={"help": "Train filename, list of filenames of pattern pathlib.glob"})
    data_valid: str = field(
        default="validation.csv",
        metadata={"help": "Validation filename, list of filenames of pattern pathlib.glob"})
    data_test: str = field(
        default="test.csv",
        metadata={"help": "Test filename, list of filenames of pattern pathlib.glob"})
    reload_dataloaders_every_epoch: bool = field(
        default=False, metadata={"help": "Reload datasets on each epoch or not"})
    resample: bool = field(
        default=False, metadata={"help": "Resample train data to have equal samples per class"})
    shuffle: bool = field(
        default=False, metadata={"help": "Shuffle train data"})
    num_workers: int = field(
        default=1, metadata={"help": "How many workers to use for dataloader"})
    seed: int = field(
        default=42, metadata={"help": "Random number"})
    gpus: int = field(
        default=1, metadata={"help": "Device to train model on. Int for number of gpus," +
                                     " str to select specific one or List[str] to select few specific gpus"})
    n_epochs: int = field(
        default=2, metadata={"help": "Number of epochs to train"})
    accumulate_grad_batches: int = field(
        default=1, metadata={"help": "Steps interval to accumulate gradient."})
    batch_size: int = field(
        default=16, metadata={"help": "Batch size"})
    learning_rate: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    warmup_steps: int = field(
        default=7, metadata={"help": "Warm up steps for optimizer."})
    gradient_clip_val: float = field(
        default=0, metadata={"help": "Clip gradient value. Set 0 to disable."})
    val_check_interval: Optional[int] = field(
        default=1.0, metadata={"help": "How often within one training epoch to check validation set." +
                                       "Set float for fraction or int for steps."})
    early_stop_callback: bool = field(
        default=True, metadata={"help": "Whether to use early stopping"})
    tensorboard_enable: bool = field(
        default=False, metadata={"help": "Whether to use tensorboard"})
    tb_log_dir: str = field(
        default=None, metadata={"help": "Directory to save Tensorboard logs"})
    early_stopping_checkpoint_path: str = field(
        default=".early_stopping", metadata={"help": "Checkpoint path."})
    patience: int = field(
        default=2, metadata={"help": "Early stopping patience"})
