from dataclasses import dataclass, field
from typing import Optional

# PathType = Union[str, List[str]]
# DevicesType = Union[List[int], int, None]
DevicesType = Optional[int]
PathType = str


@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    model_mode: str = field(
        default="base", metadata={"help": "Model mode according to HF, e.g. `base`, `language-modeling`"})
    freeze_backbone: bool = field(
        default=False, metadata={"help": "Freeze Roberta model and train only classifier"})
    max_len: int = field(
        default=128, metadata={"help": "Max number of tokens for model input"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be saved"}
    )


@dataclass
class TrainArgs:
    load_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to resume training from last checkpoint"})
    data_path: str = field(
        default="data/", metadata={"help": "Folder where data located"})
    data_train: str = field(  #
        default="jigsaw-toxic-comment-train.csv",
        metadata={"help": "Train filename, list of filenames of pattern pathlib.glob"})
    # data_valid: PathType = field(
    #     default="validation.csv",
    #     metadata={"help": "Validation filename, list of filenames of pattern pathlib.glob"})
    shuffle: bool = field(
        default=False, metadata={"help": "Shuffle train data"})
    data_test: PathType = field(
        default="test.csv",
        metadata={"help": "Test filename, list of filenames of pattern pathlib.glob"})
    valid_pct: float = field(
        default=0.1, metadata={"help": "What percent of training data to use for validation"})
    val_check_interval: Optional[int] = field(
        default=1.0, metadata={"help": "How often within one training epoch to check validation set." +
                                       "Set float for fraction or int for steps."})
    limit_val_batches: float = field(
        default=1.0, metadata={"help": "How much of validation dataset to check (floats = percent, int = num_batches)"})
    reload_dataloaders_every_epoch: bool = field(
        default=False, metadata={"help": "Reload datasets on each epoch or not"})
    resample: bool = field(
        default=False, metadata={"help": "Resample train data to have equal samples per class"})
    num_workers: int = field(
        default=1, metadata={"help": "How many workers to use for dataloader"})
    seed: int = field(
        default=42, metadata={"help": "Random number"})
    min_epochs: int = field(
        default=1, metadata={"help": "Force training for at least these many epochs"})
    max_epochs: int = field(
        default=10, metadata={"help": "Stop training once this number of epochs is reached"})
    gpus: Optional[int] = field(  # Union[int, str, List[int], None]
        default=None, metadata={"help": "Device to train model on. Int for number of gpus," +
                                        " str to select specific one or List[str] to select few specific gpus"})
    tpu_cores: Optional[int] = field(  #
        default=None, metadata={"help": "Train model on TPU if n > 0, number or list of specific TPU cores."})
    accumulate_grad_batches: int = field(
        default=1, metadata={"help": "Steps interval to accumulate gradient."})
    auto_scale_batch_size: Optional[str] = field(
        default=False,
        metadata={"help": "If set to True, will initially run a batch size finder trying to find "
                          "the largest batch size that fits into memory. The result will be stored in self.batch_size "}
    )
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
        default=1.0, metadata={"help": "Clip gradient value. Set 0 to disable."})
    early_stop_callback: bool = field(
        default=True, metadata={"help": "Whether to use early stopping"})
    tensorboard_enable: bool = field(
        default=False, metadata={"help": "Whether to use tensorboard"})
    tb_log_dir: str = field(
        default=None, metadata={"help": "Directory to save Tensorboard logs"})
    wandb_enable: bool = field(
        default=False, metadata={"help": "Whether to use log experiments in Weights & Biases"})
    early_stopping_checkpoint_path: str = field(
        default=".early_stopping", metadata={"help": "Checkpoint path."})
    patience: int = field(
        default=2, metadata={"help": "Early stopping patience"})
    do_train: bool = field(
        default=True, metadata={"help": "Whether to run a model training"}
    )
    do_test: bool = field(
        default=False, metadata={"help": "Whether to run a model test"}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run predictions on the test set."}
    )
    mlm: bool = field(
        default=False, metadata={"help": "Whether to use Masked Language Model (collator)"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Masking probability for Language Model"}
    )
    effective_batch_size: int = field(init=False)

    @property
    def n_devices(self) -> int:
        devices = None  # CPU
        if self.tpu_cores and self.gpus:
            raise ValueError("Both TPUS and GPU options are supplied. Specify only one.")
        elif self.tpu_cores is not None:
            devices = self.tpu_cores
        elif self.gpus is not None:
            devices = self.gpus
        else:
            raise ValueError("Unknown error with TPU and GPU arguments")

        if isinstance(devices, int):
            n_devices = devices
        elif isinstance(devices, list):
            n_devices = len(devices)
        elif isinstance(devices, str):
            n_devices = 1
        else:
            n_devices = 1  # CPU
        return n_devices

    def __post_init__(self):
        if 0 < self.mlm_probability > 1.0:
            raise ValueError("Incorrect value for mlm_probability. Should be between 0 and 1.0, default=0.15 (15%)")
        if 0 < self.valid_pct > 1.0:
            raise ValueError("Incorrect value for valid_pct. Should be between 0 and 1.0, default=0.1 (10%)")
        self.effective_batch_size = self.batch_size * self.n_devices
