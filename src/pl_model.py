import dataclasses
import datetime
import logging
import warnings
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (XLMRobertaConfig,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup
                          )

from src.config_base import ModelArgs, TrainingArgs
from src.utils import load_or_parse_args


class TokenizerCollateFn:
    def __init__(self, max_tokens=512, model_name="xlm-roberta-base"):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.max_len = max_tokens

    def __call__(self, batch):
        tok_output = self.tokenizer.batch_encode_plus([x[0] for x in batch],
                                                      add_special_tokens=True,
                                                      max_length=self.max_len,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt'
                                                      )
        labels = torch.tensor([x[1] for x in batch])
        return tok_output['input_ids'], tok_output['attention_mask'], labels


class ToxicMultilangDataset(Dataset):
    def __init__(self, folder, filenames, kind, resample=False):
        """
        Load data and create dataset for Dataloader
        Args:
            folder: Union[str, Path]
                path to folder
            filenames: Union[str, List[str]]
                filename, pattern in pathlib.glob format (e.g. "train_*csv") or list of filenames
            kind: str
                one of three options: ("train", "valid", "test")
            resample: bool
                whether to resample dataframe
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        assert kind in ["train", "valid", "test"], "Incorrect kind, should be one of three: ['train', 'valid', 'test']"
        self.column = SimpleNamespace(**{"text": "comment_text",
                                         "target": "toxic" if kind != "test" else None})
        self.kind = kind
        path = Path(folder)
        if kind == "train":
            if '*' in filenames:
                path = path.resolve()
                filenames = list(path.glob(filenames))
                if len(filenames) == 0:
                    raise FileNotFoundError(f"No files were found in directory {path} matching pattern `{filenames}`")
            if isinstance(filenames, str):
                self.data = pd.read_csv(path / filenames, usecols=[self.column.text, self.column.target])
            elif isinstance(filenames, list):
                self.data = pd.concat([pd.read_csv(path / fn, usecols=[self.column.text, self.column.target])
                                       for fn in filenames])
            if resample:
                data_pos = self.data[self.data[self.column.target] == 1]
                data_neg = self.data[self.data[self.column.target] == 0].sample(n=data_pos.shape[0])
                # , random_state=17)
                self.data = pd.concat([data_pos, data_neg])  # .sample(frac=1)  # shuffle
        elif kind == "valid":
            self.data = pd.read_csv(path / filenames, usecols=[self.column.text, self.column.target])
        elif kind == "test":
            self.column.text = "content"
            self.data = pd.read_csv(path / filenames, usecols=[self.column.content])  # , self.column.lang

        pos_samples = self.data[self.data[self.column.target] == 1].shape[0]
        self.logger.info(f"{kind.capitalize()} dataset shape: ({','.join(map(str, self.data.shape))})."
                         f" Target positive samples ratio: {pos_samples / self.data.shape[0]:3f}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row[self.column.text], row.get(self.column.target, None)

    def __len__(self):
        return self.data.shape[0]


class XLMRobertaSeqClassificationPL(pl.LightningModule):
    def __init__(self, model_args, training_args, num_labels=2, **config_kwargs):
        """Initialize model"""
        super().__init__()
        self.model_args = model_args
        self.hparams = Namespace(**dataclasses.asdict(training_args), **dataclasses.asdict(model_args))
        cache_dir = self.model_args.cache_dir if self.model_args.cache_dir else None

        model_name_or_path = model_args.model_checkpoint_path if model_args.load_checkpoint else model_args.model_name
        self.config = XLMRobertaConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            **config_kwargs
        )
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else model_name_or_path,
            cache_dir=cache_dir
        )
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )

        if model_args.freeze_backbone:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        loss, logits = self.model(input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=None,
                                  labels=labels.long())

        return loss, logits

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda param: param.requires_grad, self.model.parameters()),
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon
                          )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        loss, logits = self(b_input_ids, b_input_mask, labels=b_labels)
        tensorboard_logs = {'train/train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        loss, logits = self(b_input_ids, b_input_mask, labels=b_labels)
        b_pred = torch.argmax(logits, dim=1)
        return {'batch/val_loss': loss, 'batch/val_pred': b_pred, 'batch/val_labels': b_labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['batch/val_loss'] for x in outputs]).mean()
        val_preds = torch.cat([x['batch/val_pred'] for x in outputs])
        val_labels = torch.cat([x['batch/val_labels'] for x in outputs])
        accuracy = torch.eq(val_preds, val_labels).float().mean()

        auc_score = roc_auc_score(val_labels.cpu(), val_preds.cpu())
        auc = torch.tensor(auc_score)
        tensorboard_logs = {'train/val_loss': avg_loss, 'train/val_auc': auc, 'train/val_accuracy': accuracy}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # def prepare_data(self):
    #     return None

    def train_dataloader(self):
        return DataLoader(
            ToxicMultilangDataset(self.hparams.data_path,
                                  self.hparams.data_train,
                                  kind="train",
                                  resample=self.hparams.resample),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            # sampler: Optional[Sampler[int]] = ...,
            num_workers=self.hparams.num_workers,
            collate_fn=TokenizerCollateFn()
        )

    def val_dataloader(self):
        return DataLoader(
            ToxicMultilangDataset(self.hparams.data_path,
                                  self.hparams.data_valid,
                                  kind="valid",
                                  resample=False),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            # sampler: Optional[Sampler[int]] = ...,
            num_workers=self.hparams.num_workers,
            collate_fn=TokenizerCollateFn()
        )

    # def test_dataloader(self):
    #     return None

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.n_epochs


def main():
    logger = logging.getLogger(__name__)

    dt_str = datetime.datetime.now().strftime("%y%m%d_%H-%M")
    path_output = Path("experiments") / dt_str
    path_output.mkdir(exist_ok=True, parents=True)

    warnings.filterwarnings('ignore', category=UserWarning)
    m_args, tr_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)
    hparams = Namespace(**dataclasses.asdict(m_args), **dataclasses.asdict(tr_args))

    path_log_file = path_output / "train.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=path_log_file,
        filemode='w'
    )
    if not hparams.load_checkpoint:
        logger.info("Creating the model and trainer.")
        pl_loggers = [loggers.TestTubeLogger(save_dir=path_output),
                      loggers.TensorBoardLogger('tb_logs/')]
        model = XLMRobertaSeqClassificationPL(m_args, tr_args)
        trainer = pl.Trainer.from_argparse_args(hparams,
                                                # fast_dev_run=True,
                                                default_root_dir=path_output,
                                                # weights_save_path=path_output,
                                                weights_summary=None,
                                                # auto_lr_find=True,
                                                logger=pl_loggers,
                                                progress_bar_refresh_rate=10
                                                )
    else:
        logger.info("Loading the model and trainer from checkpoint:", hparams.model_checkpoint_path)
        model = XLMRobertaSeqClassificationPL.load_from_checkpoint(hparams.model_checkpoint_path)
        trainer = pl.Trainer.resume_from_checkpoint(hparams.model_checkpoint_path)

    trainer.fit(model)
    trainer.save_checkpoint(path_output / "model.pt")


if __name__ == '__main__':
    main()
