import dataclasses
from argparse import Namespace

import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import (XLMRobertaConfig,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup
                          )

from src.dataset import TokenizerCollateFn, ToxicMultilangDataset


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

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collate_fn = TokenizerCollateFn(max_tokens=self.hparams.max_len,
                                             tokenizer_name=self.hparams.tokenizer_name,
                                             cache_dir=self.hparams.cache_dir)

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

    def prepare_data(self):
        self.train_dataset = ToxicMultilangDataset(self.hparams.data_path,
                                                   self.hparams.data_train,
                                                   kind="train",
                                                   resample=self.hparams.resample)

        self.val_dataset = ToxicMultilangDataset(self.hparams.data_path,
                                                 self.hparams.data_valid,
                                                 kind="valid",
                                                 resample=False)
        return None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn
        )

    # def test_dataloader(self):
    #     return None

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.n_epochs
