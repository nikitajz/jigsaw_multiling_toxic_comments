import logging
from argparse import Namespace

import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from src.dataset import TokenizerCollateFn, ToxicMultilangDataset

logger = logging.getLogger(__name__)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
}


class DistilBERTClassifier(pl.LightningModule):
    def __init__(self, hparams: Namespace, num_labels=2, mode="sequence-classification", **config_kwargs):
        """Initialize a model."""

        super().__init__()
        self.hparams = hparams
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.config = AutoConfig.from_pretrained(
            self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
            **({"num_labels": num_labels} if num_labels is not None else {}),
            cache_dir=cache_dir,
            **config_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
            cache_dir=cache_dir,
            use_fast=True
        )
        self.model = MODEL_MODES[mode].from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=self.config,
            cache_dir=cache_dir,
        )
        if self.hparams.model_mlm_finetuned is not None:
            logger.info("Loading MLM model to use as backbone")
            model_mlm_finetuned = AutoModelWithLMHead.from_pretrained(self.hparams.model_mlm_finetuned)
            self.model.distilbert = model_mlm_finetuned.distilbert
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collate_fn = TokenizerCollateFn(max_tokens=self.hparams.max_len,
                                             tokenizer_name=self.hparams.tokenizer_name,
                                             cache_dir=self.hparams.cache_dir)

    def forward(self, input_ids, attention_mask, labels=None):
        loss, logits = self.model(input_ids,
                                  attention_mask=attention_mask,
                                  labels=labels.long())

        return loss, logits

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        loss, _ = self(b_input_ids, b_input_mask, labels=b_labels)
        logs = {'epoch': batch_idx, 'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        loss, logits = self(b_input_ids, b_input_mask, labels=b_labels)
        b_pred = torch.argmax(logits, dim=1)

        return {'val_loss': loss, 'val_pred': b_pred, 'val_labels': b_labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_preds = torch.cat([x['val_pred'] for x in outputs])
        val_labels = torch.cat([x['val_labels'] for x in outputs])
        accuracy = torch.eq(val_preds, val_labels).float().mean()

        auc_score = roc_auc_score(val_labels.cpu(), val_preds.cpu())
        auc = torch.tensor(auc_score)

        logs = {'val_loss': avg_loss, 'val_auc': auc, 'val_accuracy': accuracy}
        return {'val_loss': avg_loss, 'val_auc': auc, 'val_accuracy': accuracy, 'logs': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        loss, logits = self(b_input_ids, b_input_mask, labels=b_labels)
        b_pred = torch.argmax(logits, dim=1)

        return {'test_loss': loss, 'test_pred': b_pred, 'test_labels': b_labels}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_preds = torch.cat([x['test_pred'] for x in outputs])
        test_labels = torch.cat([x['test_labels'] for x in outputs])
        accuracy = torch.eq(test_preds, test_labels).float().mean()

        auc_score = roc_auc_score(test_labels.cpu(), test_preds.cpu())
        auc = torch.tensor(auc_score)

        logs = {'test_loss': avg_loss, 'test_auc': auc, 'test_accuracy': accuracy}
        return {'test_loss': avg_loss, 'test_auc': auc, 'test_accuracy': accuracy, 'logs': logs, 'progress_bar': logs}

    def is_logger(self):
        return self.trainer.local_rank <= 0

    # def layerwise_lr(self, lr, decay):
    #     """
    #     returns grouped model parameters with layer-wise decaying learning rate
    #     """
    #     bert = self.transformer
    #     num_layers = bert.config.n_layers
    #     opt_parameters = [{'params': bert.embeddings.parameters(), 'lr': lr * decay ** num_layers}]
    #     opt_parameters += [{'params': bert.transformer.layer[layer_n].parameters(),
    #                         'lr': lr * decay ** (num_layers - layer_n + 1)}
    #                        for layer_n in range(num_layers)]
    #     return opt_parameters

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        head_parameters = [(n, p) for n, p in model.named_parameters() if model.base_model_prefix not in n]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.base_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [p for n, p in model.base_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [p for n, p in head_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate * 500,
            },
            {
                "params": [p for n, p in head_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate * 500,
            }

        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps(),
        )

        # TODO: separate optimizer for classification head

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def total_steps(self):
        return (len(self.train_dataloader()) // self.hparams.effective_batch_size
                // self.hparams.accumulate_grad_batches * max(1, self.hparams.min_epochs))

    def prepare_data(self):
        dataset = ToxicMultilangDataset(self.hparams.data_path,
                                        self.hparams.data_train,
                                        kind="train",
                                        resample=self.hparams.resample)
        train_size = int(self.hparams.valid_pct * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, test_size])
        return None

    def train_dataloader(self):
        # self.prepare_data()
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers
        )

        return dataloader

    def test_dataloader(self):
        test_dataset = ToxicMultilangDataset(self.hparams.data_path,
                                             self.hparams.data_test,
                                             kind="valid",  # due to PL confusion test means valid
                                             resample=False)
        dataloader = DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers
        )
        return dataloader
