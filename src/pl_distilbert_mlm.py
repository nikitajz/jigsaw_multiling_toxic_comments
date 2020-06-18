import logging
import os
from argparse import Namespace

import pytorch_lightning as pl
import torch
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
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset
)

logger = logging.getLogger(__name__)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
}


class BaseTransformer(pl.LightningModule):
    def __init__(self, hparams: Namespace, num_labels=None, mode="base", **config_kwargs):
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
        self.train_dataset = None
        self.valid_dataset = None
        self.collate_fn = None

    def forward(self, input_ids, mlm_labels, **kwargs):
        outputs = self.model.forward(input_ids=input_ids, masked_lm_labels=mlm_labels)
        return outputs

    def training_step(self, batch, batch_idx):
        b_input_ids = batch["input_ids"]
        b_labels = batch["masked_lm_labels"]

        loss, logits = self(b_input_ids, b_labels)
        logs = {'epoch': batch_idx, 'train/train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch["input_ids"]
        b_labels = batch["masked_lm_labels"]

        loss, logits = self(b_input_ids, b_labels)
        logs = {'epoch': batch_idx, 'train/val_loss': loss}
        return {'val_loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'loss': avg_loss, 'logs': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs}

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
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

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

    def prepare_dataset(self):
        dataset = LineByLineTextDataset(tokenizer=self.tokenizer,
                                        file_path=os.path.join(self.hparams.data_path, self.hparams.data_train),
                                        block_size=self.hparams.max_len)
        train_size = int(self.hparams.valid_pct * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(dataset, [train_size, test_size])
        mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=self.hparams.mlm, mlm_probability=self.hparams.mlm_probability
        )
        self.collate_fn = mlm_collator.collate_batch

    def train_dataloader(self):
        self.prepare_dataset()
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
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers
        )

        return dataloader

    def test_dataloader(self):
        dataset = LineByLineTextDataset(tokenizer=self.tokenizer,
                                        file_path=os.path.join(self.hparams.data_path, self.hparams.data_test),
                                        block_size=self.hparams.max_len)

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers
        )

        return dataloader

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )


class LoggingCallback(pl.Callback):
    # def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    #     logger.info("***** Validation results *****")
    #     if pl_module.is_logger():
    #         metrics = trainer.callback_metrics
    #         # Log results
    #         for key in sorted(metrics):
    #             if key not in ["log", "progress_bar"]:
    #                 logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))
