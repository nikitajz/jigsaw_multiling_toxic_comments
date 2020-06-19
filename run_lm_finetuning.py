import dataclasses
import logging
import os
import warnings
from argparse import Namespace
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers

from src.config_pl import ModelArgs, TrainArgs
from src.pl_distilbert_mlm import DistilBERTMaskedLM, LoggingCallback
from src.utils import load_or_parse_args, seed_everything

warnings.filterwarnings('ignore', category=UserWarning)


def main():
    logger = logging.getLogger(__name__)
    m_args, tr_args = load_or_parse_args((ModelArgs, TrainArgs), verbose=True)
    hparams = Namespace(**dataclasses.asdict(m_args), **dataclasses.asdict(tr_args))

    # init model
    seed_everything(hparams.seed)
    logger.info(f"Effective batch size: {hparams.effective_batch_size}")

    # if os.path.exists(hparams.output_dir) and os.listdir(hparams.output_dir) and hparams.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(hparams.output_dir))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=hparams.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    if not hparams.load_checkpoint:
        logger.info("Creating the model and trainer.")
        model = DistilBERTMaskedLM(hparams, mode=hparams.model_mode)
        logs_dir = os.path.join(hparams.output_dir, "pl_logs")
        os.makedirs(logs_dir, exist_ok=True)

        pl_loggers = [loggers.TestTubeLogger(save_dir=hparams.output_dir, name="test_tube", create_git_tag=True)]
        if hparams.wandb_enable:
            dt_now_str = datetime.strftime(datetime.now(), "%y%m%d_%H%M")
            wandb_experiment_name = hparams.model_name_or_path.split("-")[0] + "@" + dt_now_str
            pl_loggers.append(loggers.WandbLogger(name=wandb_experiment_name, project='jigsaw_multilang'))

        trainer = pl.Trainer.from_argparse_args(hparams,
                                                # fast_dev_run=True,
                                                # auto_lr_find=True,
                                                logger=pl_loggers,
                                                progress_bar_refresh_rate=100,
                                                # distributed_backend="dp",
                                                checkpoint_callback=checkpoint_callback,
                                                default_root_dir=logs_dir,
                                                callbacks=[LoggingCallback()]
                                                )
    else:
        logger.info("Loading the model and trainer from checkpoint:", hparams.model_checkpoint_path)
        model = DistilBERTMaskedLM.load_from_checkpoint(hparams.model_checkpoint_path)
        trainer = pl.Trainer.resume_from_checkpoint(hparams.model_checkpoint_path)

    if hparams.do_train:
        trainer.fit(model)
        trainer.save_checkpoint(os.path.join(hparams.output_dir, "trainer.pt"))
        out_dir = os.path.join(hparams.output_dir, "model-finetuned")
        os.makedirs(out_dir, exist_ok=True)
        model.model.save_pretrained(out_dir)
        model.tokenizer.save_pretrained(out_dir)
        # https://github.com/huggingface/transformers/issues/5081
        # model.tokenizer.save_pretrained(out_dir)

    if hparams.do_test:
        trainer.test(model)

    return trainer


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    main()
