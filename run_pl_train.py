import dataclasses
import datetime
import logging
import warnings
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers

from src.config_base import ModelArgs, TrainingArgs
from src.pl_model import XLMRobertaSeqClassificationPL
from src.utils import load_or_parse_args


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
