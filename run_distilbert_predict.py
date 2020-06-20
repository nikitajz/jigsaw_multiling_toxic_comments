import dataclasses
import logging
import os
from argparse import Namespace

import pandas as pd
from torch.utils.data.dataloader import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertConfig

from src.config_pl import ModelArgs, TrainArgs
from src.dataset import ToxicMultilangDataset, TokenizerCollateFn
from src.trainer import predict_toxic
from src.utils import load_or_parse_args


def main():
    logger = logging.getLogger(__name__)

    m_args, tr_args = load_or_parse_args((ModelArgs, TrainArgs), verbose=True)
    hparams = Namespace(**dataclasses.asdict(m_args), **dataclasses.asdict(tr_args))

    device = "cuda:0"
    if isinstance(tr_args.gpus, int) and tr_args.gpus == 0:
        device = 'cpu'
    elif isinstance(tr_args.gpus, list):
        device = f"cuda:{tr_args.gpus[0]}"
    logger.info(f"Device: {device}")

    config = DistilBertConfig.from_pretrained(
        hparams.config_name if hparams.config_name else hparams.model_name_or_path,
        num_labels=2,
        cache_dir=hparams.cache_dir,
    )

    # tokenizer = DistilBertTokenizerFast.from_pretrained(
    #     hparams.tokenizer_name if hparams.tokenizer_name else hparams.model_name_or_path,
    #     cache_dir=hparams.cache_dir,
    #     use_fast=True
    # )
    model = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(hparams.output_dir, "model-finetuned"),
        from_tf=bool(".ckpt" in hparams.model_name_or_path),
        config=config,
        cache_dir=hparams.cache_dir,
    )
    model = model.to(device)

    logger.info('Loading dataset for prediction')
    test_dataset = ToxicMultilangDataset(hparams.data_path, "test.csv",
                                         kind="test",
                                         resample=False)

    collate_fn = TokenizerCollateFn(max_tokens=hparams.max_len,
                                    tokenizer_name=hparams.tokenizer_name,
                                    cache_dir=hparams.cache_dir,
                                    target=False)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=hparams.num_workers
    )

    logger.info('Predicting test set')

    preds = predict_toxic(model, test_dataloader, device)

    submit = pd.read_csv('data/sample_submission.csv')
    submit['toxic'] = preds
    logger.info(f"There are {(preds == 1).sum()}/{preds.shape[0]} positive samples")
    submit.to_csv('data/submit.csv', index=False)


if __name__ == "__main__":
    main()
