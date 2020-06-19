import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TokenizerCollateFn:
    def __init__(self, max_tokens=512, tokenizer_name="xlm-roberta-base", cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            use_fast=True
        )
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
            self.data = pd.read_csv(path / filenames, usecols=[self.column.text])  # , self.column.lang

        pos_samples = self.data[self.data[self.column.target] == 1].shape[0]
        self.logger.info(f"{kind.capitalize()} dataset shape: ({','.join(map(str, self.data.shape))})."
                         f" Target positive samples ratio: {pos_samples / self.data.shape[0]:3f}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row[self.column.text], row.get(self.column.target, None)

    def __len__(self):
        return self.data.shape[0]
