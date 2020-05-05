import logging 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNMultiling(nn.Module):
    def __init__(self, emb_vectors_dict, kernel_sizes, num_channels, hidden_size, dropout_p=0.5, pad_idx=1):
        super().__init__()
        self.logger = logging.getLogger()
        self.cur_lang = None # track last language used for embeddings, required for memory optimization
        self.embs = {}
        in_channels = emb_vectors_dict['en'].shape[1]
        for lang, vect in emb_vectors_dict.items():
            assert in_channels == vect.shape[1], f"Vector for language [{lang}] has shape {vect.shape[1]}, but should be {in_channels}"
            self.embs[lang] = nn.Embedding.from_pretrained(emb_vectors_dict[lang], padding_idx=pad_idx, freeze=True, sparse=True)
        self.convs = nn.ModuleList()
        # in_channels = num_channels
        for ks in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels,
                              out_channels=num_channels, kernel_size=ks),
                    nn.BatchNorm1d(num_channels),
                    nn.ReLU()  # ,
                    # nn.Dropout(dropout_p)
                )
            )
            in_channels = num_channels

        self.projection = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features=num_channels * len(kernel_sizes), out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, x, lang):
        #TODO: find more elegant solution for using lang (not list in batch)
        if isinstance(lang, list):
            lang = lang[0]
        if self.cur_lang is None:
            self.cur_lang = lang 

        self._emb_to_device(x, lang)

        x_emb = self.embs[self.cur_lang](x).permute(0,2,1)
        pooled_list = []
        for i in range(len(self.convs)):
            cnn_out = self.convs[i](x_emb)
            pooled, _ = cnn_out.max(dim=2)
            pooled_list.append(pooled)
        
        cat = torch.cat(pooled_list, dim=1)
        return self.projection(cat).squeeze(1)

    def _emb_to_device(self, x, lang):
        """It's quite memory-consuming to store embeddings for all languages on gpu, hence dynamic loading/unloading required.

        Parameters
        ----------
        x : [torch.Tensor]
            Input tensor (data)
        lang : [str]
            Language of current input tensor
        """
        if x.device != self.embs[lang].weight.device:           
            if self.cur_lang != lang:
                self.logger.debug(f"Unload embedding for lang [{self.cur_lang}] from cuda")
                self.embs[self.cur_lang].to('cpu')
                self.cur_lang = lang
            self.logger.debug(f'Moving embedding for language [{lang}] to device {x.device}')
            self.embs[self.cur_lang].to(x.device)