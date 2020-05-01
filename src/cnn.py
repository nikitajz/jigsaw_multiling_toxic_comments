import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNModel(nn.Module):
    def __init__(self, emb_vectors, kernel_sizes, num_channels, hidden_size, dropout_p=0.5, pad_idx=1):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb_vectors, padding_idx=pad_idx, sparse=True)
        self.convs = nn.ModuleList()
        in_channels = emb_vectors.shape[1]
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

    def forward(self, x):
        x_emb = self.emb(x).permute(0,2,1)
        pooled_list = []
        for i in range(len(self.convs)):
            cnn_out = self.convs[i](x_emb)
            pooled, _ = cnn_out.max(dim=2)
            pooled_list.append(pooled)
        
        cat = torch.cat(pooled_list, dim=1)
        return self.projection(cat).squeeze(1)
