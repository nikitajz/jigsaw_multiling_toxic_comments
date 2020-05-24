import torch
import torch.nn as nn

from transformers import XLMRobertaModel

class ToxicXLMRobertaModel(nn.Module):

    def __init__(self, backbone_path, num_labels=2, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels

        self.roberta = XLMRobertaModel.from_pretrained(backbone_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(
            in_features=self.roberta.pooler.dense.out_features*2,
            out_features=2,
        )

    def forward(self, 
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,):
        bs, seq_length = input_ids.shape

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        seq_out = outputs[0]

        apool = torch.mean(seq_out, 1)
        mpool, _ = torch.max(seq_out, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        logits = self.classifier(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

            return loss, logits
        return (logits,)
