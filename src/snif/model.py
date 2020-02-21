from typing import Dict, Text

import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.nn import util


@Model.register('claim_only')
class ClaimOnlyModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 dropout: float,
                 pool: Text,
                 label_namespace: str = "claim_labels"):
        super().__init__(vocab)
        self._pool = pool
        self._bert = PretrainedBertEmbedder('bert-base-uncased', requires_grad=True)
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._bert.get_output_dim(), self._num_labels),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                text: Dict[str, torch.LongTensor],
                metadata = None,
                label: torch.IntTensor = None):
        input_ids: torch.LongTensor = text['text']
        # Grab the representation of CLS token, which is always first
        if self._pool == 'cls':
            bert_emb = self._bert(input_ids)[:, 0, :]
        elif self._pool == 'mean':
            mask = (input_ids != 0).long()[:, :, None]
            bert_seq_emb = self._bert(input_ids)
            bert_emb = util.masked_mean(bert_seq_emb, mask, dim=1)
        else:
            raise ValueError('Invalid config')

        logits = self._classifier(bert_emb)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {
            'logits': logits,
            'probs': probs,
            'preds': torch.argmax(logits, 1)
        }

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict['loss'] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics
