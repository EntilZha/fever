from typing import Dict, Text

import numpy as np
import torch
from torch import nn

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from serene.util import get_logger


log = get_logger(__name__)
N_TRAIN = 145449
N_TRAIN_SUPPORT = 80035
N_TRAIN_REFUTE = 29775
N_TRAIN_NEI = 35639
BASELINE_RATIO = 1 / 3
SUPPORT_RATIO = N_TRAIN_SUPPORT / N_TRAIN
REFUTE_RATIO = N_TRAIN_REFUTE / N_TRAIN
NEI_RATIO = N_TRAIN_NEI / N_TRAIN


@Model.register("claim_only")
class ClaimOnlyModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        dropout: float,
        transformer: str,
        pool: Text,
        label_namespace: str = "claim_labels",
    ):
        super().__init__(vocab)
        self._pool = pool
        self._claim_embedder = BasicTextFieldEmbedder({
            "claim_tokens": PretrainedTransformerEmbedder(transformer)
        })
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._claim_embedder.get_output_dim(), self._num_labels),
        )
        self._accuracy = CategoricalAccuracy()
        weights = {
            "SUPPORTS": BASELINE_RATIO / SUPPORT_RATIO,
            "REFUTES": BASELINE_RATIO / REFUTE_RATIO,
            "NOT ENOUGH INFO": BASELINE_RATIO / NEI_RATIO,
        }
        weight_array = []
        for idx in range(3):
            class_name = self.vocab.get_token_from_index(idx, namespace=label_namespace)
            weight_array.append(weights[class_name])
            log.info(f"Class weight: {class_name}={weights[class_name]}")
        weight_array = torch.from_numpy(np.array(weight_array)).float()
        self._loss = torch.nn.CrossEntropyLoss(weight=weight_array)

    def forward(
        self,
        claim_tokens: Dict[str, torch.LongTensor],
        metadata=None,
        label: torch.IntTensor = None,
    ):
        claim_embeddings = self._claim_embedder(claim_tokens)[:, 0, :]
        logits = self._classifier(claim_embeddings)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {
            "logits": logits,
            "probs": probs,
            "preds": torch.argmax(logits, 1),
        }

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
