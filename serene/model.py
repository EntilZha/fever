from typing import Dict, Text

import numpy as np
import torch
from torch import nn  # type: ignore

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import TimeDistributed

from serene.util import get_logger
from serene.metrics import Recall


log = get_logger(__name__)
N_TRAIN = 145449
N_TRAIN_SUPPORT = 80035
N_TRAIN_REFUTE = 29775
N_TRAIN_NEI = 35639
BASELINE_RATIO = 1 / 3
SUPPORT_RATIO = N_TRAIN_SUPPORT / N_TRAIN
REFUTE_RATIO = N_TRAIN_REFUTE / N_TRAIN
NEI_RATIO = N_TRAIN_NEI / N_TRAIN


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
        self._claim_embedder = BasicTextFieldEmbedder(
            {"claim_tokens": PretrainedTransformerEmbedder(transformer)}
        )
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
        torch_weights = torch.from_numpy(np.array(weight_array)).float()
        self._loss = torch.nn.CrossEntropyLoss(weight=torch_weights)

    def forward(
        self,
        claim_tokens: Dict[str, torch.LongTensor],
        evidence_tokens=None,
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


class FeverVerifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        dropout: float,
        transformer: str,
        pool: Text,
        classifier_dim: int = 200,
        in_batch_negatives: bool = False,
        label_namespace: str = "claim_labels",
    ):
        super().__init__(vocab)
        self._pool = pool
        self._in_batch_negatives = in_batch_negatives
        self._label_namespace = label_namespace
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._claim_embedder = BasicTextFieldEmbedder(
            {"claim_tokens": PretrainedTransformerEmbedder(transformer)}
        )
        self._evidence_embedder = BasicTextFieldEmbedder(
            {"evidence_tokens": PretrainedTransformerEmbedder(transformer)}
        )
        self._dropout = nn.Dropout(dropout)
        classifier = nn.Sequential(
            nn.Linear(self._claim_embedder.get_output_dim(), classifier_dim),
            nn.GELU(),
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, self._num_labels),
        )
        self._accuracy = CategoricalAccuracy()
        weights = {
            "SUPPORTS": BASELINE_RATIO / SUPPORT_RATIO,
            "REFUTES": BASELINE_RATIO / REFUTE_RATIO,
            "NOT ENOUGH INFO": BASELINE_RATIO / NEI_RATIO,
        }
        self._nei_idx = self.vocab.get_token_index(
            "NOT ENOUGH INFO", namespace=label_namespace
        )
        if self._in_batch_negatives:
            self._loss = nn.CrossEntropyLoss()
            self._classifier = TimeDistributed(classifier)
            self._in_batch_accuracy = CategoricalAccuracy()
        else:
            self._classifier = classifier
            weight_array = []
            for idx in range(3):
                class_name = self.vocab.get_token_from_index(
                    idx, namespace=label_namespace
                )
                weight_array.append(weights[class_name])
                log.info(f"Class weight: {class_name}={weights[class_name]}")
            torch_weights = torch.from_numpy(np.array(weight_array)).float()
            self._loss = nn.CrossEntropyLoss(weight=torch_weights)
            self._in_batch_accuracy = None

    def _in_batch_loss(
        self, claim_embeddings, evidence_embeddings, label: torch.IntTensor = None
    ):
        batch_size = claim_embeddings.shape[0]
        logits = self._classifier(
            torch.einsum("ai,bi->abi", [claim_embeddings, evidence_embeddings])
        )
        ix = torch.arange(
            0, batch_size, dtype=torch.long, device=claim_embeddings.device
        )
        normal_logits = logits[ix, ix].view(-1, 3)
        normal_probs = torch.nn.functional.softmax(normal_logits, dim=-1)
        output_dict = {
            "logits": normal_logits,
            "probs": normal_probs,
            "preds": torch.argmax(normal_logits, 1),
        }

        if label is not None:
            label = label.long()
            all_labels = torch.full(
                (batch_size, batch_size),
                self._nei_idx,
                dtype=label.dtype,
                device=label.device,
            )
            all_labels[ix, ix] = label
            flattened_logits = logits.view(-1, 3)
            all_labels = all_labels.view(-1)
            loss = self._loss(flattened_logits, all_labels)
            output_dict["loss"] = loss
            self._in_batch_accuracy(flattened_logits, all_labels)
            self._accuracy(normal_logits, label)
        return output_dict

    def forward(
        self,
        claim_tokens: Dict[str, torch.LongTensor],
        evidence_tokens: Dict[str, torch.LongTensor],
        metadata=None,
        label: torch.IntTensor = None,
    ):
        claim_embeddings = self._dropout(self._claim_embedder(claim_tokens)[:, 0, :])
        evidence_embeddings = self._dropout(
            self._evidence_embedder(evidence_tokens)[:, 0, :]
        )
        if self._in_batch_negatives:
            return self._in_batch_loss(
                claim_embeddings, evidence_embeddings, label=label
            )
        logits = self._classifier(claim_embeddings * evidence_embeddings)
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
        if self._in_batch_negatives:
            metrics["in_batch_accuracy"] = self._in_batch_accuracy.get_metric(reset)
        return metrics

    def make_output_human_readable(self, output_dict):
        preds = [
            self.vocab.get_token_from_index(idx.item(), namespace=self._label_namespace)
            for idx in output_dict["preds"].cpu()
        ]
        output_dict["pred_readable"] = preds
        return output_dict


class FeverEvidenceRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        dropout: float,
        transformer: str,
        label_namespace: str = "claim_labels",
    ):
        super().__init__(vocab)
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._claim_embedder = BasicTextFieldEmbedder(
            {"claim_tokens": PretrainedTransformerEmbedder(transformer)}
        )
        self._evidence_embedder = BasicTextFieldEmbedder(
            {"evidence_tokens": PretrainedTransformerEmbedder(transformer)}
        )
        self._dropout = nn.Dropout(dropout)
        self._accuracy = BooleanAccuracy()
        self._recall = Recall()
        self._loss = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        claim_tokens: Dict[str, torch.LongTensor],
        evidence_tokens: Dict[str, torch.LongTensor],
        metadata=None,
        label: torch.IntTensor = None,
    ):
        emb_dim = self._claim_embedder.get_output_dim()
        claim_embeddings = self._dropout(
            self._claim_embedder(claim_tokens)[:, 0, :]
        ).view(-1, emb_dim)
        evidence_embeddings = self._dropout(
            self._evidence_embedder(evidence_tokens)[:, 0, :]
        ).view(-1, emb_dim)

        # Compute in batch negatives scoring
        # a=b=batch_size
        # repeated i is significant
        # (batch_size, batch_size)
        batch_size = claim_embeddings.shape[0]
        logits = torch.einsum("ai,bi->ab", [claim_embeddings, evidence_embeddings])
        # (batch_size, batch_size)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        # Create a diagonal matrix of ones
        labels = torch.eye(batch_size, batch_size).float().to(logits.device)
        # Create labels, which are by construction on the diagonal (paired claim/evidence)
        output_dict = {"logits": logits, "probs": probs, "preds": preds}

        if label is not None:
            loss = self._loss(logits, labels)
            output_dict["loss"] = loss
            metric_logits = logits.view(-1, 1).long()
            metric_labels = labels.view(-1, 1).long()
            metric_preds = preds.view(-1, 1).long()
            self._accuracy(metric_logits, metric_labels)
            self._recall(metric_preds, metric_labels)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
            "recall": self._recall.get_metric(reset),
        }
        return metrics
