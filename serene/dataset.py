from typing import Dict, Optional, List
import random

from overrides import overrides
from tqdm import tqdm
from pedroai.io import read_jsonlines

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from serene.util import get_logger
from serene.wiki_db import WikiDatabase


log = get_logger(__name__)


def get_evidence(db: WikiDatabase, page_set: List[str], evidence_sets) -> Optional[str]:
    for ev_set in evidence_sets:
        for _, _, page, sent_id in ev_set:
            if page is not None:
                page = page.replace("_", " ")
                maybe_evidence = db.get_page_sentence(page, sent_id)
                if maybe_evidence is not None:
                    return maybe_evidence
    # If no evidence is found, should grab something random
    while True:
        page = random.choice(page_set)
        maybe_evidence = db.get_page_sentence(page, 0)
        if maybe_evidence is not None:
            return maybe_evidence


@DatasetReader.register("fever")
class FeverReader(DatasetReader):
    def __init__(self, transformer: str, include_evidence: bool, lazy: bool = False):
        super().__init__(lazy)
        self._include_evidence = include_evidence
        self._tokenizer = PretrainedTransformerTokenizer(transformer)
        self._claim_indexers = {
            "claim_tokens": PretrainedTransformerIndexer(transformer),
        }
        if include_evidence:
            self._evidence_indexers = {
                "evidence_tokens": PretrainedTransformerIndexer(transformer),
            }
        else:
            self._evidence_indexers = None

    @overrides
    def _read(self, file_path):
        log.info(f"Reading instances from: {file_path}")
        db = WikiDatabase()
        page_set = db.get_wikipedia_urls()
        examples = read_jsonlines(file_path)
        for ex in tqdm(examples):
            if self._include_evidence:
                evidence = get_evidence(db, page_set, ex["evidence"])
            else:
                evidence = None
            yield self.text_to_instance(
                ex["claim"],
                label=ex["label"],
                claim_id=ex["id"],
                evidence_text=evidence,
            )

    @overrides
    def text_to_instance(
        self,
        claim_text: str,
        evidence_text: str = None,
        label: str = None,
        claim_id: int = None,
    ):
        fields: Dict[str, Field] = {}
        tokenized_claim = self._tokenizer.tokenize(claim_text)
        if evidence_text is not None:
            tokenized_evidence = self._tokenizer.tokenize(evidence_text)
        else:
            tokenized_evidence = None
        fields["claim_tokens"] = TextField(
            tokenized_claim, token_indexers=self._claim_indexers
        )
        if tokenized_evidence is not None:
            fields["evidence_tokens"] = TextField(
                tokenized_evidence, token_indexers=self._evidence_indexers
            )
        if label is not None:
            fields["label"] = LabelField(label, label_namespace="claim_labels")
        fields["metadata"] = MetadataField({"claim_id": claim_id, "label": label,})
        return Instance(fields)
