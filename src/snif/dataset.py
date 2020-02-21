from typing import Dict, List
import json

from overrides import overrides
from tqdm import tqdm

from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer

from snif.util import get_logger


log = get_logger(__name__)


def read_jsonlines(path):
    objects = []
    with open(path) as f:
        for line in f:
            objects.append(json.loads(line))
    return objects



@DatasetReader.register('fever')
class FeverReader(DatasetReader):
    def __init__(self, lazy: bool = False):
        super().__init__(lazy)
        self._tokenizer = PretrainedTransformerTokenizer(
            'bert-base-uncased', do_lowercase=True,
            start_tokens=[], end_tokens=[]
        )
        self._token_indexers = {'text': PretrainedBertIndexer('bert-base-uncased')}

    @overrides
    def _read(self, file_path):
        log.info(f"Reading instances from: {file_path}")
        examples = read_jsonlines(file_path)
        for claim in tqdm(examples):
            yield self.text_to_instance(
                claim['claim'],
                label=claim['label'],
                claim_id=claim['id']
            )

    @overrides
    def text_to_instance(self,
                         text: str,
                         label: str = None,
                         claim_id: int = None):
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label, label_namespace='claim_labels')
        fields['metadata'] = MetadataField({
            'claim_id': claim_id,
            'label': label,
        })
        return Instance(fields)
