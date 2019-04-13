from typing import Dict
import logging
import os

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, TokenCharactersIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("name_dataset_reader")
class NamesDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {'tokens': TokenCharactersIndexer()}

    @overrides
    def _read(self, folder_path):
        files = os.walk(folder_path)
        for root, _, names in files:
            for name in names:
                lang = name.split('.')[0]
                with open(os.path.join(root, name), encoding='utf-8') as f:
                    for line in f:
                        name = line.strip()
                        yield self.text_to_instance(name, lang)

    @overrides
    def text_to_instance(self, name: str, lang: str) -> Instance:
        tokenized_name = self._tokenizer.tokenize(name)
        name_field = TextField(tokenized_name, self._token_indexers)
        fields = {'name': name_field}
        if lang is not None:
            fields['label'] = LabelField(lang)
        return Instance(fields)