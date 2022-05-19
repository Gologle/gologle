from nltk.corpus import reuters

from typing import Iterator

from sklearn.feature_extraction.text import CountVectorizer

from .base import DatasetEntry, DatasetParser


class ReutersEntry(DatasetEntry):

    def __init__(self, id : str):

        full_id = reuters.fileids(id)
        split_id = full_id.split('/')

        self._set = split_id[0]
        self._categories = reuters.categories(id)
        self._raw = reuters.raw(id)

        super(ReutersEntry, self).__init__(split_id[1])

    @property
    def raw_text(self):
        return self._raw

    @property
    def set(self):
        return self._set

    @property
    def categories(self):
        return self._categories



class ReutersParser(DatasetParser):
    """Pseudo parser for the Reuters dataset"""

    def __init__(self):
        super(ReutersParser, self).__init__(
            data=self.root / "reuters-21578", #useless?
            count_vzer=CountVectorizer(
                input="content",
                decode_error="ignore",
                stop_words="english"
            ),
            total=10788
        )

        self.entries: list[ReutersEntry] = []

        for id in reuters.fileids():
                self.entries.append(ReutersEntry(id))

        assert len(self.entries) == self.total

    def __iter__(self) -> Iterator[ReutersEntry]:
        return iter(self.entries)

    def fit_transform(self):
        return self.count_vzer.fit_transform(
            tuple(entry.raw_text for entry in self)
        )