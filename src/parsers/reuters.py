from nltk.corpus import reuters

from typing import Iterator

from sklearn.feature_extraction.text import CountVectorizer

from .base import DatasetEntry, DatasetParser


class ReutersEntry(DatasetEntry):

    def __init__(self, full_id: str):
        _, id_ = full_id.split("/")

        super(ReutersEntry, self).__init__(id_)

        self._categories = reuters.categories(full_id)
        self._raw = reuters.raw(full_id)

    @property
    def raw_text(self) -> str:
        return self._raw

    @property
    def main_content(self) -> str:
        return self._raw

    @property
    def labels(self) -> list[str]:
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

        for full_id in reuters.fileids():
            self.entries.append(ReutersEntry(full_id))

        assert len(self.entries) == self.total

    def __iter__(self) -> Iterator[ReutersEntry]:
        return iter(self.entries)

    def fit_transform(self):
        return self.count_vzer.fit_transform(
            tuple(entry.raw_text for entry in self)
        )

    def get_test_cases(self):
        raise NotImplementedError()
