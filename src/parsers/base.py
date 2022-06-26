from typing import Iterator
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer

from src.utils import DocResult


DATASETS_ROOT = Path("./datasets")


class DatasetEntry(ABC):

    def __init__(self, id_: str):
        self.id = id_

    @property
    @abstractmethod
    def raw_text(self) -> str:
        """This property is intended to return the raw text of the document for
        this parsed entry"""

    @property
    @abstractmethod
    def main_content(self):
        """Relevant text of the document."""


class DatasetParser(ABC):

    root: Path = DATASETS_ROOT

    def __init__(self, data: Path, count_vzer: CountVectorizer, total: int):
        self.data = data
        self.count_vzer = count_vzer
        self.total = total
        self.entries: list[DatasetEntry] = []

    @property
    def name(self) -> str:
        return self.data.name

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetEntry]:
        """Iteration over datasets must return instances of entries parsed"""

    @abstractmethod
    def fit_transform(self):
        """Calls the method fit_transform of this instance CountVectorizer over
        the data parsed.

        Returns:
            Document-term matrix.
        """

    @abstractmethod
    def get_test_cases(self) -> dict[str, list[DocResult]]:
        """Gets a map of queries with its expected documents as result for the
        dataset.

        Returns:
            A dict mapping a query to its list of relevant documents.
        """