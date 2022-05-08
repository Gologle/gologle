from typing import Iterator
from abc import ABC, abstractmethod
from pathlib import Path


DATASETS_ROOT = Path("./datasets")


class DatasetEntry(ABC):

    def __init__(self, id_: str):
        self.id = id_

    @property
    @abstractmethod
    def raw_text(self) -> str:
        """This property is intended to return the raw text of the document for
        this parsed entry"""


class DatasetParser(ABC):

    root: Path = DATASETS_ROOT

    def __init__(self, data: Path, total: int):
        self.data = data
        self.total = total

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetEntry]:
        """Iteration over datasets must return instances of entries parsed"""
