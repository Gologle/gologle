from __future__ import annotations
from typing import Iterator
from pathlib import Path
from abc import ABC, abstractmethod


DATASETS_ROOT = Path("./datasets")


class DatasetEntry(ABC):

    @property
    @abstractmethod
    def raw_text(self) -> str:
        """This property is intended to return the raw text of the document for
        this parsed entry"""


class DatasetParser(ABC):

    root: Path = DATASETS_ROOT

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetEntry]:
        """Iteration over datasets must return instances of entries parsed"""
