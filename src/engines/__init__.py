"""Implementation of different engines for the Retrieval Information System."""
from abc import ABC, abstractmethod
from time import time
from pathlib import Path

from sqlmodel import create_engine

from src.parsers import DatasetParser
from src.utils import QueryResults, TimeLogger


class Engine(ABC):

    def __init__(self, name: str, dataset: DatasetParser):
        self.name = name
        self.dataset = dataset
        self.db = Path(f"{self.dataset.name}({self.name}).db")
        self.db_engine = create_engine("sqlite:///" + self.db.name)

    @abstractmethod
    def answer(self, query: str, max_length: int) -> QueryResults:
        """Answers a query to the user."""

    def predict_labels(self, query: str) -> list[str]:
        raise NotImplementedError()

    def update_index(self):
        """Indexes all the documents of the dataset. This is an expensive method
        that must be called only if new documents are added."""
        with TimeLogger(
            enter_msg=f"Creating index for {self.name} ({self.dataset.name})...",
            exit_msg=(
                f"Created index for {self.name} ({self.dataset.name}). "
                f"Took %F seconds."
            )
        ):
            self._update_index()

    @abstractmethod
    def _update_index(self):
        pass
