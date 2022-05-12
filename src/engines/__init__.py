"""Implementation of different engines for the Retrieval Information System."""
from abc import ABC, abstractmethod
from time import time

from src.parsers import DatasetParser
from src.utils import QueryResults


class Engine(ABC):

    def __init__(self, name: str, dataset: DatasetParser):
        self.name = name
        self.dataset = dataset

    @abstractmethod
    def answer(self, query: str, max_length: int) -> QueryResults:
        """Answers a query to the user."""

    def update_index(self):
        """Indexes all the documents of the dataset. This is an expensive method
        that must be called only if new documents are added."""
        ts = time()
        print(f"Creating index for {self.name} ({self.dataset.name})...")
        self._update_index()
        print(
            f"Created index for {self.name} ({self.dataset.name}). "
            f"Took {round((time() - ts), 2)} seconds."
        )

    @abstractmethod
    def _update_index(self):
        pass
