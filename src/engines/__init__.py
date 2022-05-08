"""Implementation of different engines for the Retrieval Information System."""
from abc import ABC, abstractmethod

from src.parsers import DatasetParser
from src.utils import QueryResults


class Engine(ABC):

    def __init__(self, name: str, dataset: DatasetParser):
        self.name = name
        self.dataset = dataset

    @abstractmethod
    def answer(self, query: str) -> QueryResults:
        """Answers a query to the user."""

    @abstractmethod
    def update_index(self):
        """Indexes all the documents of the dataset. This is an expensive method
        that must be called only if new documents are added."""