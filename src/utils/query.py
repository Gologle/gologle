from __future__ import annotations
from heapq import heappush, heappushpop

from pydantic import BaseModel


class DocResult(BaseModel):
    id: str
    sim: float
    description: str

    def __lt__(self, other: DocResult):
        return self.sim < other.sim

    def __le__(self, other: DocResult):
        return self.sim <= other.sim

    def __eq__(self, other: DocResult):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class QueryResults(BaseModel):
    use_rank: bool = True
    max_length: int = 10
    docs: list[DocResult] = []

    def add_result(self, result: DocResult) -> None:
        """Adds results to the docs, represented by a heap if use_rank is used."""
        if not self.use_rank:
            self.docs.append(result)
        else:
            if len(self.docs) < self.max_length:
                heappush(self.docs, result)
            else:
                heappushpop(self.docs, result)

    @property
    def rank(self) -> list[DocResult]:
        if self.use_rank:
            return sorted(self.docs, reverse=True)
        else:
            return self.docs

    def __len__(self):
        return len(self.docs)
