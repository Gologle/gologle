from typing import Iterable
from pathlib import Path
from collections import defaultdict
from math import log, sqrt

from sqlmodel import SQLModel, Session, create_engine, select

from src.engines import Engine
from src.parsers import DatasetParser
from src.utils import DocResult, QueryResults, get_terms
from .models import (
    Term,
    Document,
    NormTermFrequency,
    InverseDocumentFrequency,
)


class VectorialModel(Engine):

    def __init__(self, dataset: DatasetParser, softened: float = 0.5):
        """
        Args:
            dataset: dataset with the entries parsed
            softened: value for soft the contribution of the frequency of a
                term in a query. Use to be 0.4 or 0.5.

        Raises:
            ValueError: the softened is not a value between 0 and 1.
        """
        if softened < 0 or softened > 1:
            raise ValueError(f"Softened must be between 0 and 1")

        super(VectorialModel, self).__init__("Vectorial Model", dataset)

        self.softened = softened
        self.db = Path(f"{self.dataset.name}({self.name}).db")
        update_db = not self.db.is_file()
        self.db_engine = create_engine("sqlite:///" + self.db.name)

        if update_db:
            self.update_index()

    def _update_index(self) -> None:
        """Builds the index of the engine in a sqlite database."""
        SQLModel.metadata.create_all(self.db_engine)

        idf_count: dict[str, int] = defaultdict(lambda: 0)

        with Session(self.db_engine) as session:
            seen_terms = set()
            for entry in self.dataset:
                # cache this property in case that is reading from disk
                text = entry.raw_text

                # add the document to the db for faster access
                session.add(Document(id=entry.id, text=text))

                terms = get_terms(text)
                for term, ntf in VectorialModel._ntf(terms).items():
                    # add terms of the document to the db
                    if term not in seen_terms:
                        seen_terms.add(term)
                        session.add(Term(id=term))

                    # add normalized term frequency
                    session.add(NormTermFrequency(
                        term_id=term,
                        document_id=entry.id,
                        value=ntf
                    ))

                    # count the occurrence for idf
                    idf_count[term] += 1

            # store the idf for every term
            for term, count in idf_count.items():
                session.add(InverseDocumentFrequency(
                    term_id=term,
                    value=log(self.dataset.total / count)
                ))

            session.commit()

    @staticmethod
    def _ntf(terms: Iterable[str]) -> dict[str, float]:
        """Calculates the normalized term frequency (ntf) for every term in terms.

        Returns:
            A dict containing every term mapped to its ntf value
        """
        term_count = {}
        for term in terms:
            try:
                term_count[term] += 1
            except KeyError:
                term_count[term] = 1

        try:
            max_term = max(term_count.values())
        except ValueError:  # max() arg is an empty sequence
            return {}

        return {term: count / max_term for term, count in term_count.items()}

    def answer(self, query: str, max_length: int) -> QueryResults:
        query_terms = get_terms(query)
        query_ntf = VectorialModel._ntf(query_terms)
        query_idf: dict[str, float] = {}
        with Session(self.db_engine) as session:
            for term in query_terms:
                statement = select(InverseDocumentFrequency).where(
                    InverseDocumentFrequency.term_id == term
                )
                idf = session.exec(statement).first().value
                query_idf[term] = idf

        query_weights = {
            term: (self.softened + (1 - self.softened) * query_ntf[term]) * query_idf[term]
            for term in query_terms
        }

        results = QueryResults(ranking=True, max_length=max_length)
        with Session(self.db_engine) as session:
            for doc in session.exec(select(Document)):
                sim = self._sim(query_weights, doc)
                results.add_result(DocResult(
                    id=doc.id,
                    sim=sim,
                    description=""
                ))

        return results

    def _sim(self, query_weights: dict[str, float], doc: Document) -> float:
        """Calculates the similarity between a query and a document.

        Args:
            query_weights: terms mapped to its weight in the query.
            doc: document instance fetched from the index.

        Returns:
            A value indicating the similarity between the query and the document.
        """
        doc_weights: dict[str, float] = {}
        with Session(self.db_engine) as session:
            for query_term in query_weights:
                select_ntf = select(NormTermFrequency).where(
                    NormTermFrequency.term_id == query_term,
                    NormTermFrequency.document_id == doc.id
                )
                select_idf = select(InverseDocumentFrequency).where(
                    InverseDocumentFrequency.term_id == query_term
                )
                ntf = session.exec(select_ntf).first()
                idf = session.exec(select_idf).first()
                if ntf is None:
                    doc_weights[query_term] = 0
                else:
                    doc_weights[query_term] = ntf.value * idf.value

        prod_vectors = sum(doc_weights[term] * query_weights[term] for term in query_weights)
        prod_norm_vectors = (
            sqrt(sum(weight ** 2 for weight in doc_weights.values())) +
            sqrt(sum(weight ** 2 for weight in query_weights.values()))
        )

        return prod_vectors / prod_norm_vectors
