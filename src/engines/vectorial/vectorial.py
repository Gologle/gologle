from typing import Iterable
from math import sqrt
from time import time

from sqlmodel import SQLModel, Session, select
from sklearn.feature_extraction.text import TfidfTransformer

from src.engines import Engine
from src.parsers import DatasetParser
from src.utils import DocResult, QueryResults, get_terms, DatabaseBatchCommit
from .models import (
    Term,
    Document,
    InverseDocumentFrequency,
    WeightTermDocument,
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

        if not self.db.is_file():
            self.update_index()

    def _update_index(self) -> None:
        """Builds the index of the engine in a sqlite database."""
        SQLModel.metadata.create_all(self.db_engine)

        X = self.dataset.fit_transform()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)

        with DatabaseBatchCommit(self.db_engine) as batcher:
            # add the documents to the db for faster access
            ts = time()
            print("Adding Documents to db... ", end="")
            for entry in self.dataset:
                batcher.add(Document(id=entry.id, text=entry.raw_text))
            print(f"Done, took {round(time() - ts, 2)} seconds.")

            print("Adding Terms and IDFs to db... ", end="")
            ts = time()
            for term in self.dataset.count_vzer.get_feature_names_out():
                term_index = self.dataset.count_vzer.vocabulary_[term]

                # store the term and its idf
                batcher.add(Term(id=term))
                batcher.add(InverseDocumentFrequency(
                    term_id=term,
                    value=transformer.idf_[term_index]
                ))
            print(f"Done, took {round(time() - ts, 2)} seconds.")

            print("Adding Weights to db... ", end="")
            ts = time()
            inv_feat = {v: k for k, v in self.dataset.count_vzer.vocabulary_.items()}
            for i, x in enumerate(tfidf):
                for j, v in zip(x.indices, x.data):
                    batcher.add(WeightTermDocument(
                        term_id=inv_feat[j],
                        document_id=self.dataset.entries[i].id,
                        value=v
                    ))
            print(f"Done, took {round(time() - ts, 2)} seconds.")

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
                db_term = session.exec(statement).first()
                query_idf[term] = 0 if db_term is None else db_term.value

        query_weights = {
            term: (self.softened + (1 - self.softened) * query_ntf[term]) * query_idf[term]
            for term in query_terms
        }

        results = QueryResults(max_length=max_length)
        with Session(self.db_engine) as session:
            for doc in session.exec(select(Document)):
                sim = self._sim(query_weights, doc)
                if sim > 0:
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
                weight_term_doc = session.exec(
                    select(WeightTermDocument).where(
                        WeightTermDocument.term_id == query_term,
                        WeightTermDocument.document_id == doc.id
                    )
                ).first()
                if weight_term_doc is None:
                    doc_weights[query_term] = 0
                else:
                    doc_weights[query_term] = weight_term_doc.value

        prod_vectors = sum(doc_weights[term] * query_weights[term] for term in query_weights)
        prod_norm_vectors = (
            sqrt(sum(weight ** 2 for weight in doc_weights.values())) +
            sqrt(sum(weight ** 2 for weight in query_weights.values()))
        )

        try:
            sim = prod_vectors / prod_norm_vectors
        except ZeroDivisionError:
            sim = 0

        return sim
