from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sqlmodel import SQLModel, Session, select
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from src.utils.functions import lemmatize_query, lemmatize_word
from src.engines import Engine
from src.parsers import DatasetParser
from src.utils import DatabaseBatchCommit, TimeLogger, QueryResults, DocResult
from src.engines.models import Document, Feedback, LabeledDoc


class Doc2VecModel(Engine):

    def __init__(self, dataset: DatasetParser, model_path: Path | None = None):
        """
        Args:
            dataset: dataset with the entries parsed
        """
        super(Doc2VecModel, self).__init__("Doc2Vec Model", dataset)

        if model_path is None:
            self.model_path = Path(f"{self.dataset.name}({self.name}).model")
        else:
            self.model_path = model_path

        if not self.model_path.is_file():
            self._train_model()

        self.model = Doc2Vec.load(str(self.model_path))

        if not self.db.is_file():
            self.update_index()

    def _train_model(self) -> None:
        tagged_docs = []
        for entry in self.dataset:
            tokens = simple_preprocess(entry.main_content)
            lemmatized_tokens = [lemmatize_word(token) for token in tokens]
            tagged_docs.append(TaggedDocument(lemmatized_tokens, [entry.id]))

        self.model = Doc2Vec(vector_size=50, min_count=2, epochs=200)
        self.model.build_vocab(tagged_docs)
        with TimeLogger(f"Training {self.name} with dataset {self.dataset.name}... "):
            self.model.train(
                corpus_iterable=tagged_docs,
                total_examples=self.model.corpus_count,
                epochs=self.model.epochs
            )
        self.model.save(str(self.model_path))

    def _update_index(self) -> None:
        """Builds the index of the engine in a sqlite database."""
        SQLModel.metadata.create_all(self.db_engine)

        with DatabaseBatchCommit(self.db_engine) as batcher:
            with TimeLogger("Adding Documents to db... "):
                for entry in self.dataset:
                    batcher.add(Document(id=entry.id, text=entry.raw_text))
                    for label in entry.labels:
                        batcher.add(LabeledDoc(document_id=entry.id, label=label))

    def _get_docs_by_feedback(self, session: Session, relevance: int, query: str):
        """Filter all docs with Document.feedback.relevance == relevance"""
        query_vector = self.model.infer_vector(simple_preprocess(query))
        feedbacks = session.exec(
            select(Feedback).where(
                Feedback.relevance == relevance).group_by(
                    Feedback.query)
        ).all()

        # Filter feedback instances by it's query cosine similarity
        filtered = []
        for feedback in feedbacks:
            fq_query = self.model.infer_vector(
                simple_preprocess(feedback.query))
            distance = cosine_distances([query_vector, fq_query])[0, 1]
            # print(
            #     f"distance between {query}-{feedback.query} in doc {feedback.document_id}: {distance}")
            if distance <= 0.3:
                filtered.append(feedback.query)

        doc_ids = session.exec(
            select(Feedback.document_id).where(Feedback.query.in_(filtered))
        ).all()
        docs_related = session.exec(
            select(Document).where(Document.id.in_(doc_ids))
        ).all()

        return docs_related

    def apply_feedback(self, query: str):
        """Optimize query string based on known feedback applying Rocchio algorithm
            q' = alpha * q + rel_doc * beta - nonrel_doc * gamma
        """
        query_vector = self.model.infer_vector(
            simple_preprocess(query.lower()))

        with Session(self.db_engine) as session:
            relevant_docs = self._get_docs_by_feedback(session, 1, query)[:4]
            non_relevant_docs = self._get_docs_by_feedback(
                session, -1, query)[:4]

            alpha = 0.97
            beta = 0 if len(relevant_docs) == 0 else 0.4 / len(relevant_docs)
            gamma = 0 if len(non_relevant_docs) == 0 else 0.15 / \
                len(non_relevant_docs)

            query_vector = alpha * query_vector
            for doc in relevant_docs:
                doc_vector = self.model.infer_vector(
                    simple_preprocess(doc.text))
                query_vector += beta * self.model.dv[doc.id]

            for doc in non_relevant_docs:
                doc_vector = self.model.infer_vector(
                    simple_preprocess(doc.text))
                query_vector -= gamma * self.model.dv[doc.id]
        # query_vector = normalize(query_vector[:, np.newaxis], axis=0).ravel()

        # TEST
        # print("NON RELEVANT")
        # for doc in non_relevant_docs:
        #     print(cosine_distances(
        #         [query_vector, self.model.dv[doc.id]])[0][1])

        # print("RELEVANT")
        # for doc in relevant_docs:
        #     print(cosine_distances(
        #         [query_vector, self.model.dv[doc.id]])[0][1])

        return query_vector

    def answer(self, query: str, max_length: int) -> QueryResults:
        inferred_vector = self.apply_feedback(lemmatize_query(query))

        sims = self.model.dv.most_similar([inferred_vector], topn=max_length)

        results = QueryResults(use_rank=False, max_length=max_length)
        for doc_id, sim in sims:
            results.add_result(DocResult(
                id=doc_id,
                sim=sim,
                description=""
            ))

        return results
