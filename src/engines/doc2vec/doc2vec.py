from __future__ import annotations
from pathlib import Path

from sklearn.metrics.pairwise import cosine_distances
from sqlmodel import SQLModel, Session, select
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from src.engines import Engine
from src.parsers import DatasetParser
from src.utils import DatabaseBatchCommit, TimeLogger, QueryResults, DocResult
from src.engines.models import Document, Feedback


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
            # TODO: use the raw text may no be the best approach, can be
            #  declared an abstract property for the main content of a document
            tokens = simple_preprocess(entry.raw_text)
            tagged_docs.append(TaggedDocument(tokens, [entry.id]))

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

    def _get_docs_by_feedback(self, session: Session, relevance: int, query: str):
        """Filter all docs with Document.feedback.relevance == relevance"""
        query_vector = self.model.infer_vector(simple_preprocess(query))
        feedbacks = session.exec(
            select(Feedback).where(
                Feedback.relevance == relevance).group_by(
                    Feedback.query)
        ).all()

        # Filter feedback instances by it's query cosine distance
        filtered = []
        for feedback in feedbacks:
            fq_query = self.model.infer_vector(
                simple_preprocess(feedback.query))
            distance = cosine_distances([query_vector, fq_query])[0, 1]
            print(
                f"Distance between {query}-{feedback.query} in doc {feedback.document_id}: {distance}")
            if distance <= .3:
                filtered.append(feedback.query)

        # doc_ids = [f.document_id for f in filtered]
        doc_ids = session.exec(
            select(Feedback).where(Feedback.id.in_(filtered))
        ).all()
        docs_related = session.exec(
            select(Document).where(Document.id.in_(doc_ids))
        ).all()
        return docs_related

    def apply_feedback(self, query: str):
        """Optimize query string based on known feedback applying Rocchio algorithm
            q' = q + rel_doc * beta - nonrel_doc * gamma
        """
        query_vector = self.model.infer_vector(
            simple_preprocess(query.lower()))

        with Session(self.db_engine) as session:
            relevant_docs = self._get_docs_by_feedback(session, 1, query)
            non_relevant_docs = self._get_docs_by_feedback(session, -1, query)

            beta = 0 if len(relevant_docs) == 0 else 0.75 / len(relevant_docs)
            gamma = 0 if len(non_relevant_docs) == 0 else 0.15 / \
                len(non_relevant_docs)

            for doc in relevant_docs:
                # doc_vector = self.model.infer_vector(
                #     simple_preprocess(doc.text))
                query_vector = query_vector + beta * self.model.dv[doc.id]

            for doc in non_relevant_docs:
                # doc_vector = self.model.infer_vector(
                #     simple_preprocess(doc.text))
                query_vector = query_vector - gamma * self.model.dv[doc.id]

        return query_vector

    def answer(self, query: str, max_length: int) -> QueryResults:
        inferred_vector = self.apply_feedback(query)

        sims = self.model.dv.most_similar([inferred_vector], topn=max_length)

        results = QueryResults(use_rank=False, max_length=max_length)
        for doc_id, sim in sims:
            results.add_result(DocResult(
                id=doc_id,
                sim=sim,
                description=""
            ))

        return results
