from __future__ import annotations
from pathlib import Path

from sqlmodel import SQLModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from src.engines import Engine
from src.parsers import DatasetParser
from src.utils import DatabaseBatchCommit, TimeLogger, QueryResults, DocResult
from src.engines.models import Document


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

    def answer(self, query: str, max_length: int) -> QueryResults:
        query = simple_preprocess(query)
        inferred_vector = self.model.infer_vector(query)
        sims = self.model.dv.most_similar([inferred_vector], topn=max_length)

        results = QueryResults(use_rank=False, max_length=max_length)
        for doc_id, sim in sims:
            results.add_result(DocResult(
                id=doc_id,
                sim=sim,
                description=""
            ))

        return results
