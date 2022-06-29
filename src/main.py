from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.engine import Engine as SqlEngine
from sqlmodel import Field, Session, select

from src.parsers import CranfieldParser, NewsgroupsParser
from src.engines.vectorial import VectorialModel
from src.engines.doc2vec import Doc2VecModel
from src.engines.models import Document, Feedback, LabeledDoc
from src.utils.timeit import timed
from src.classifiers.predictor import load_predictor


# Caching for sqlachemy, improves performance, check the reference
# https://github.com/tiangolo/sqlmodel/issues/189#issuecomment-1025190094
from sqlmodel.sql.expression import Select, SelectOfScalar
SelectOfScalar.inherit_cache = True
Select.inherit_cache = True


class Dataset(Enum):
    cranfield = "cranfield"
    newsgroups = "newsgroups"


class Model(Enum):
    vectorial = "vectorial"
    doc2vec = "doc2vec"


class SetFeedbackRequest(BaseModel):
    query: str
    rating: int


CRANFIELD = CranfieldParser()
NEWSGROUPS = NewsgroupsParser()

ENGINES = {
    Model.vectorial: {
        Dataset.cranfield: VectorialModel(CRANFIELD),
        Dataset.newsgroups: VectorialModel(NEWSGROUPS),
    },
    Model.doc2vec: {
        Dataset.cranfield: Doc2VecModel(CRANFIELD),
        Dataset.newsgroups: Doc2VecModel(NEWSGROUPS),
    }
}


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def paginated(limit: int, offset: int, results: list, total: int):
    return {
        "results": results,
        "limit": limit,
        "offset": offset,
        "total": total,
        "hasMore": offset + limit < total,
    }


def fetch_documents(q: str, model: Model, dataset: Dataset, limit: int, offset: int):
    engine = ENGINES[model][dataset]

    results = engine.answer(q, max_length=200)
    rank = {doc.id: doc.sim for doc in results.rank}
    ids = [result.id for result in results.docs]

    with Session(engine.db_engine) as session:
        docs = session.query(Document).filter(Document.id.in_(ids)).all()

        for elem in sorted(docs, key=lambda doc: rank[doc.id], reverse=True)[:10]:
            print(elem.id, rank[elem.id])

        docs = sorted(docs, key=lambda doc: rank[doc.id], reverse=True)

        # for elem in sorted(docs, key=lambda doc: rank[doc.id], reverse=True)[:10]:
        #     print(elem.id, rank[elem.id])

        docs = sorted(docs, key=lambda doc: rank[doc.id], reverse=True)

    return docs


@app.get("/query")
async def query(
    q: str,
    model: Model = Model.vectorial,
    dataset: Dataset = Dataset.cranfield,
    limit: int = 10,
    offset: int = 0
):
    (docs, time) = timed(fetch_documents, q, model, dataset, limit, offset)
    return {"query": q, "time": time} | paginated(limit, offset, docs[offset:offset + limit], len(docs))


@app.post("/related")
async def related(q: str, dataset: Dataset, model: Model = Model.doc2vec):
    engine = ENGINES[model][dataset]
    labels = engine.predict_labels(q)

    with Session(engine.db_engine) as session:
        labeled_docs = session.exec(
            select(LabeledDoc).where(LabeledDoc.label.in_(labels))
        )

    docs: dict[str, list[str]] = {}
    for lab_doc in labeled_docs:
        try:
            docs[lab_doc].append(lab_doc.label)
        except KeyError:
            docs[lab_doc] = [lab_doc.label]

    return {
        "predicted_labels": labels,
        "docs": [
            {
                "id": doc,
                "labels": labels
            }
            for doc, labels in docs.items()
        ]
    }

@app.get("/document/{dataset}/{doc_id}")
async def details(dataset: Dataset, doc_id: str):
    engine = ENGINES[Model.vectorial][dataset]

    with Session(engine.db_engine) as session:
        doc = session.exec(
            select(Document).where(Document.id == doc_id)
        ).first()

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return doc


def set_feedback_to_engine(db_engine, dataset: Dataset, doc_id: str, body: SetFeedbackRequest):
    normalized_query = body.query.lower()

    with Session(db_engine) as session:
        doc = session.exec(
            select(Document).where(Document.id == doc_id)
        ).one_or_none()

        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        feedback = session.exec(
            select(Feedback).where(Feedback.query ==
                                   normalized_query, Feedback.document_id == doc_id)
        ).one_or_none()

        if feedback is not None:
            feedback.relevance = body.rating
        else:
            feedback = Feedback(query=normalized_query,
                                document_id=doc_id, relevance=body.rating)

        # print(feedback)
        session.add(feedback)
        session.commit()


@app.post("/document/{dataset}/{doc_id}/feedback")
async def set_feedback(dataset: Dataset, doc_id: str, body: SetFeedbackRequest):
    vectorial_engine = ENGINES[Model.vectorial][dataset]
    doc2vec_engine = ENGINES[Model.doc2vec][dataset]

    # Set feedback to all engines in dataset
    set_feedback_to_engine(vectorial_engine.db_engine, dataset, doc_id, body)
    set_feedback_to_engine(doc2vec_engine.db_engine, dataset, doc_id, body)

    return {"message": "OK"}
