from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from src.parsers import CranfieldParser, NewsgroupsParser
from src.engines.vectorial import VectorialModel
from src.engines.doc2vec import Doc2VecModel
from src.engines.models import Document
from src.utils.timeit import timed


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
    rank = { doc.id: doc.sim for doc in results.rank }
    ids = [result.id for result in results.docs]

    with Session(engine.db_engine) as session:
        docs = session.query(Document).filter(Document.id.in_(ids)).all()
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
