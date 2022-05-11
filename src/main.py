from enum import Enum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from src.parsers import CranfieldParser, NewsgroupsParser
from src.engines.vectorial.vectorial import VectorialModel
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


ENGINES = {
    Model.vectorial: {
        Dataset.cranfield: VectorialModel(CranfieldParser()),
        # Dataset.newsgroups: VectorialModel(NewsgroupsParser()),
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
    try:
        (docs, time) = timed(fetch_documents, q, model, dataset, limit, offset)
        return {"query": q, "time": time} | paginated(limit, offset, docs[offset:offset + limit], len(docs))
            
    except Exception as e:  # TODO: Remove this shitty exception handler by fixing empty results bug
        print(e)
        return {"query": q, "time": 0} | paginated(limit, offset, [], 0)
