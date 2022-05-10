from enum import Enum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from src.parsers import CranfieldParser
from src.engines.vectorial.vectorial import VectorialModel
from src.engines.models import Document
from src.utils.timeit import timed


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Dataset(Enum):
    cranfield = 'cranfield'


def paginated(limit: int, offset: int, results: list, total: int):
    return {
        "results": results,
        "limit": limit,
        "offset": offset,
        "total": total,
        "hasMore": offset + limit < total,
    }


def fetch_documents(q: str, dataset: Dataset.cranfield):
    if dataset == Dataset.cranfield:
        model = VectorialModel(CranfieldParser())

    results = model.answer(q)
    rank = { doc.id: doc.sim for doc in results.rank }
    ids = [result.id for result in results.docs]

    with Session(model.engine) as session:
        docs = session.query(Document).filter(Document.id.in_(ids)).all()
        docs = sorted(docs, key=lambda doc: rank[doc.id], reverse=True)

    return docs

@app.get("/query")
async def query(q: str, dataset: Dataset = Dataset.cranfield, limit: int = 10, offset: int = 0):
    try:
        (docs, time) = timed(fetch_documents, q, dataset)
        return { "query": q, "time": time } | paginated(limit, offset, docs[offset:offset + limit], len(docs))
            
    except Exception as e: # TODO: Remove this shitty exception handler by fixing empty results bug
        print(e)
        return { "query": q, "time": 0 } | paginated(limit, offset, [], 0)
