from ast import In
from enum import Enum
from xml.etree.ElementInclude import include

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from src.parsers import CranfieldParser
from src.engines.vectorial.vectorial import VectorialModel
from src.engines.models import Document


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
        "hasMore": offset + limit < total,
    }


@app.get("/query")
async def query(q: str, dataset: Dataset = Dataset.cranfield, limit: int = 10, offset: int = 0):
    try:
        if dataset == Dataset.cranfield:
            model = VectorialModel(CranfieldParser())

        results = model.answer(q)
        sim = { doc.id: doc.sim for doc in results.docs }
        ids = [result.id for result in results.docs]
        print(sim)

        with Session(model.engine) as session:
            docs = session.query(Document).filter(Document.id.in_(ids)).all()
            docs = sorted(docs, key=lambda doc: sim[doc.id], reverse=True)

            return { "query": q } | paginated(limit, offset, docs[offset:offset + limit], len(docs))
            
    except Exception as e: # TODO: Remove this shitty exception handler by fixing empty results bug
        print(e)
        return { "query": q } | paginated(limit, offset, [], 0)
