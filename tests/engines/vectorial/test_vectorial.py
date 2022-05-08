from pathlib import Path

import pytest
from sqlmodel import Session, select

from src.engines.vectorial.vectorial import VectorialModel
from src.parsers import CranfieldParser
from src.engines.models import Term, Document
from src.engines.vectorial.models import InverseDocumentFrequency


class TestVectorialModel:

    def test___init__(self):
        # clean old databases
        for db_file in Path(".").glob("*.db"):
            db_file.unlink(missing_ok=True)

        model = VectorialModel(CranfieldParser())

        with Session(model.engine) as session:
            docs = session.exec(select(Document)).all()
            terms = session.exec(select(Term)).all()
            idfs = session.exec(select(InverseDocumentFrequency)).all()

        assert len(docs) == model.dataset.total
        assert len(terms) == len(idfs)

    def test_answer(self):
        # TODO: Design some test for this method
        # model = VectorialModel(CranfieldParser())
        # answer = model.answer("shock wave").rank
        pass

    @pytest.mark.parametrize(
        "terms, expected", [
            (
                ("Lorem", "ipsum", "dolor", "sit", "amet", ),
                {"Lorem": 1, "ipsum": 1, "dolor": 1, "sit": 1, "amet": 1, }
            ),
            (
                ("Lorem", "ipsum", "dolor", "ipsum", "amet", ),
                {"Lorem": 0.5, "ipsum": 1, "dolor": 0.5, "amet": 0.5, }
            ),
            (
                ("Lorem", "ipsum", "dolor", "ipsum", "amet", "ipsum", "amet", ),
                {"Lorem": 1 / 3, "ipsum": 1, "dolor": 1 / 3, "amet": 2 / 3, }
            ),
            (
                ("Lorem", "Lorem", "Lorem"),
                {"Lorem": 1, }
            ),
            (tuple(), dict())
        ]
    )
    def test__nft(self, terms, expected):
        assert VectorialModel._ntf(terms) == expected

    def test__sim(self):
        # TODO: workaround to db fetching
        pass
