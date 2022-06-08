from typing import Callable

import pytest
from fastapi.testclient import TestClient

from src.main import app, Dataset


client = TestClient(app)


@pytest.mark.parametrize(
    ["dataset", "q", "eval_len"],
    [
        (Dataset.cranfield, "even event", lambda x: len(x) > 0),
        (Dataset.cranfield, "shock wave", lambda x: len(x) > 0),
        (Dataset.cranfield, "Lorem 001 ipsum", lambda x: len(x) > 0),
        (Dataset.cranfield, "strength", lambda x: len(x) > 0),
        (Dataset.cranfield, "the FLDSMDFR", lambda x: len(x) == 0),
        (Dataset.cranfield, "Dumv qu3ry", lambda x: len(x) == 0),
        (Dataset.cranfield, "ThisGetsNoResult", lambda x: len(x) == 0),
        (Dataset.cranfield, "a", lambda x: len(x) == 0),
        (Dataset.newsgroups, "Rochester", lambda x: len(x) > 0),
        (Dataset.newsgroups, "Kekule", lambda x: len(x) > 0),
        (Dataset.newsgroups, "Lorem ipsum Kekule", lambda x: len(x) > 0),
    ]
)
def test_query_with_data(dataset: Dataset, q: str, eval_len: Callable):
    resp = client.get(f"/query?q={q}&model=vectorial&dataset={dataset.value}")
    assert resp.status_code == 200
    data = resp.json()
    assert eval_len(data["results"]), f"Results: {data['results']}"


@pytest.mark.parametrize(
    "doc_id, status", [
        *[(str(i), 200) for i in range(100, 200, 10)],      # 10 cases
        *[(str(i), 404) for i in range(9100, 9200, 10)],    # 10 cases
        ("1", 200), ("1a", 404), ("a1", 404), ("a1a", 404),
        ("abc", 404)
    ]
)
def test_details(doc_id: str, status: int):
    resp = client.get(f"/document/cranfield/{doc_id}")
    assert resp.status_code == status
    if status == 200:
        assert resp.json()["id"] == doc_id
