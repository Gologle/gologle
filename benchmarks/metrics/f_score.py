from pathlib import Path

from tqdm import tqdm
from sqlmodel import Session, select

import context
from src.engines import Engine
from src.utils import DocResult
from src.parsers import CranfieldParser
from src.engines.doc2vec import Doc2VecModel
from src.main import set_feedback_to_engine, SetFeedbackRequest
from src.engines.models import Feedback


def get_precision_and_recall(
    result: list[DocResult],
    relevant: list[DocResult]
) -> tuple[float, float]:
    # recuperated and relevant
    rr = sum(res in relevant for res in result)
    return rr / len(result), rr / len(relevant)


def f_score(
    result: list[DocResult],
    relevant: list[DocResult],
    beta: float
) -> float:
    try:
        pre, rec = get_precision_and_recall(result, relevant)
        score = (1 + beta ** 2) * pre * rec / (beta ** 2 * pre + rec)
    except ZeroDivisionError:
        pre, rec, score = 0, 0, 0
        score = 0

    # print(">>>", query)
    # print("> result:", sorted(map(lambda x: x.id, result), key=lambda x: int(x)))
    # print("> relevant:", sorted(map(lambda x: x.id, relevant), key=lambda x: int(x)))
    # print("> score:", score, "\t> precision:", pre, "\t> recall:", rec)
    # print("-" * 100)

    return score


def evaluate_f_score(engine: Engine, beta: float = 1.0) -> None:
    with Session(engine.db_engine) as session:
        feedback_to_delete = session.exec(select(Feedback))
        for inst in feedback_to_delete:
            session.delete(inst)
        session.commit()
    queries = engine.dataset.get_test_cases()
    queries = {query: queries[query] for query in list(queries.keys())[:9]}
    final_scores: dict[str, list[float]] = {query: [] for query in queries}
    for i in tqdm(range(50)):
        for query, relevant, in queries.items():
            result = engine.answer(query, max_length=len(relevant) * 10).rank
            score = f_score(result, relevant, 3)
            final_scores[query].append(score)
            for doc in result:
                rating = 1 if doc in relevant else -1
                set_feedback_to_engine(
                    db_engine=engine.db_engine,
                    dataset=engine.dataset,
                    doc_id=doc.id,
                    body=SetFeedbackRequest(query=query, rating=rating)
                )

    # plot the scores
    print(len(final_scores))
    for query, score in final_scores.items():
        print(query[:10] + "...", "==>", *map(lambda x: round(x, 4), score))
        print("-" * 100)


if __name__ == "__main__":
    model_path = Path("benchmarks/metrics/f-score.model")
    cranfield = CranfieldParser()
    engine = Doc2VecModel(
        dataset=cranfield,
        model_path=model_path
    )

    evaluate_f_score(engine)

# if __name__ == '__main__':
#     from src.parsers import CranfieldParser
#     from src.engines.doc2vec import Doc2VecModel
#
#     avg = f_score(
#         engine=Doc2VecModel(CranfieldParser()),
#         beta=1.0
#     )
#     print(avg)
