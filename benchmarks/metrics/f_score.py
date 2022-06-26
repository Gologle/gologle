import context
from src.engines import Engine
from src.utils import DocResult


def get_precision_and_recall(
    result: list[DocResult],
    relevant: list[DocResult]
) -> tuple[float, float]:
    # recuperated and relevant
    rr = sum(res in relevant for res in result)
    return rr / len(result), rr / len(relevant)


def f_score(engine: Engine, beta: float = 1.0):
    queries = engine.dataset.get_test_cases()
    scores = []
    for query, relevant in queries.items():
        result = engine.answer(query, max_length=len(relevant)).rank
        try:
            pre, rec = get_precision_and_recall(result, relevant)
            score = (1 + beta**2) * pre * rec / (beta**2 * pre + rec)
        except ZeroDivisionError:
            pre, rec, score = 0, 0, 0
        scores.append(score)
        #
        # print(">>>", query)
        # print("> result:", sorted(map(lambda x: x.id, result), key=lambda x: int(x)))
        # print("> relevant:", sorted(map(lambda x: x.id, relevant), key=lambda x: int(x)))
        # print("> score:", score, "\t> precision:", pre, "\t> recall:", rec)
        # print("-" * 100)

    print(scores)
    return sum(scores) / len(scores)


if __name__ == '__main__':
    from src.parsers import CranfieldParser
    from src.engines.doc2vec import Doc2VecModel

    avg = f_score(
        engine=Doc2VecModel(CranfieldParser()),
        beta=1.0
    )
    print(avg)
