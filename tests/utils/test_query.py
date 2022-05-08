from src.utils.query import DocResult, QueryResults


class TestDocResult:

    doc = DocResult(
        id="999",
        sim=0.65,
        description="Lorem ipsum dolor sit amet"
    )

    def test_operators(self):
        d1 = DocResult(
            id="111",
            sim=0.15,
            description="Lorem ipsum dolor sit amet"
        )
        assert self.doc > d1
        assert self.doc >= d1
        assert not self.doc < d1
        assert not self.doc <= d1

        d2 = DocResult(
            id="111",
            sim=0.75,
            description="Lorem ipsum dolor sit amet"
        )
        assert self.doc < d2
        assert self.doc <= d2
        assert not self.doc > d2
        assert not self.doc >= d2

        d3 = DocResult(
            id="111",
            sim=0.65,
            description="Lorem ipsum dolor sit amet"
        )
        assert self.doc >= d3
        assert self.doc <= d3
        assert not self.doc > d3
        assert not self.doc < d3


class TestQueryResults:

    def test_add_result_with_ranking(self):
        results = QueryResults(max_length=3)

        results.add_result(
            d1 := DocResult(
                id="101",
                sim=0.5,
                description="Lorem ipsum dolor sit amet"
            )
        )

        assert len(results) == 1
        assert results.rank[0] == d1

        results.add_result(
            d2 := DocResult(
                id="102",
                sim=0.4,
                description="Lorem ipsum dolor sit amet"
            )
        )

        assert len(results) == 2
        assert results.rank[0] == d1
        assert results.rank[1] == d2

        results.add_result(
            d3 := DocResult(
                id="103",
                sim=0.1,
                description="Lorem ipsum dolor sit amet"
            )
        )

        assert len(results) == 3
        assert results.rank[0] == d1
        assert results.rank[1] == d2
        assert results.rank[2] == d3

        results.add_result(
            d4 := DocResult(
                id="104",
                sim=0.3,
                description="Lorem ipsum dolor sit amet"
            )
        )

        assert len(results) == 3
        assert results.rank[0] == d1
        assert results.rank[1] == d2
        assert results.rank[2] == d4

    def test_add_result_no_ranking(self):
        results = QueryResults(ranking=False)

        for i in range(20):
            results.add_result(
                DocResult(
                    id=str(i),
                    sim=1,
                    description="Lorem ipsum dolor sit amet"
                )
            )

        assert len(results) == 20
        assert results.docs == results.rank
