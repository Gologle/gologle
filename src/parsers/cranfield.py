from typing import Iterator
import re


from sklearn.feature_extraction.text import CountVectorizer

from src.utils import DocResult
from .base import DatasetEntry, DatasetParser
from src.utils.functions import lemmatize_query


class CranfieldEntry(DatasetEntry):

    def __init__(self, raw_text: str):
        # this regex may be insane
        # check it here https://regex101.com/r/giiYJQ/1
        match = re.match(
            pattern=r"\A(?P<id>\d+)\n\.T\n(?P<title>[\w\W]*?).A\n(?P<author>[\w\W]*?)\.B\n(?P<bib>[\w\W]*?)\.W(?P<text>[\w\W]*?)\Z",
            string=raw_text
        )

        super(CranfieldEntry, self).__init__(match["id"])

        self._raw_text = raw_text
        self.title = match["title"].strip()
        self.author = match["author"].strip()
        self.bib = match["bib"].strip()
        self.text = match["text"].strip()

    @property
    def raw_text(self):
        return self._raw_text


class CranfieldParser(DatasetParser):
    """Parser for the Cranfield dataset"""

    def __init__(self):
        super(CranfieldParser, self).__init__(
            data=self.root / "cranfield-1400" / "cran.all.1400",
            count_vzer=CountVectorizer(
                input="content",
                decode_error="ignore",
                stop_words="english",
            ),
            total=1400
        )

        self.entries: list[CranfieldEntry] = []

        raw_entries = self.data.read_text().split(".I ")
        for entry in raw_entries:
            if entry and not entry.isspace():
                self.entries.append(CranfieldEntry(entry))

        assert self.total == len(self.entries)

    def __iter__(self) -> Iterator[CranfieldEntry]:
        return iter(self.entries)

    def fit_transform(self):
        return self.count_vzer.fit_transform([lemmatize_query(entry.raw_text) for entry in self])

    def get_test_cases(self) -> dict[str, list[DocResult]]:
        """Gets the queries from cran.qry file and the relevance from cranqrel.
        There are queries with no relevance related and vice versa. Only are being
        used the results with relevance 1, 2, 3 or 4, in agreement with the description
        in cranqel.readme file.

         Returns:
             A dict that maps the query text with the list of relevant DocResult for it.
         """
        cran_qry_text = (self.root / "cranfield-1400" /
                         "cran.qry").read_text(encoding="utf8")
        cran_qry_regex = r".I (\d+)\n.W\n([\w\W]+?)\n((?=.I)|$)"
        cranqrel_text = (self.root / "cranfield-1400" /
                         "cranqrel").read_text(encoding="utf8")

        queries: dict[str, str] = {}
        for id_, query, _ in re.findall(cran_qry_regex, cran_qry_text):
            # converts '001' to '1'
            queries[str(int(id_))] = query

        test_cases: dict[str, list[DocResult]] = {}
        for line in cranqrel_text.split("\n"):
            query_number, doc_id, rel, *_ = line.split(" ")

            if rel not in ("1", "2", "3", "4"):
                continue

            try:
                query = queries[query_number]
            except KeyError:
                continue

            doc = DocResult(id=doc_id, sim=float(rel), description="")

            if query in test_cases:
                test_cases[query].append(doc)
            else:
                test_cases[query] = [doc]

        return test_cases
