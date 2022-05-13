from typing import Iterator
import re

from sklearn.feature_extraction.text import CountVectorizer

from .base import DatasetEntry, DatasetParser


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
        return self.count_vzer.fit_transform(
            tuple(entry.raw_text for entry in self)
        )

