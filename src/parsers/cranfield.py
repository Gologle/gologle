from abc import ABC
from typing import Iterator
from pathlib import Path
import re

from . import DatasetEntry, DatasetParser


class CranfieldEntry(DatasetEntry):

    def __init__(self, raw_text: str):
        self._raw_text = raw_text

        # this regex may be insane
        # check it here https://regex101.com/r/giiYJQ/1
        match = re.match(
            pattern=r"\A(?P<id>\d+)\n\.T\n(?P<title>[\w\W]*?).A\n(?P<author>[\w\W]*?)\.B\n(?P<B>[\w\W]*?)\.W(?P<text>[\w\W]*?)\Z",
            string=raw_text
        )

        self.id = int(match["id"])
        self.title = match["title"].strip()
        self.author = match["author"].strip()
        self.B = match["B"].strip()             # TODO: what is the right name for this attribute?
        self.text = match["text"].strip()

    @property
    def raw_text(self):
        return self.raw_text


class CranfieldParser(DatasetParser):
    """Parser for the Cranfield dataset"""

    def __init__(self, root: Path = None):
        if root is not None:
            self.root = root
        self.data = self.root / "cranfield-1400" / "cran.all.1400"
        self.total: int = 1400
        self.entries: list[CranfieldEntry] = []

        raw_entries = self.data.read_text().split(".I ")
        for entry in raw_entries:
            if entry and not entry.isspace():
                self.entries.append(CranfieldEntry(entry))

        assert self.total == len(self.entries)

    def __iter__(self) -> Iterator[CranfieldEntry]:
        return iter(self.entries)
