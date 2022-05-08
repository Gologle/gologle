from typing import Iterator
from pathlib import Path

from .base import DatasetEntry, DatasetParser


class NewsgroupsEntry(DatasetEntry):

    def __init__(self, entry_path: Path):
        try:
            text = entry_path.read_text(errors="ignore")
        except UnicodeDecodeError as e:
            print(entry_path)
            raise e
        end_of_line1 = text.find("\n")
        end_of_line2 = text.find("\n", end_of_line1 + 1)
        line1 = text[:end_of_line1]
        line2 = text[end_of_line1 + 1: end_of_line2]

        super(NewsgroupsEntry, self).__init__(entry_path.name)

        self._path = entry_path
        self.group = entry_path.parent.name

        if line1.startswith("From: "):
            self.from_ = line1[6:]
            assert line2.startswith("Subject: ")
            self.subject = line2[9:]
        elif line1.startswith("Subject: "):
            self.subject = line1[9:]
            assert line2.startswith("From: ")
            self.from_ = line2[6:]
        else:
            assert False, f"From/Subject not found in {entry_path}"

        self.text = text[end_of_line2:].strip()

    @property
    def raw_text(self):
        return self._path.read_text(errors="ignore")


class NewsgroupsParser(DatasetParser):
    """Parser for the 20 Newsgroups dataset"""

    def __init__(self):
        super(NewsgroupsParser, self).__init__(
            data=self.root / "20newsgroups-18828",
            total=18828
        )

        self.entries: list[NewsgroupsEntry] = []

        for folder in self.data.iterdir():
            for file in folder.iterdir():
                self.entries.append(NewsgroupsEntry(file))

        assert len(self.entries) == self.total

    def __iter__(self) -> Iterator[NewsgroupsEntry]:
        return iter(self.entries)
