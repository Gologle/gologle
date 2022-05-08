from typing import Optional

from sqlmodel import SQLModel, Field


# This is not the best approach but let's keep it this way.
# The use of the term text as id is for simplicity.
# Maybe in future is useful more data in this entity for handling of synonyms.
class Term(SQLModel, table=True):
    id: str = Field(primary_key=True)


class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    text: str
