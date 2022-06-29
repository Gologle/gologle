from typing import List, Optional
from sqlmodel import Relationship, SQLModel, Field


# The use of the term text as id is for simplicity.
# Maybe in future is useful more data in this entity for handling of synonyms.
class Term(SQLModel, table=True):
    id: str = Field(primary_key=True)


class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    text: str

    feedbacks: List["Feedback"] = Relationship(back_populates="document")


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str
    document_id: str = Field(default=None, foreign_key="document.id")
    relevance: float

    document: Optional[Document] = Relationship(back_populates="feedbacks")


class LabeledDoc(SQLModel, table=True):
    document_id: str = Field(foreign_key="document.id", primary_key=True)
    label: Optional[str] = Field(default=None)

    document: Document = Relationship(back_populates="labeled_docs")
