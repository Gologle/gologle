from pydantic import condecimal
from sqlmodel import SQLModel, Field

from src.engines.models import Term, Document


class NormTermFrequency(SQLModel, table=True):
    term_id: str = Field(foreign_key="term.id", primary_key=True)
    document_id: str = Field(foreign_key="document.id", primary_key=True)
    value: float


class InverseDocumentFrequency(SQLModel, table=True):
    term: str = Field(foreign_key="term.id", primary_key=True)
    value: float
