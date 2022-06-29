from sqlmodel import SQLModel, Field

# This import is needed for init the database
# noinspection PyUnresolvedReferences
from src.engines.models import Term, Document, LabeledDoc


class WeightTermDocument(SQLModel, table=True):
    term_id: str = Field(foreign_key="term.id", primary_key=True)
    document_id: str = Field(foreign_key="document.id", primary_key=True)
    value: float


class InverseDocumentFrequency(SQLModel, table=True):
    term_id: str = Field(foreign_key="term.id", primary_key=True)
    value: float
