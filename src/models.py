from typing import List, Union
from uuid import UUID, uuid4
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class MinimalSource(BaseModel):
    """Minimal source representation with file path and character indices."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question representation without answer or ground truth sources."""

    question_id: UUID = Field(default_factory=uuid4)
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question representation with ground truth answer and sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset containing a list of RAG questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Retrieved search results for a single question."""

    question_id: UUID = Field(default_factory=uuid4)
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results and generated answer for a single question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Container for student search results over a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Container for search results and generated answers over a dataset."""

    model_config = ConfigDict(populate_by_name=True)

    search_results: List[MinimalAnswer] = Field(
        default_factory=list,
        validation_alias=AliasChoices("search_results", "results"),
        serialization_alias="search_results",
    )

    @property
    def results(self) -> List[MinimalAnswer]:
        """Alias for search_results."""
        return self.search_results
