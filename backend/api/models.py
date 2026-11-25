from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    embedding_model: Literal["loaded", "unavailable"]
    generation_model: Literal["lazy", "unavailable"]
    document_count: int
    error_code: Optional[str] = None


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    uploaded_at: str
    page_count: int
    chunk_count: int
    source_available: bool = False


class CitationResponse(BaseModel):
    citation: int
    document_id: str
    page: int
    chunk_id: str


class RetrievalAssessmentResponse(BaseModel):
    sufficient: bool
    empirical_precision: Optional[float]
    calibration_id: str
    reason: Optional[str]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: List[str] = Field(..., min_items=1, max_items=20)
    top_k: Optional[int] = Field(None, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "abstained", "calibration_required"]
    citations: List[CitationResponse]
    abstention_reason: Optional[str]
    retrieval_assessment: Optional[RetrievalAssessmentResponse]
