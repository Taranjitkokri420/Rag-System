from typing import Literal, TypedDict, Optional, List
from pydantic import BaseModel, Field

class CandidateAssessment(BaseModel):
    """Evaluation schema for an individual applicant."""
    candidate_id: str = Field(description="Exact candidate identifier from context")
    name: str = Field(description="Candidate full name")
    match_score: int = Field(description="0 to 10 relevance fit score, 0=unrelated, 10=exceptional match")
    evidence: str = Field(description="Direct quote or factual excerpt from resume validating score")
    unknowns: str = Field(description="Explicitly state unmentioned attributes or 'none'")

class ScreeningReport(BaseModel):
    """Aggregate report evaluating candidates against recruiter requirements."""
    summary: str = Field(description="Executive summary of candidate assessment")
    candidates: List[CandidateAssessment]

class ChunkScore(BaseModel):
    """Individual chunk relevance score for re-ranking."""
    index: int = Field(description="The numeric index of the chunk being evaluated")
    score: int = Field(description="0-10 relevance score of chunk answering the query directly")

class RerankBatchResult(BaseModel):
    """Batch container for re-ranking scores."""
    scores: List[ChunkScore]

class QueryRoute(BaseModel):
    """Classifier output directing query execution flow."""
    route: Literal["single_candidate_lookup", "candidate_comparison", "pool_screening", "out_of_scope"]
    candidate_names: List[str] = Field(default_factory=list, description="Target candidate names mentioned, if any")
    cleaned_query: str = Field(description="Search-optimized reformulation of the recruiter query")

class RetrievalGrade(BaseModel):
    """Evaluator grade on whether retrieved context satisfies query needs."""
    is_sufficient: bool = Field(description="True if context has sufficient factual detail to answer the query")
    relevance_score: int = Field(description="0-10 confidence score of retrieved evidence relevance")
    explanation: str = Field(description="Brief assessment of missing or present facts")

class AgentState(TypedDict):
    """State machine state tracking query progression through LangGraph nodes."""
    query: str
    current_search_query: str
    route: str
    target_candidates: List[str]
    context: str
    retrieval_grade: Optional[RetrievalGrade]
    rewrite_attempts: int
    report: Optional[ScreeningReport]
    direct_response: Optional[str]
