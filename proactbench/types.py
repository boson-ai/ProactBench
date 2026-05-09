"""Pydantic models for the offline-evaluation pipeline."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EvaluationRubric(BaseModel):
    type: str              # EMERGENT | CRITICAL | RECOVERY
    pass_criteria: str
    partial_criteria: str
    fail_criteria: str


class EvaluationResult(BaseModel):
    status: str    # EVALUATED | SKIPPED
    score: str     # PASS | PARTIAL | FAIL (only meaningful when status=EVALUATED)
    rationale: str
    evidence: str


class TriggerPoint(BaseModel):
    turn: int
    evaluation_rubric: EvaluationRubric
    evaluation_result: Optional[EvaluationResult] = None


class JudgeOutput(BaseModel):
    """Structured response from the offline judge."""
    evaluation_result: EvaluationResult
