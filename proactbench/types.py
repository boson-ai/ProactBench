"""Pydantic models for ProactBench.

Runtime types used by the Planner / User-Agent / evaluation loop.
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, model_validator


# ═════════════════════════════════════════════════════════════════════════════
# RUNTIME TYPES (Planner / User Agent / evaluation loop)
# ═════════════════════════════════════════════════════════════════════════════

class StateAudit(BaseModel):
    current_turn: int
    previous_trigger_count: int
    is_max_reached: bool  # True when previous_trigger_count >= 6


class EvaluationRubric(BaseModel):
    type: str              # EMERGENT | CRITICAL | RECOVERY
    pass_criteria: str
    partial_criteria: str
    fail_criteria: str


class TacticalOrder(BaseModel):
    intent: str
    content_payload: list[str]
    behavioral_directive: str


class PlannerOutput(BaseModel):
    state_audit: StateAudit
    # Prospective declaration: is_trigger_point=True means the assistant's
    # response to the user message at THIS turn should be evaluated at the
    # NEXT turn by the User Agent.
    is_trigger_point: bool
    stop_conversation: bool
    evaluation_rubric: Optional[EvaluationRubric] = None
    tactical_order: TacticalOrder

    @model_validator(mode="after")
    def enforce_max_reached_constraint(self) -> "PlannerOutput":
        if self.state_audit.is_max_reached and self.is_trigger_point:
            raise ValueError("is_trigger_point must be False when is_max_reached is True (max 6)")
        return self


class EvaluationResult(BaseModel):
    status: str    # EVALUATED | SKIPPED
    score: str     # PASS | PARTIAL | FAIL (only meaningful when status=EVALUATED)
    rationale: str
    evidence: str


class UserAgentOutput(BaseModel):
    evaluation_result: EvaluationResult
    user_message: str


class TriggerPoint(BaseModel):
    turn: int
    evaluation_rubric: EvaluationRubric
    evaluation_result: Optional[EvaluationResult] = None


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHESIS TYPES (scenario + blueprint generation)
# ═════════════════════════════════════════════════════════════════════════════

class ProactiveSubtask(BaseModel):
    task: str
    logic: str


class TrajectoryStep(BaseModel):
    step: int
    type: str  # "Reactive" | "Inference" | "Synthesis" | "Recovery"
    description: str
    grounding: Optional[str] = None


class ProactiveScenario(BaseModel):
    scenario_id: str
    hidden_main_goal: str
    explicit_trigger: str
    implicit_anchors: list[str]
    proactive_subtasks: list[ProactiveSubtask]
    ideal_assistant_trajectory: list[TrajectoryStep]
    persona_alignment_check: str


class PersonaCategoryScenarios(BaseModel):
    scenarios: list[ProactiveScenario]


class EvaluationCheckpoint(BaseModel):
    is_trigger: bool
    type: str  # EMERGENT | CRITICAL | RECOVERY
    expected_inference: str


class ReactionLogic(BaseModel):
    on_proactivity: str
    on_reactivity: str


class InteractionTurn(BaseModel):
    turn: Union[int, str]  # e.g. 1 or "2-N"
    phase: str
    strategic_objective: str
    anchors_to_reveal: list[str]
    evaluation_checkpoint: Optional[EvaluationCheckpoint] = None
    tactical_instructions: Optional[str] = None
    reaction_logic: Optional[ReactionLogic] = None


class BlueprintOutput(BaseModel):
    persona_uuid: Optional[str] = None
    scenario_id: Optional[str] = None
    style_combination_index: Optional[int] = None
    blueprint_id: str
    strategic_overview: str
    interaction_roadmap: list[InteractionTurn]
    style_guardrails: str


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION TYPES (independent-judge blueprint audit)
# ═════════════════════════════════════════════════════════════════════════════

class BlankSlateVerification(BaseModel):
    status: str  # "VALID" | "INVALID"
    rationale: str


class LogicalNecessityPath(BaseModel):
    ideal_inference_sequence: list[str]
    solvability_confirmation: str


class ConstraintCheck(BaseModel):
    persona_alignment: str
    style_compatibility: str


class InferenceSpecificity(BaseModel):
    clarity_score: str  # "1-10"
    required_refinements: list[str]


class ValidationOutput(BaseModel):
    audit_decision: str  # "PASS" | "FAIL" | "NEEDS_REFINEMENT"
    blank_slate_verification: BlankSlateVerification
    logical_necessity_path: LogicalNecessityPath
    constraint_check: ConstraintCheck
    inference_specificity: InferenceSpecificity
    final_summary: str
