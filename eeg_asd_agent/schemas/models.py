from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class InputKind(str, Enum):
    RAW_EEG = "raw_eeg"
    FEATURE_TABLE = "feature_table"
    MODEL_RESULT = "model_result"
    MEDICAL_REPORT = "medical_report"
    KNOWLEDGE_ONLY = "knowledge_only"
    INCOMPLETE = "incomplete"


class RequestedGoal(str, Enum):
    FULL_ASSESSMENT = "full_assessment"
    QUALITY_CHECK = "quality_check"
    READINESS_CHECK = "readiness_check"
    FEATURE_EXTRACTION = "feature_extraction"
    KNOWLEDGE_SEARCH = "knowledge_search"
    INTERVENTION = "intervention"
    REPORT_ONLY = "report_only"
    UNSUPPORTED = "unsupported"


class CaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    WAITING = "waiting"
    CANCELLED = "cancelled"


class ModelReadiness(str, Enum):
    UNKNOWN = "unknown"
    READY = "ready"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ImplementationKind(str, Enum):
    REAL = "real"
    ADAPTER = "adapter"
    OFFLINE = "offline"
    FAKE = "fake"
    RESTRICTED_REAL = "restricted_real"
    FIXTURE = "fixture"
    REPLAY = "replay"
    STUB = "stub"
    UNAVAILABLE = "unavailable"


class AgentPlanItemStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class AgentPlanItem(StrictModel):
    step_id: str
    objective: str
    status: AgentPlanItemStatus = AgentPlanItemStatus.PENDING
    candidate_tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class AgentPlan(StrictModel):
    """A mutable high-level plan owned by the Agent, not an executable DAG."""

    revision: int = Field(default=1, ge=1)
    objective: str
    strategy: str
    steps: list[AgentPlanItem] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)


class ToolCapabilityManifest(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "1.0"
    category: str
    description: str = ""
    use_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    required_state_fields: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    produced_outputs: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("produced_outputs", "output_fields"),
    )
    side_effects: list[str] = Field(default_factory=list)
    supported_goals: list[RequestedGoal] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    allowlisted_fields: dict[str, list[str]] = Field(default_factory=dict)
    retry_limit: int = Field(default=1, ge=0, le=3)
    timeout_seconds: float = Field(default=30, gt=0, le=600)
    fallback_tool: str | None = None
    enabled: bool = True
    repeatable: bool = False
    idempotent: bool = True
    implementation_reference: str = Field(
        validation_alias=AliasChoices("implementation_reference", "implementation_ref")
    )
    real_or_fake: ImplementationKind

    @property
    def output_fields(self) -> list[str]:
        """Compatibility view for the pre-audit manifest API."""
        return self.produced_outputs

    @property
    def implementation_ref(self) -> str:
        """Compatibility view for the pre-audit manifest API."""
        return self.implementation_reference


class ToolResult(StrictModel):
    invocation_id: str = Field(default_factory=lambda: uuid4().hex)
    tool_name: str
    manifest_version: str
    status: ToolStatus
    started_at: datetime = Field(default_factory=utc_now)
    duration_ms: float = Field(default=0, ge=0)
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    implementation_kind: ImplementationKind


class Observation(StrictModel):
    observation_id: str = Field(default_factory=lambda: uuid4().hex)
    invocation_id: str
    tool_name: str
    status: ToolStatus
    facts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    implementation_kind: ImplementationKind


class LLMVisibleToolResult(StrictModel):
    observation_id: str
    tool_name: str
    status: ToolStatus
    visible_facts: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    provenance_labels: list[str] = Field(default_factory=list)
    implementation_kind: ImplementationKind


class ActionDecision(StrictModel):
    action_id: str = Field(default_factory=lambda: uuid4().hex)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str
    terminal: bool = False
    based_on_observations: list[str] = Field(default_factory=list)


class AgentTurnDecision(StrictModel):
    """One RAC/ReAct turn: revise the plan, then perform at most one action."""

    plan: AgentPlan
    action: ActionDecision
    goal_satisfied: bool = False
    assistant_message: str | None = None
    replan_reason: str | None = None

    @model_validator(mode="after")
    def validate_turn(self) -> AgentTurnDecision:
        if self.action.tool_name is None:
            if not self.action.terminal:
                raise ValueError("a turn without a tool must be terminal")
            if not self.goal_satisfied:
                raise ValueError("a terminal turn without a tool must satisfy the goal")
        else:
            if self.action.terminal:
                raise ValueError(
                    "a tool action cannot be terminal; observe its result before stopping"
                )
            if self.goal_satisfied:
                raise ValueError("goal_satisfied cannot accompany another tool call")
        return self


class StateUpdate(StrictModel):
    """The only supported channel for a tool to change case state."""

    completed_workflows_add: list[str] = Field(default_factory=list)
    failed_workflows_add: list[str] = Field(default_factory=list)
    pending_requirements_add: list[str] = Field(default_factory=list)
    pending_requirements_remove: list[str] = Field(default_factory=list)
    feature_updates: dict[str, float | int | str | None] = Field(default_factory=dict)
    model_readiness: ModelReadiness | None = None
    data_quality: str | None = None
    existing_model_result: dict[str, Any] | None = None
    model_result_history_add: list[dict[str, Any]] = Field(default_factory=list)
    rag_query: str | None = None
    report_spec: FinalAssessmentReportSpec | None = None
    agent_plan: AgentPlan | None = None
    latest_assistant_message: str | None = None
    warnings_add: list[str] = Field(default_factory=list)
    limitations_add: list[str] = Field(default_factory=list)
    current_action: str | None = None
    execution_status: CaseStatus | None = None
    increment_step_count: int = Field(default=1, ge=0, le=1)


class ToolExecutionOutcome(StrictModel):
    result: ToolResult
    state_update: StateUpdate


class PlanStep(StrictModel):
    step_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None
    on_success: str = "continue"
    on_failure: str = "stop"
    max_retries: int = Field(default=1, ge=0, le=3)


class PlanSpec(StrictModel):
    schema_version: str = "1.0"
    plan_id: str = Field(default_factory=lambda: uuid4().hex)
    case_id: str
    goal: RequestedGoal
    planner_kind: str
    steps: list[PlanStep]
    stop_conditions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dag(self) -> PlanSpec:
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step_id values must be unique")
        known: set[str] = set()
        for step in self.steps:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(
                    f"step {step.step_id} has missing or forward dependencies: {sorted(missing)}"
                )
            known.add(step.step_id)
        return self


class AssessmentCaseState(StrictModel):
    schema_version: str = "1.0"
    case_id: str
    user_request: str = ""
    requested_goal: RequestedGoal | None = None
    input_type: InputKind = Field(validation_alias=AliasChoices("input_type", "input_kind"))
    raw_files: list[str] = Field(default_factory=list)
    feature_table: dict[str, float | int | str | None] = Field(default_factory=dict)
    existing_model_result: dict[str, Any] | None = None
    model_result_history: list[dict[str, Any]] = Field(default_factory=list)
    existing_report: str | None = None
    rag_query: str | None = None
    modalities: list[str] = Field(default_factory=list)
    metadata_summary: dict[str, Any] = Field(default_factory=dict)
    completed_workflows: list[str] = Field(default_factory=list)
    failed_workflows: list[str] = Field(default_factory=list)
    pending_workflows: list[str] = Field(default_factory=list)
    pending_requirements: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    model_readiness: ModelReadiness = ModelReadiness.UNKNOWN
    data_quality: str = "unknown"
    plan: PlanSpec | None = None
    current_action: str | None = None
    step_count: int = Field(default=0, ge=0)
    execution_status: CaseStatus = Field(
        default=CaseStatus.PENDING,
        validation_alias=AliasChoices("execution_status", "status"),
    )
    report_spec: FinalAssessmentReportSpec | None = None
    agent_plan: AgentPlan | None = None
    latest_assistant_message: str | None = None
    action_history: list[ActionDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)

    @property
    def input_kind(self) -> InputKind:
        """Compatibility view for pre-audit callers."""
        return self.input_type

    @property
    def status(self) -> CaseStatus:
        """Compatibility view for pre-audit callers."""
        return self.execution_status

    @status.setter
    def status(self, value: CaseStatus) -> None:
        self.execution_status = value


class ReportAssessment(StrictModel):
    status: str
    summary: str
    source_tools: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class KnowledgeInterpretation(StrictModel):
    status: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class StructuredKnowledgeItem(StrictModel):
    evidence_id: str
    title: str
    source: str
    retrieved_chunk: str
    summary: str
    relevance: float = Field(ge=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @property
    def chunk(self) -> str:
        """Compatibility view for adapters written before the evidence contract."""
        return self.retrieved_chunk

    @property
    def score(self) -> float:
        return self.relevance


class StructuredKnowledgeResult(StrictModel):
    query: str
    items: list[StructuredKnowledgeItem]
    mode: str


class EEGProcessingConfig(StrictModel):
    version: str = "eeg-v2.0"
    target_sfreq_hz: int = Field(default=128, ge=64, le=1024)
    highpass_hz: float = Field(default=1.0, ge=0.1, lt=10)
    lowpass_hz: float = Field(default=40.0, gt=10, le=100)
    notch_hz: float = Field(default=50.0, gt=0, le=60)
    reference: str = "average"
    epoch_seconds: float = Field(default=2.0, ge=0.5, le=30)
    max_abs_amplitude_uv: float = Field(default=250.0, gt=0)
    min_duration_seconds: float = Field(default=4.0, gt=0)
    min_channels: int = Field(default=2, ge=1)


class ProvenanceEntry(StrictModel):
    tool_name: str
    observation_id: str
    implementation_kind: ImplementationKind


class FinalAssessmentReportSpec(StrictModel):
    schema_version: str = "1.0"
    case_id: str
    report_type: str
    request_summary: str
    executed_workflows: list[str]
    data_summary: ReportAssessment
    model_assessment: ReportAssessment
    knowledge_interpretation: KnowledgeInterpretation
    unavailable_sections: list[str]
    limitations: list[str]
    conclusion: str
    human_review_required: bool
    provenance: list[ProvenanceEntry]
    generated_at: datetime = Field(default_factory=utc_now)


AssessmentCaseState.model_rebuild()
StateUpdate.model_rebuild()
