from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


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
    RESTRICTED_REAL = "restricted_real"
    FIXTURE = "fixture"
    UNAVAILABLE = "unavailable"


class ToolCapabilityManifest(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = "1.0"
    category: str
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    required_state_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    supported_goals: list[RequestedGoal] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    allowlisted_fields: dict[str, list[str]] = Field(default_factory=dict)
    retry_limit: int = Field(default=1, ge=0, le=3)
    timeout_seconds: float = Field(default=30, gt=0, le=600)
    fallback_tool: str | None = None
    implementation_ref: str
    real_or_fake: ImplementationKind


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
    tool_name: str
    status: ToolStatus
    visible_facts: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    provenance_labels: list[str] = Field(default_factory=list)
    implementation_kind: ImplementationKind


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
    input_kind: InputKind
    raw_files: list[str] = Field(default_factory=list)
    feature_table: dict[str, float | int | str | None] = Field(default_factory=dict)
    existing_model_result: dict[str, Any] | None = None
    existing_report: str | None = None
    modalities: list[str] = Field(default_factory=list)
    metadata_summary: dict[str, Any] = Field(default_factory=dict)
    completed_workflows: list[str] = Field(default_factory=list)
    pending_workflows: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    model_readiness: ModelReadiness = ModelReadiness.UNKNOWN
    data_quality: str = "unknown"
    plan: PlanSpec | None = None
    status: CaseStatus = CaseStatus.PENDING
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)


class Finding(StrictModel):
    finding_id: str
    title: str
    statement: str
    support_status: str
    source_workflows: list[str] = Field(default_factory=list)
    source_tools: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ReportSection(StrictModel):
    section_id: str
    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)


class FinalAssessmentReportSpec(StrictModel):
    schema_version: str = "1.0"
    case_id: str
    title: str
    summary: str
    sections: list[ReportSection]
    findings: list[Finding]
    limitations: list[str]
    safety_notice: str
    provenance: list[str]
    evaluation_mode: str
    generated_at: datetime = Field(default_factory=utc_now)

