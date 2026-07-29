from __future__ import annotations

import math
from typing import Any, Protocol

from eeg_asd_agent.adapters import SeniorKGAdapter, SeniorOOFAdapter
from eeg_asd_agent.execution import AllowlistProjector, ContextBuilder
from eeg_asd_agent.providers import OfflineReportGenerator
from eeg_asd_agent.reporting import MarkdownRenderer, StructuredReportGenerator
from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    CaseStatus,
    ImplementationKind,
    ModelReadiness,
    RequestedGoal,
    StateUpdate,
    ToolCapabilityManifest,
    ToolExecutionOutcome,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools.base import FunctionTool
from eeg_asd_agent.tools.registry import ToolRegistry


class ReportProvider(Protocol):
    name: str

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str: ...


class KnowledgeAdapter(Protocol):
    def retrieve_asd(self, query: str, limit: int = 5) -> list[dict[str, str]]: ...


ALL_CORE_GOALS = [
    RequestedGoal.FULL_ASSESSMENT,
    RequestedGoal.QUALITY_CHECK,
    RequestedGoal.READINESS_CHECK,
    RequestedGoal.KNOWLEDGE_SEARCH,
    RequestedGoal.REPORT_ONLY,
    RequestedGoal.UNSUPPORTED,
]


def _manifest(
    name: str,
    *,
    category: str,
    produced_outputs: list[str],
    allowlisted_fields: list[str],
    implementation: ImplementationKind,
    goals: list[RequestedGoal] | None = None,
    required_inputs: list[str] | None = None,
    required_state_fields: list[str] | None = None,
    preconditions: list[str] | None = None,
    limitations: list[str] | None = None,
    retry_limit: int = 1,
) -> ToolCapabilityManifest:
    active_goals = goals or ALL_CORE_GOALS
    return ToolCapabilityManifest(
        name=name,
        category=category,
        required_inputs=required_inputs or [],
        required_state_fields=required_state_fields or [],
        preconditions=preconditions or [],
        produced_outputs=produced_outputs,
        supported_goals=active_goals,
        unsupported_claims=[
            "clinical ASD diagnosis",
            "exclusion of ASD",
            "treatment prescription",
            "prospective model performance",
        ],
        limitations=limitations or [],
        allowlisted_fields={goal.value: allowlisted_fields for goal in active_goals},
        retry_limit=retry_limit,
        implementation_reference=f"eeg_asd_agent.tools.core_workflow:{name}",
        real_or_fake=implementation,
    )


def _outcome(
    manifest: ToolCapabilityManifest,
    status: ToolStatus,
    *,
    data: dict[str, Any] | None = None,
    update: StateUpdate | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> ToolExecutionOutcome:
    combined_limitations = list(
        dict.fromkeys([*manifest.limitations, *(limitations or [])])
    )
    result = ToolResult(
        tool_name=manifest.name,
        manifest_version=manifest.version,
        status=status,
        data=data or {},
        errors=errors or [],
        warnings=warnings or [],
        limitations=combined_limitations,
        implementation_kind=manifest.real_or_fake,
    )
    state_update = update or StateUpdate()
    if status is ToolStatus.SUCCESS and manifest.name not in state_update.completed_workflows_add:
        state_update = state_update.model_copy(
            update={
                "completed_workflows_add": [
                    *state_update.completed_workflows_add,
                    manifest.name,
                ],
                "limitations_add": list(
                    dict.fromkeys(
                        [*state_update.limitations_add, *combined_limitations]
                    )
                ),
            }
        )
    elif status in {ToolStatus.FAILURE, ToolStatus.BLOCKED}:
        state_update = state_update.model_copy(
            update={
                "failed_workflows_add": list(
                    dict.fromkeys(
                        [*state_update.failed_workflows_add, manifest.name]
                    )
                ),
                "limitations_add": list(
                    dict.fromkeys(
                        [*state_update.limitations_add, *combined_limitations]
                    )
                ),
            }
        )
    return ToolExecutionOutcome(result=result, state_update=state_update)


def build_core_registry(
    *,
    report_provider: ReportProvider | None = None,
    knowledge_adapter: KnowledgeAdapter | None = None,
    oof_adapter: SeniorOOFAdapter | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    provider = report_provider or OfflineReportGenerator()
    knowledge = knowledge_adapter or SeniorKGAdapter()
    oof = oof_adapter or SeniorOOFAdapter()

    inspect_manifest = _manifest(
        "inspect_case",
        category="inspection",
        produced_outputs=[
            "input_type",
            "feature_count",
            "has_model_result",
            "has_report",
        ],
        allowlisted_fields=["input_type", "feature_count", "has_model_result", "has_report"],
        implementation=ImplementationKind.REAL,
    )

    def inspect_case(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        return _outcome(
            inspect_manifest,
            ToolStatus.SUCCESS,
            data={
                "input_type": state.input_type.value,
                "feature_count": len(state.feature_table),
                "has_model_result": state.existing_model_result is not None,
                "has_report": state.existing_report is not None,
            },
        )

    registry.register(FunctionTool(inspect_manifest, inspect_case))

    quality_manifest = _manifest(
        "check_data_quality",
        category="quality",
        produced_outputs=["quality_status", "issue_count", "issues", "finite_feature_ratio"],
        allowlisted_fields=["quality_status", "issue_count", "issues", "finite_feature_ratio"],
        implementation=ImplementationKind.REAL,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.QUALITY_CHECK,
            RequestedGoal.READINESS_CHECK,
        ],
        preconditions=["feature_table_present"],
    )

    def check_data_quality(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        values = [
            value
            for value in state.feature_table.values()
            if isinstance(value, (int, float))
        ]
        finite = [value for value in values if math.isfinite(float(value))]
        ratio = len(finite) / len(values) if values else None
        issues = [] if ratio in (None, 1.0) else ["non-finite feature values detected"]
        quality = "pass" if not issues else "blocked"
        return _outcome(
            quality_manifest,
            ToolStatus.SUCCESS,
            data={
                "quality_status": quality,
                "issue_count": len(issues),
                "issues": issues,
                "finite_feature_ratio": ratio,
            },
            update=StateUpdate(data_quality=quality),
        )

    registry.register(FunctionTool(quality_manifest, check_data_quality))

    readiness_manifest = _manifest(
        "check_model_readiness",
        category="model",
        produced_outputs=["readiness_status", "reason", "model_mode"],
        allowlisted_fields=["readiness_status", "reason", "model_mode"],
        implementation=ImplementationKind.REAL,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.QUALITY_CHECK,
            RequestedGoal.READINESS_CHECK,
        ],
        preconditions=["feature_table_present"],
        limitations=[
            "Readiness checks availability and known leakage risks; it does not validate clinical performance."
        ],
    )

    def check_model_readiness(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        if state.existing_model_result is not None:
            readiness = ModelReadiness.RESTRICTED
            reason = "an existing result can be interpreted but must not be re-run"
            mode = "existing_result"
        elif "senior_row_index" in state.metadata_summary:
            readiness = ModelReadiness.RESTRICTED
            reason = "legacy OOF replay only; prospective inference is not validated"
            mode = "legacy_oof_replay"
        else:
            readiness = ModelReadiness.BLOCKED
            reason = "no validated deployable model or replay row is available"
            mode = "unavailable"
        return _outcome(
            readiness_manifest,
            ToolStatus.SUCCESS,
            data={
                "readiness_status": readiness.value,
                "reason": reason,
                "model_mode": mode,
            },
            update=StateUpdate(model_readiness=readiness),
        )

    registry.register(FunctionTool(readiness_manifest, check_model_readiness))

    existing_result_manifest = _manifest(
        "inspect_existing_model_result",
        category="model",
        produced_outputs=[
            "model_name",
            "model_version",
            "readiness_status",
            "predicted_pattern",
            "score",
            "output_meaning",
            "limitations",
        ],
        allowlisted_fields=[
            "model_name",
            "model_version",
            "readiness_status",
            "predicted_pattern",
            "score",
            "output_meaning",
            "limitations",
        ],
        implementation=ImplementationKind.ADAPTER,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        preconditions=["model_result_present"],
        limitations=[
            "The tool validates and projects a supplied result; it does not re-run or validate the originating model."
        ],
    )

    def inspect_existing_model_result(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        assert state.existing_model_result is not None
        allowed = set(existing_result_manifest.produced_outputs)
        projected = {
            key: value
            for key, value in state.existing_model_result.items()
            if key in allowed
        }
        return _outcome(
            existing_result_manifest,
            ToolStatus.SUCCESS,
            data=projected,
        )

    registry.register(FunctionTool(existing_result_manifest, inspect_existing_model_result))

    classifier_manifest = _manifest(
        "research_replay_classifier",
        category="model",
        produced_outputs=[
            "model_name",
            "model_version",
            "readiness_status",
            "predicted_pattern",
            "score",
            "output_meaning",
            "limitations",
        ],
        allowlisted_fields=[
            "model_name",
            "model_version",
            "readiness_status",
            "predicted_pattern",
            "score",
            "output_meaning",
            "limitations",
        ],
        implementation=ImplementationKind.REPLAY,
        goals=[RequestedGoal.FULL_ASSESSMENT],
        preconditions=[
            "feature_table_present",
            "model_result_absent",
            "model_readiness_not_blocked",
            "model_checked",
        ],
        limitations=[
            "This is historical OOF replay, not prospective inference on a new participant.",
            "The replay cannot support a clinical diagnosis or validated probability.",
        ],
    )

    def research_replay_classifier(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        if state.model_readiness is ModelReadiness.BLOCKED:
            return _outcome(
                classifier_manifest,
                ToolStatus.BLOCKED,
                errors=["model readiness is blocked"],
            )
        if "senior_row_index" not in state.metadata_summary:
            return _outcome(
                classifier_manifest,
                ToolStatus.BLOCKED,
                errors=["senior_row_index is required for historical replay"],
            )
        replay = oof.replay(int(state.metadata_summary["senior_row_index"]))
        model_result = {
            "model_name": "senior_legacy_oof_ensemble",
            "model_version": "exported_oof_v1",
            "readiness_status": state.model_readiness.value,
            "predicted_pattern": str(replay["predicted_label"]),
            "output_meaning": "Historical research replay requiring human interpretation.",
            "limitations": classifier_manifest.limitations,
        }
        return _outcome(
            classifier_manifest,
            ToolStatus.SUCCESS,
            data=model_result,
            update=StateUpdate(existing_model_result=model_result),
        )

    registry.register(FunctionTool(classifier_manifest, research_replay_classifier))

    knowledge_manifest = _manifest(
        "retrieve_asd_knowledge",
        category="knowledge",
        required_inputs=["query"],
        produced_outputs=["query", "items", "mode"],
        allowlisted_fields=[
            "query",
            "mode",
            "items[].evidence_id",
            "items[].title",
            "items[].normalized_summary",
            "items[].source",
            "items[].relevance",
            "items[].limitations",
        ],
        implementation=ImplementationKind.OFFLINE,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        limitations=[
            "Phase 2 uses a read-only lexical export; this is not yet the senior live RAG runtime.",
            "Retrieved background knowledge is not participant-level evidence.",
        ],
    )

    def retrieve_asd_knowledge(
        _: AssessmentCaseState, arguments: dict[str, object]
    ) -> ToolExecutionOutcome:
        query = str(arguments["query"])
        rows = knowledge.retrieve_asd(query, limit=5)
        items = [
            {
                "evidence_id": str(row.get("id", f"kg-{index}")),
                "title": str(row.get("name", "ASD knowledge entry")),
                "normalized_summary": str(row.get("description", "")),
                "source": str(row.get("source", "senior_kg_export")),
                "relevance": "lexical_export_match",
                "limitations": ["Background knowledge only; not case-specific evidence."],
            }
            for index, row in enumerate(rows, start=1)
        ]
        status = ToolStatus.SUCCESS if items else ToolStatus.FAILURE
        return _outcome(
            knowledge_manifest,
            status,
            data={"query": query, "items": items, "mode": "offline_legacy_csv"},
            errors=[] if items else ["no knowledge entries returned"],
        )

    registry.register(FunctionTool(knowledge_manifest, retrieve_asd_knowledge))

    report_kind = (
        ImplementationKind.OFFLINE
        if provider.name.startswith("offline")
        else ImplementationKind.ADAPTER
    )
    report_manifest = _manifest(
        "generate_report_spec",
        category="report",
        produced_outputs=["schema_valid", "report_type", "repaired"],
        allowlisted_fields=[],
        implementation=report_kind,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        limitations=[
            "OfflineReportGenerator validates structure and flow only; it is not LLM efficacy evidence."
        ]
        if report_kind is ImplementationKind.OFFLINE
        else [],
    )

    def generate_report_spec(
        state: AssessmentCaseState, _arguments: dict[str, object]
    ) -> ToolExecutionOutcome:
        assert state.requested_goal is not None
        context = ContextBuilder(
            registry, AllowlistProjector(registry)
        ).report_context(state, state.requested_goal)
        report, _raw_output, repaired = StructuredReportGenerator(provider).generate(context)
        return _outcome(
            report_manifest,
            ToolStatus.SUCCESS,
            data={
                "schema_valid": True,
                "report_type": report.report_type,
                "repaired": repaired,
            },
            update=StateUpdate(report_spec=report),
        )

    registry.register(FunctionTool(report_manifest, generate_report_spec))

    render_manifest = _manifest(
        "render_report",
        category="report",
        produced_outputs=["format", "content"],
        allowlisted_fields=[],
        implementation=ImplementationKind.REAL,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        preconditions=["report_spec_present"],
        limitations=["Renderer is deterministic and cannot add missing evidence."],
    )

    def render_report(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        if state.report_spec is None:
            return _outcome(
                render_manifest,
                ToolStatus.BLOCKED,
                errors=["validated report_spec is required"],
            )
        return _outcome(
            render_manifest,
            ToolStatus.SUCCESS,
            data={"format": "markdown", "content": MarkdownRenderer().render(state.report_spec)},
            update=StateUpdate(execution_status=CaseStatus.COMPLETED),
        )

    registry.register(FunctionTool(render_manifest, render_report))

    for stop_name, reason in (
        ("stop_with_insufficient_input", "Required case input is unavailable."),
        ("stop_with_unsupported_request", "The request exceeds the closed workflow."),
    ):
        stop_manifest = _manifest(
            stop_name,
            category="control",
            produced_outputs=["stop_reason", "human_review_required"],
            allowlisted_fields=["stop_reason", "human_review_required"],
            implementation=ImplementationKind.REAL,
            limitations=["No assessment conclusion was generated."],
            retry_limit=0,
        )

        def stop(
            _: AssessmentCaseState,
            __: dict[str, object],
            *,
            manifest: ToolCapabilityManifest = stop_manifest,
            stop_reason: str = reason,
        ) -> ToolExecutionOutcome:
            return _outcome(
                manifest,
                ToolStatus.SUCCESS,
                data={"stop_reason": stop_reason, "human_review_required": True},
                update=StateUpdate(
                    execution_status=CaseStatus.BLOCKED,
                    pending_requirements_add=[stop_reason],
                ),
            )

        registry.register(FunctionTool(stop_manifest, stop))

    return registry
