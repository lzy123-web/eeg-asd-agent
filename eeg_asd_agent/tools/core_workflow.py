from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Protocol

from eeg_asd_agent.adapters import (
    CompositeKnowledgeAdapter,
    EEGSignalAdapter,
    SeniorMLPBaselineAdapter,
    SeniorOOFAdapter,
)
from eeg_asd_agent.execution import AllowlistProjector, ContextBuilder
from eeg_asd_agent.providers import OfflineReportGenerator
from eeg_asd_agent.reporting import MarkdownRenderer, StructuredReportGenerator
from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    CaseStatus,
    FinalAssessmentReportSpec,
    ImplementationKind,
    ModelReadiness,
    RequestedGoal,
    StateUpdate,
    StructuredKnowledgeResult,
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
    def retrieve(self, query: str, limit: int = 5) -> StructuredKnowledgeResult: ...


ALL_CORE_GOALS = [
    RequestedGoal.FULL_ASSESSMENT,
    RequestedGoal.QUALITY_CHECK,
    RequestedGoal.READINESS_CHECK,
    RequestedGoal.KNOWLEDGE_SEARCH,
    RequestedGoal.REPORT_ONLY,
    RequestedGoal.UNSUPPORTED,
]


def _full_assessment_coverage_gaps(
    report: FinalAssessmentReportSpec,
    state: AssessmentCaseState,
) -> list[str]:
    """Reject a structurally valid report that drops available core evidence."""
    if state.requested_goal is not RequestedGoal.FULL_ASSESSMENT:
        return []
    raw = report.model_dump(mode="json")
    details = raw.get("data_summary", {}).get("details", {})
    eeg_description = details.get("eeg_description", {}) if isinstance(details, dict) else {}
    profile_observed = any(
        item.tool_name == "build_eeg_asd_profile" and item.status is ToolStatus.SUCCESS
        for item in state.observations
    )
    gaps: list[str] = []
    if profile_observed:
        for section in ("spectral_analysis", "regional_analysis", "laterality", "connectivity"):
            value = eeg_description.get(section, {}) if isinstance(eeg_description, dict) else {}
            if not isinstance(value, dict) or value.get("status") in {None, "unavailable"}:
                gaps.append(section)
    model_assessment = raw.get("model_assessment", {})
    if profile_observed and (
        not isinstance(model_assessment, dict)
        or model_assessment.get("status") in {None, "unavailable"}
    ):
        gaps.append("asd_research_profile")
    knowledge_observed = any(
        item.tool_name == "retrieve_asd_knowledge"
        and item.status is ToolStatus.SUCCESS
        and bool(item.facts.get("items"))
        for item in state.observations
    )
    evidence_ids = raw.get("knowledge_interpretation", {}).get("evidence_ids", [])
    if knowledge_observed and not evidence_ids:
        gaps.append("knowledge_evidence")
    return gaps


def _ground_report_with_observations(
    report: FinalAssessmentReportSpec,
    state: AssessmentCaseState,
) -> FinalAssessmentReportSpec:
    """Deterministically retain allowlisted summaries that the LLM omitted."""
    if state.requested_goal is not RequestedGoal.FULL_ASSESSMENT:
        return report
    profile = next(
        (
            item.facts
            for item in reversed(state.observations)
            if item.tool_name == "build_eeg_asd_profile"
            and item.status is ToolStatus.SUCCESS
        ),
        None,
    )
    data_summary = report.data_summary
    model_assessment = report.model_assessment
    if profile:
        details = deepcopy(data_summary.details)
        eeg_description = details.get("eeg_description", {})
        if not isinstance(eeg_description, dict):
            eeg_description = {}
        spectral = deepcopy(profile.get("spectral_summary", {}))
        eeg_description.update(
            {
                "status": "available",
                "spectral_analysis": {
                    "status": "available",
                    "summary": "已计算全局频带功率、脑区频带功率与窗口稳定性。",
                    "values": spectral,
                },
                "regional_analysis": {
                    "status": "available",
                    "summary": "已按标准头皮脑区汇总描述性相对频带功率。",
                    "values": (
                        spectral.get("regional_relative_band_power", {})
                        if isinstance(spectral, dict)
                        else {}
                    ),
                },
                "laterality": {
                    "status": "available",
                    "summary": "已计算左右同源通道的描述性对数功率差异。",
                    "values": deepcopy(profile.get("asymmetry_summary", {})),
                },
                "connectivity": {
                    "status": "available",
                    "summary": "已计算描述性通道相关性与频带相干性摘要。",
                    "values": deepcopy(profile.get("connectivity_summary", {})),
                },
            }
        )
        details["eeg_description"] = eeg_description
        details.setdefault("technical_description", deepcopy(profile.get("recording_summary", {})))
        details["research_highlights"] = deepcopy(profile.get("research_highlights", []))
        data_summary = data_summary.model_copy(
            update={
                "status": "available",
                "summary": (
                    data_summary.summary
                    + " 已保留本地工具生成的频谱、脑区、左右差异和连接性摘要。"
                ),
                "source_tools": list(
                    dict.fromkeys([*data_summary.source_tools, "build_eeg_asd_profile"])
                ),
                "details": details,
            }
        )
        if model_assessment.status == "unavailable":
            model_assessment = model_assessment.model_copy(
                update={
                    "status": "available",
                    "summary": (
                        "已生成非诊断性的 EEG-ASD 研究性描述档案；该档案不是分类器，"
                        "不能估计诊断、严重程度或能力。"
                    ),
                    "source_tools": list(
                        dict.fromkeys(
                            [*model_assessment.source_tools, "build_eeg_asd_profile"]
                        )
                    ),
                    "details": {
                        **model_assessment.details,
                        "profile_status": profile.get("profile_status"),
                        "interpretation_scope": profile.get("interpretation_scope"),
                        "research_highlights": profile.get("research_highlights", []),
                    },
                }
            )
    knowledge = report.knowledge_interpretation
    retrieved_ids = [
        str(item.get("evidence_id"))
        for observation in state.observations
        if observation.tool_name == "retrieve_asd_knowledge"
        and observation.status is ToolStatus.SUCCESS
        for item in observation.facts.get("items", [])
        if isinstance(item, dict) and item.get("evidence_id")
    ]
    if retrieved_ids and not knowledge.evidence_ids:
        knowledge = knowledge.model_copy(
            update={
                "status": "available",
                "summary": (
                    knowledge.summary
                    + " 已附加检索工具实际返回的 Evidence ID，供人工回溯。"
                ),
                "evidence_ids": retrieved_ids,
            }
        )
    return report.model_copy(
        update={
            "data_summary": data_summary,
            "model_assessment": model_assessment,
            "knowledge_interpretation": knowledge,
        }
    )


def _manifest(
    name: str,
    *,
    version: str = "1.0",
    category: str,
    produced_outputs: list[str],
    allowlisted_fields: list[str],
    implementation: ImplementationKind,
    goals: list[RequestedGoal] | None = None,
    required_inputs: list[str] | None = None,
    optional_inputs: list[str] | None = None,
    required_state_fields: list[str] | None = None,
    preconditions: list[str] | None = None,
    limitations: list[str] | None = None,
    retry_limit: int = 1,
    description: str | None = None,
    use_when: list[str] | None = None,
    avoid_when: list[str] | None = None,
    tags: list[str] | None = None,
    enabled: bool = True,
    repeatable: bool = False,
    idempotent: bool = True,
) -> ToolCapabilityManifest:
    active_goals = goals or ALL_CORE_GOALS
    return ToolCapabilityManifest(
        name=name,
        version=version,
        category=category,
        description=description or name.replace("_", " "),
        use_when=use_when or [],
        avoid_when=avoid_when or [],
        tags=tags or ["eeg-asd", category],
        required_inputs=required_inputs or [],
        optional_inputs=optional_inputs or [],
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
        enabled=enabled,
        repeatable=repeatable,
        idempotent=idempotent,
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
    eeg_adapter: EEGSignalAdapter | None = None,
    baseline_adapter: SeniorMLPBaselineAdapter | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    provider = report_provider or OfflineReportGenerator()
    knowledge = knowledge_adapter or CompositeKnowledgeAdapter()
    oof = oof_adapter or SeniorOOFAdapter()
    eeg = eeg_adapter or EEGSignalAdapter()
    baseline = baseline_adapter or SeniorMLPBaselineAdapter()

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
        description="Inspect which case assets exist without deciding how they must be used.",
        use_when=["The Agent needs an inventory of available EEG-ASD case assets."],
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

    eeg_goals = [
        RequestedGoal.FULL_ASSESSMENT,
        RequestedGoal.QUALITY_CHECK,
        RequestedGoal.FEATURE_EXTRACTION,
    ]
    inspect_eeg_manifest = _manifest(
        "inspect_eeg",
        version=eeg.version,
        category="eeg",
        produced_outputs=[
            "pipeline_version",
            "source_format",
            "synthetic",
            "channel_count",
            "sample_count",
            "sampling_rate_hz",
            "duration_seconds",
            "event_count",
            "event_source",
            "event_labels",
            "unit_assumption",
            "channel_names",
            "auxiliary_channel_names",
            "reference_channel_names",
            "montage_family",
        ],
        allowlisted_fields=[
            "pipeline_version",
            "source_format",
            "synthetic",
            "channel_count",
            "sample_count",
            "sampling_rate_hz",
            "duration_seconds",
            "event_count",
            "event_source",
            "event_labels",
            "unit_assumption",
            "channel_names",
            "auxiliary_channel_names",
            "reference_channel_names",
            "montage_family",
        ],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=eeg_goals,
        preconditions=["raw_eeg_present", "case_inspected"],
        limitations=[
            "Synthetic recordings, when used, are workflow fixtures and not participant evidence.",
            "V2 supports synthetic URI, NPZ, MAT, and equal-rate EEG channels in EDF input.",
        ],
    )

    def inspect_eeg(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        inspected = eeg.inspect(state)
        limitations = (
            ["This recording is synthetic and can validate only software execution."]
            if inspected["synthetic"]
            else []
        )
        return _outcome(
            inspect_eeg_manifest,
            ToolStatus.SUCCESS,
            data=inspected,
            limitations=limitations,
        )

    registry.register(FunctionTool(inspect_eeg_manifest, inspect_eeg))

    eeg_quality_manifest = _manifest(
        "check_eeg_quality",
        version=eeg.version,
        category="quality",
        produced_outputs=[
            "quality_status",
            "finite_sample_ratio",
            "flat_channel_count",
            "flat_channels",
            "excessive_amplitude_channels",
            "max_abs_amplitude_uv",
            "median_p99_abs_amplitude_uv",
            "max_artifact_sample_fraction",
            "median_line_noise_ratio",
            "median_high_frequency_ratio",
            "median_absolute_channel_correlation",
            "usable_channel_count",
            "issue_count",
            "issues",
            "warning_count",
            "warnings",
            "synthetic",
            "pipeline_version",
        ],
        allowlisted_fields=[
            "quality_status",
            "finite_sample_ratio",
            "flat_channel_count",
            "flat_channels",
            "excessive_amplitude_channels",
            "max_abs_amplitude_uv",
            "median_p99_abs_amplitude_uv",
            "max_artifact_sample_fraction",
            "median_line_noise_ratio",
            "median_high_frequency_ratio",
            "median_absolute_channel_correlation",
            "usable_channel_count",
            "issue_count",
            "issues",
            "warning_count",
            "warnings",
            "synthetic",
            "pipeline_version",
        ],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=eeg_goals,
        preconditions=["raw_eeg_present", "case_inspected", "eeg_inspected"],
        limitations=["Signal thresholds are engineering gates, not clinical EEG interpretation."],
    )

    def check_eeg_quality(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        checked = eeg.quality(state)
        return _outcome(
            eeg_quality_manifest,
            ToolStatus.SUCCESS,
            data=checked,
            update=StateUpdate(data_quality=str(checked["quality_status"])),
        )

    registry.register(FunctionTool(eeg_quality_manifest, check_eeg_quality))

    preprocess_manifest = _manifest(
        "preprocess_eeg",
        version=eeg.version,
        category="eeg",
        produced_outputs=[
            "pipeline_version",
            "source_sampling_rate_hz",
            "target_sampling_rate_hz",
            "channel_count",
            "sample_count",
            "highpass_hz",
            "lowpass_hz",
            "notch_hz",
            "normalization",
            "reference",
            "excluded_auxiliary_channels",
            "excluded_reference_channels",
            "synthetic",
        ],
        allowlisted_fields=[
            "pipeline_version",
            "target_sampling_rate_hz",
            "channel_count",
            "highpass_hz",
            "lowpass_hz",
            "notch_hz",
            "normalization",
            "reference",
            "excluded_auxiliary_channels",
            "excluded_reference_channels",
            "synthetic",
        ],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        preconditions=[
            "raw_eeg_present",
            "data_quality_pass",
            "eeg_quality_checked",
        ],
        limitations=["Processing parameters are fixed by the versioned eeg-v2.0 profile."],
    )

    def preprocess_eeg(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        return _outcome(
            preprocess_manifest,
            ToolStatus.SUCCESS,
            data=eeg.preprocess_summary(state),
        )

    registry.register(FunctionTool(preprocess_manifest, preprocess_eeg))

    events_manifest = _manifest(
        "extract_events",
        version=eeg.version,
        category="eeg",
        produced_outputs=[
            "event_count",
            "event_mode",
            "event_codes",
            "event_source",
            "event_labels",
            "pipeline_version",
            "synthetic",
        ],
        allowlisted_fields=[
            "event_count",
            "event_mode",
            "event_codes",
            "event_source",
            "event_labels",
            "pipeline_version",
            "synthetic",
        ],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        preconditions=[
            "raw_eeg_present",
            "data_quality_pass",
            "eeg_preprocessed",
        ],
        limitations=["When annotations are absent, events are inferred as fixed windows."],
    )

    def extract_events(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        return _outcome(events_manifest, ToolStatus.SUCCESS, data=eeg.event_summary(state))

    registry.register(FunctionTool(events_manifest, extract_events))

    epochs_manifest = _manifest(
        "create_epochs",
        version=eeg.version,
        category="eeg",
        produced_outputs=[
            "epoch_count",
            "channel_count",
            "samples_per_epoch",
            "epoch_seconds",
            "event_mode",
            "pipeline_version",
            "synthetic",
        ],
        allowlisted_fields=[
            "epoch_count",
            "channel_count",
            "samples_per_epoch",
            "epoch_seconds",
            "event_mode",
            "pipeline_version",
            "synthetic",
        ],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        preconditions=[
            "raw_eeg_present",
            "data_quality_pass",
            "events_extracted",
        ],
        limitations=["Epochs are deterministic analysis windows, not clinically annotated trials."],
    )

    def create_epochs(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        return _outcome(epochs_manifest, ToolStatus.SUCCESS, data=eeg.epoch_summary(state))

    registry.register(FunctionTool(epochs_manifest, create_epochs))

    feature_manifest = _manifest(
        "extract_eeg_features",
        version=eeg.version,
        category="eeg",
        produced_outputs=[
            "features",
            "feature_count",
            "epoch_count",
            "event_mode",
            "pipeline_version",
            "synthetic",
            "feature_semantics",
            "assessment_summary",
        ],
        allowlisted_fields=[
            "feature_count",
            "epoch_count",
            "event_mode",
            "pipeline_version",
            "synthetic",
            "feature_semantics",
            "assessment_summary.global_relative_band_power",
            "assessment_summary.regional_relative_band_power",
            "assessment_summary.temporal",
            "assessment_summary.asymmetry_log_power_ratio",
            "assessment_summary.connectivity",
            "assessment_summary.window_stability",
            "assessment_summary.channel_count",
            "assessment_summary.channel_names",
            "assessment_summary.duration_seconds",
            "assessment_summary.sampling_rate_hz",
        ],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        preconditions=[
            "raw_eeg_present",
            "data_quality_pass",
            "epochs_created",
        ],
        limitations=[
            "Extracted spectral summaries are engineering features and do not establish ASD status.",
            "Raw signal and full feature values are never exposed to the LLM context.",
        ],
    )

    def extract_eeg_features(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        extracted = eeg.extract_features(state)
        raw_features = extracted["features"]
        assert isinstance(raw_features, dict)
        return _outcome(
            feature_manifest,
            ToolStatus.SUCCESS,
            data=extracted,
            update=StateUpdate(
                feature_updates={key: float(value) for key, value in raw_features.items()},
                limitations_add=(
                    ["Features were extracted from deterministic synthetic EEG."]
                    if extracted["synthetic"]
                    else []
                ),
            ),
        )

    registry.register(FunctionTool(feature_manifest, extract_eeg_features))

    profile_manifest = _manifest(
        "build_eeg_asd_profile",
        version="asd-eeg-profile-v1.0",
        category="assessment",
        produced_outputs=[
            "profile_version",
            "profile_status",
            "quality_status",
            "quality_summary",
            "recording_summary",
            "spectral_summary",
            "temporal_summary",
            "asymmetry_summary",
            "connectivity_summary",
            "research_highlights",
            "subject_context",
            "missing_context",
            "interpretation_scope",
            "synthetic",
        ],
        allowlisted_fields=[
            "profile_version",
            "profile_status",
            "quality_status",
            "quality_summary",
            "recording_summary",
            "spectral_summary",
            "temporal_summary",
            "asymmetry_summary",
            "connectivity_summary",
            "research_highlights",
            "subject_context",
            "missing_context",
            "interpretation_scope",
            "synthetic",
        ],
        implementation=ImplementationKind.REAL,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY],
        preconditions=["raw_eeg_present", "features_extracted"],
        repeatable=True,
        description=(
            "Synthesize deterministic EEG quality, spectral, temporal, asymmetry, and "
            "connectivity summaries into a non-diagnostic ASD research profile."
        ),
        limitations=[
            "This descriptive profile is not an ASD classifier and cannot estimate severity or ability.",
            "Age and recording-state context are reported as unavailable when not supplied.",
        ],
    )

    def build_eeg_asd_profile(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        profile = eeg.build_asd_profile(state)
        return _outcome(
            profile_manifest,
            ToolStatus.SUCCESS,
            data=profile,
            limitations=(
                ["Profile was generated from deterministic synthetic EEG."]
                if profile["synthetic"]
                else []
            ),
        )

    registry.register(FunctionTool(profile_manifest, build_eeg_asd_profile))

    quality_manifest = _manifest(
        "check_data_quality",
        category="quality",
        produced_outputs=["quality_status", "issue_count", "issues", "finite_feature_ratio"],
        allowlisted_fields=["quality_status", "issue_count", "issues", "finite_feature_ratio"],
        implementation=ImplementationKind.REAL,
        repeatable=True,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.QUALITY_CHECK,
            RequestedGoal.READINESS_CHECK,
        ],
        preconditions=[
            "feature_table_present",
            "case_inspected",
            "tabular_quality_required",
        ],
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
        repeatable=True,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.QUALITY_CHECK,
            RequestedGoal.READINESS_CHECK,
        ],
        preconditions=[
            "feature_table_present",
            "case_inspected",
            "feature_pipeline_ready",
        ],
        limitations=[
            "Readiness checks availability and known leakage risks; it does not validate clinical performance."
        ],
    )

    def check_model_readiness(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        if state.data_quality == "blocked":
            readiness = ModelReadiness.BLOCKED
            reason = "data quality is blocked"
            mode = "unavailable"
        else:
            available, reason, missing = baseline.readiness(state.feature_table)
            if available:
                readiness = ModelReadiness.RESTRICTED
                mode = "current_case_pretrained_research_baseline"
                if state.existing_model_result is not None:
                    reason = (
                        f"{reason}; an imported result is also present and the Agent may "
                        "reuse, compare, or recompute according to the user goal"
                    )
                if missing:
                    reason = f"{reason}; {len(missing)} of 7 inputs require reference-mean imputation"
            elif state.existing_model_result is not None:
                readiness = ModelReadiness.RESTRICTED
                reason = "only the imported model result is currently available for interpretation"
                mode = "existing_result"
            elif "senior_row_index" in state.metadata_summary:
                readiness = ModelReadiness.RESTRICTED
                reason = "current-case baseline unavailable; historical OOF replay remains available"
                mode = "legacy_oof_replay"
            else:
                readiness = ModelReadiness.BLOCKED
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

    baseline_manifest = _manifest(
        "run_asd_baseline_model",
        version=baseline.version,
        category="model",
        produced_outputs=[
            "model_name",
            "model_version",
            "readiness_status",
            "predicted_pattern",
            "score",
            "class_scores",
            "score_meaning",
            "output_meaning",
            "observed_feature_count",
            "imputed_features",
            "feature_contract_version",
            "semantic_contract_status",
            "scaler_mode",
            "scaler_version",
            "limitations",
        ],
        allowlisted_fields=[
            "model_name",
            "model_version",
            "readiness_status",
            "predicted_pattern",
            "score",
            "class_scores",
            "score_meaning",
            "output_meaning",
            "observed_feature_count",
            "imputed_features",
            "feature_contract_version",
            "semantic_contract_status",
            "scaler_mode",
            "scaler_version",
            "limitations",
        ],
        implementation=ImplementationKind.RESTRICTED_REAL,
        goals=[RequestedGoal.FULL_ASSESSMENT],
        preconditions=[
            "feature_table_present",
            "model_readiness_not_blocked",
            "model_checked",
            "data_quality_pass",
        ],
        repeatable=True,
        use_when=[
            "The user asks for current-case inference, recomputation, validation, or comparison."
        ],
        avoid_when=["The user explicitly asks to interpret an imported result without recomputation."],
        limitations=baseline.limitations([]),
    )

    def run_asd_baseline_model(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        if state.model_readiness is ModelReadiness.BLOCKED:
            return _outcome(
                baseline_manifest,
                ToolStatus.BLOCKED,
                errors=["model readiness is blocked"],
            )
        prediction = baseline.predict(state.feature_table)
        return _outcome(
            baseline_manifest,
            ToolStatus.SUCCESS,
            data=prediction,
            update=StateUpdate(
                existing_model_result=prediction,
                limitations_add=[str(item) for item in prediction["limitations"]],
            ),
            limitations=[str(item) for item in prediction["limitations"]],
        )

    registry.register(FunctionTool(baseline_manifest, run_asd_baseline_model))

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
        preconditions=[
            "model_result_present",
            "case_inspected",
        ],
        repeatable=True,
        use_when=["A supplied result should be interpreted, audited, reused, or compared."],
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
            "baseline_failed",
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

    query_manifest = _manifest(
        "build_asd_rag_query",
        version="1.1",
        category="knowledge",
        produced_outputs=["query", "query_terms", "grounding_sources"],
        allowlisted_fields=["query", "query_terms", "grounding_sources"],
        implementation=ImplementationKind.REAL,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        preconditions=["case_inspected"],
        repeatable=True,
        use_when=[
            "The Agent needs a retrieval query before, during, or after EEG/model analysis."
        ],
        limitations=[
            "The query contains only high-level case state and never raw signal or complete feature values."
        ],
    )

    def build_asd_rag_query(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        terms = ["ASD", "autism", "EEG", "ability", "social communication"]
        grounding = ["user_request", "input_type"]
        if state.existing_model_result:
            pattern = state.existing_model_result.get("predicted_pattern")
            if pattern:
                terms.append(str(pattern).replace("_", " "))
                grounding.append("model_observation")
        if "theta_coh" in state.feature_table:
            terms.extend(["theta coherence", "connectivity"])
            grounding.append("available_feature_names")
        if "parietal_alpha" in state.feature_table:
            terms.extend(["parietal alpha", "spectral power"])
            grounding.append("available_feature_names")
        if state.existing_report:
            terms.extend(["assessment interpretation", "limitations"])
            grounding.append("existing_report_present")
        terms = list(dict.fromkeys(terms))
        query = " ".join(terms)
        return _outcome(
            query_manifest,
            ToolStatus.SUCCESS,
            data={
                "query": query,
                "query_terms": terms,
                "grounding_sources": list(dict.fromkeys(grounding)),
            },
            update=StateUpdate(rag_query=query),
        )

    registry.register(FunctionTool(query_manifest, build_asd_rag_query))

    knowledge_manifest = _manifest(
        "retrieve_asd_knowledge",
        version="2.0",
        category="knowledge",
        optional_inputs=["query"],
        preconditions=["rag_query_present"],
        produced_outputs=["query", "items", "mode"],
        allowlisted_fields=[
            "query",
            "mode",
            "items[].evidence_id",
            "items[].title",
            "items[].source",
            "items[].retrieved_chunk",
            "items[].summary",
            "items[].relevance",
            "items[].metadata.year",
            "items[].metadata.doi",
            "items[].metadata.mode",
            "items[].metadata.ability_id",
            "items[].metadata.dimension_count",
            "items[].metadata.task_count",
            "items[].metadata.graph_path",
            "items[].limitations",
        ],
        implementation=ImplementationKind.ADAPTER,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        repeatable=True,
        use_when=[
            "The Agent needs external ASD/EEG knowledge for planning, interpretation, or reporting."
        ],
        limitations=[
            "优先复用 ASD 专用知识导出，并以 EEGAgent 的既有知识块补足 EEG 报告背景。",
            "EEGAgent 的 FAISS/BGE 权重不完整时，使用同一 chunks.pkl 语料的 BM25 检索。",
            "Retrieved background knowledge is not participant-level evidence.",
        ],
    )

    def retrieve_asd_knowledge(
        state: AssessmentCaseState, arguments: dict[str, object]
    ) -> ToolExecutionOutcome:
        query = str(arguments.get("query") or state.rag_query or "")
        retrieved = knowledge.retrieve(query, limit=5)
        items = [
            {
                "evidence_id": item.evidence_id,
                "title": item.title,
                "source": item.source,
                "retrieved_chunk": item.retrieved_chunk,
                "summary": item.summary,
                "relevance": item.relevance,
                "metadata": item.metadata,
                "limitations": item.limitations,
            }
            for item in retrieved.items
        ]
        status = ToolStatus.SUCCESS if items else ToolStatus.FAILURE
        return _outcome(
            knowledge_manifest,
            status,
            data={"query": query, "items": items, "mode": retrieved.mode},
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
        produced_outputs=["schema_valid", "report_type", "repaired", "llm_payload"],
        allowlisted_fields=[],
        implementation=report_kind,
        goals=[
            RequestedGoal.FULL_ASSESSMENT,
            RequestedGoal.KNOWLEDGE_SEARCH,
            RequestedGoal.REPORT_ONLY,
        ],
        preconditions=["case_inspected", "observations_present"],
        repeatable=True,
        use_when=["The Agent judges that the user goal requires a structured report now."],
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
        report = _ground_report_with_observations(report, state)
        coverage_gaps = _full_assessment_coverage_gaps(report, state)
        if coverage_gaps:
            return _outcome(
                report_manifest,
                ToolStatus.FAILURE,
                data={"schema_valid": True, "coverage_gaps": coverage_gaps},
                errors=[
                    "report omitted available required evidence: "
                    + ", ".join(coverage_gaps)
                ],
                update=StateUpdate(pending_requirements_add=coverage_gaps),
            )
        return _outcome(
            report_manifest,
            ToolStatus.SUCCESS,
            data={
                "schema_valid": True,
                "report_type": report.report_type,
                "repaired": repaired,
                "llm_payload": context,
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
        repeatable=True,
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
        coverage_gaps = _full_assessment_coverage_gaps(state.report_spec, state)
        if coverage_gaps:
            return _outcome(
                render_manifest,
                ToolStatus.BLOCKED,
                errors=[
                    "full-assessment completion criteria are unmet: "
                    + ", ".join(coverage_gaps)
                ],
                update=StateUpdate(pending_requirements_add=coverage_gaps),
            )
        return _outcome(
            render_manifest,
            ToolStatus.SUCCESS,
            data={"format": "markdown", "content": MarkdownRenderer().render(state.report_spec)},
            update=StateUpdate(execution_status=CaseStatus.COMPLETED),
        )

    registry.register(FunctionTool(render_manifest, render_report))

    review_manifest = _manifest(
        "request_human_review",
        category="control",
        produced_outputs=["review_required", "reason", "case_status"],
        allowlisted_fields=["review_required", "reason", "case_status"],
        implementation=ImplementationKind.REAL,
        preconditions=["case_inspected"],
        retry_limit=0,
        description=(
            "Pause the current case and record that qualified human input or review is required."
        ),
        use_when=[
            "Evidence is restricted, required information is missing, or the user asks for human review."
        ],
        limitations=[
            "This records and pauses for review; it does not contact a person or make a clinical decision."
        ],
    )

    def request_human_review(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolExecutionOutcome:
        reason = next(
            iter([*state.pending_requirements, *state.warnings]),
            "Qualified human review is required before continuing this research assessment.",
        )
        return _outcome(
            review_manifest,
            ToolStatus.SUCCESS,
            data={
                "review_required": True,
                "reason": reason,
                "case_status": CaseStatus.WAITING.value,
            },
            update=StateUpdate(
                execution_status=CaseStatus.WAITING,
                pending_requirements_add=[reason],
            ),
        )

    registry.register(FunctionTool(review_manifest, request_human_review))

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
