from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from eeg_asd_agent.adapters import LocalLiteratureAdapter, SeniorKGAdapter, SeniorOOFAdapter
from eeg_asd_agent.reporting import MarkdownRenderer
from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    FinalAssessmentReportSpec,
    ImplementationKind,
    ModelReadiness,
    RequestedGoal,
    ToolCapabilityManifest,
    ToolResult,
    ToolStatus,
)
from eeg_asd_agent.tools.base import FunctionTool
from eeg_asd_agent.tools.registry import ToolRegistry

ALL_GOALS = list(RequestedGoal)
REPORT_GOALS = [
    RequestedGoal.FULL_ASSESSMENT,
    RequestedGoal.REPORT_ONLY,
    RequestedGoal.KNOWLEDGE_SEARCH,
    RequestedGoal.INTERVENTION,
]


def _visibility(fields: list[str], goals: list[RequestedGoal] | None = None) -> dict[str, list[str]]:
    return {goal.value: fields for goal in (goals or ALL_GOALS)}


def _manifest(
    name: str,
    *,
    category: str,
    output_fields: list[str],
    allowlisted: list[str],
    implementation: ImplementationKind = ImplementationKind.REAL,
    goals: list[RequestedGoal] | None = None,
    limitations: list[str] | None = None,
    unsupported_claims: list[str] | None = None,
    required_state_fields: list[str] | None = None,
    retry_limit: int = 1,
) -> ToolCapabilityManifest:
    return ToolCapabilityManifest(
        name=name,
        category=category,
        output_fields=output_fields,
        supported_goals=goals or ALL_GOALS,
        unsupported_claims=unsupported_claims
        or ["clinical diagnosis", "treatment prescription", "population-level performance"],
        limitations=limitations or [],
        allowlisted_fields=_visibility(allowlisted, goals),
        retry_limit=retry_limit,
        implementation_ref=f"eeg_asd_agent.tools.builtin:{name}",
        real_or_fake=implementation,
        required_state_fields=required_state_fields or [],
    )


def _result(
    manifest: ToolCapabilityManifest,
    status: ToolStatus,
    data: dict[str, Any] | None = None,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name=manifest.name,
        manifest_version=manifest.version,
        status=status,
        data=data or {},
        errors=errors or [],
        warnings=warnings or [],
        limitations=list(dict.fromkeys(manifest.limitations + (limitations or []))),
        implementation_kind=manifest.real_or_fake,
    )


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    senior_oof = SeniorOOFAdapter()
    senior_kg = SeniorKGAdapter()
    literature = LocalLiteratureAdapter()

    inspect_case_manifest = _manifest(
        "inspect_case",
        category="inspection",
        output_fields=["input_kind", "modality_count", "feature_count", "has_model_result", "has_report"],
        allowlisted=["input_kind", "modality_count", "feature_count", "has_model_result", "has_report"],
    )

    def inspect_case(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        return _result(
            inspect_case_manifest,
            ToolStatus.SUCCESS,
            {
                "input_kind": state.input_kind.value,
                "modality_count": len(state.modalities),
                "feature_count": len(state.feature_table),
                "has_model_result": state.existing_model_result is not None,
                "has_report": state.existing_report is not None,
            },
        )

    registry.register(FunctionTool(inspect_case_manifest, inspect_case))

    inspect_eeg_manifest = _manifest(
        "inspect_eeg",
        category="eeg",
        output_fields=["format", "synthetic", "channel_count", "sampling_rate_hz", "duration_seconds"],
        allowlisted=["format", "synthetic", "channel_count", "sampling_rate_hz", "duration_seconds"],
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.QUALITY_CHECK, RequestedGoal.FEATURE_EXTRACTION],
        limitations=["No participant-level raw EEG was available; synthetic input is demo-only."],
        required_state_fields=["raw_files"],
    )

    def inspect_eeg(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        source = state.raw_files[0]
        synthetic = source.startswith("synthetic://")
        if synthetic:
            data = {
                "format": "synthetic_uri",
                "synthetic": True,
                "channel_count": int(state.metadata_summary.get("channel_count", 4)),
                "sampling_rate_hz": float(state.metadata_summary.get("sampling_rate_hz", 128)),
                "duration_seconds": float(state.metadata_summary.get("duration_seconds", 10)),
            }
            return _result(inspect_eeg_manifest, ToolStatus.SUCCESS, data)
        suffix = source.rsplit(".", 1)[-1].lower() if "." in source else "unknown"
        return _result(
            inspect_eeg_manifest,
            ToolStatus.SUCCESS,
            {
                "format": suffix,
                "synthetic": False,
                "channel_count": state.metadata_summary.get("channel_count"),
                "sampling_rate_hz": state.metadata_summary.get("sampling_rate_hz"),
                "duration_seconds": state.metadata_summary.get("duration_seconds"),
            },
            warnings=["Only metadata inspection is enabled for external EEG files in this profile."],
        )

    registry.register(FunctionTool(inspect_eeg_manifest, inspect_eeg))

    quality_manifest = _manifest(
        "check_data_quality",
        category="quality",
        output_fields=["quality_status", "issue_count", "issues", "finite_feature_ratio"],
        allowlisted=["quality_status", "issue_count", "issues", "finite_feature_ratio"],
        goals=ALL_GOALS,
    )

    def check_data_quality(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        issues: list[str] = []
        values = [value for value in state.feature_table.values() if isinstance(value, (int, float))]
        finite = [value for value in values if math.isfinite(float(value))]
        ratio = len(finite) / len(values) if values else None
        if state.input_kind.value == "raw_eeg" and not state.raw_files:
            issues.append("raw EEG input declared but no source is present")
        if state.input_kind.value == "feature_table" and not state.feature_table:
            issues.append("feature table input declared but no features are present")
        if ratio is not None and ratio < 1:
            issues.append("non-finite feature values detected")
        quality = "pass" if not issues else "blocked"
        return _result(
            quality_manifest,
            ToolStatus.SUCCESS,
            {
                "quality_status": quality,
                "issue_count": len(issues),
                "issues": issues,
                "finite_feature_ratio": ratio,
            },
        )

    registry.register(FunctionTool(quality_manifest, check_data_quality))

    preprocess_manifest = _manifest(
        "preprocess_eeg",
        category="eeg",
        output_fields=["preprocessing_status", "target_sampling_rate_hz", "steps", "synthetic"],
        allowlisted=["preprocessing_status", "target_sampling_rate_hz", "steps", "synthetic"],
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        limitations=["This is a deterministic research preprocessing path, not a clinical pipeline."],
        required_state_fields=["raw_files"],
    )

    def preprocess_eeg(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        synthetic = state.raw_files[0].startswith("synthetic://")
        return _result(
            preprocess_manifest,
            ToolStatus.SUCCESS,
            {
                "preprocessing_status": "completed",
                "target_sampling_rate_hz": 128,
                "steps": ["demean", "finite_check", "welch_ready"],
                "synthetic": synthetic,
            },
        )

    registry.register(FunctionTool(preprocess_manifest, preprocess_eeg))

    events_manifest = _manifest(
        "extract_events",
        category="eeg",
        output_fields=["event_count", "event_source"],
        allowlisted=["event_count", "event_source"],
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        limitations=["Synthetic events are demo fixtures."],
    )

    def extract_events(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        count = int(state.metadata_summary.get("event_count", 0))
        return _result(
            events_manifest,
            ToolStatus.SUCCESS,
            {"event_count": count, "event_source": "synthetic_fixture" if state.raw_files else "none"},
        )

    registry.register(FunctionTool(events_manifest, extract_events))

    epochs_manifest = _manifest(
        "create_epochs",
        category="eeg",
        output_fields=["epoch_count", "epoch_seconds"],
        allowlisted=["epoch_count", "epoch_seconds"],
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        limitations=["Epoch creation is deterministic and demo-oriented."],
    )

    def create_epochs(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        duration = float(state.metadata_summary.get("duration_seconds", 10))
        epoch_seconds = 2.0
        return _result(
            epochs_manifest,
            ToolStatus.SUCCESS,
            {"epoch_count": int(duration // epoch_seconds), "epoch_seconds": epoch_seconds},
        )

    registry.register(FunctionTool(epochs_manifest, create_epochs))

    features_manifest = _manifest(
        "extract_eeg_features",
        category="eeg",
        output_fields=["feature_count", "features", "method", "synthetic"],
        allowlisted=["feature_count", "method", "synthetic"],
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.FEATURE_EXTRACTION],
        limitations=["Features from synthetic EEG are not participant evidence."],
    )

    def extract_eeg_features(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        if state.feature_table:
            return _result(
                features_manifest,
                ToolStatus.SUCCESS,
                {
                    "feature_count": len(state.feature_table),
                    "features": state.feature_table,
                    "method": "existing_feature_table",
                    "synthetic": False,
                },
            )
        sfreq = 128
        duration = 10
        time = np.arange(sfreq * duration) / sfreq
        rng = np.random.default_rng(20260729)
        eeg = (
            np.sin(2 * np.pi * 10 * time)
            + 0.35 * np.sin(2 * np.pi * 6 * time)
            + rng.normal(0, 0.1, len(time))
        )
        frequencies, psd = signal.welch(eeg, sfreq, nperseg=256)

        def band(low: float, high: float) -> float:
            mask = (frequencies >= low) & (frequencies < high)
            return float(np.trapezoid(psd[mask], frequencies[mask]))

        features = {
            "delta_power": round(band(1, 4), 8),
            "theta_power": round(band(4, 8), 8),
            "alpha_power": round(band(8, 13), 8),
            "beta_power": round(band(13, 30), 8),
        }
        return _result(
            features_manifest,
            ToolStatus.SUCCESS,
            {
                "feature_count": len(features),
                "features": features,
                "method": "scipy_welch_bandpower",
                "synthetic": True,
            },
        )

    registry.register(FunctionTool(features_manifest, extract_eeg_features))

    readiness_manifest = _manifest(
        "check_model_readiness",
        category="model",
        output_fields=["readiness", "reason", "model_mode"],
        allowlisted=["readiness", "reason", "model_mode"],
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY],
        limitations=[
            "Senior model validation is compromised by legacy split and overlap concerns.",
        ],
        unsupported_claims=["validated prospective ASD diagnosis", "generalization performance"],
    )

    def check_model_readiness(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        if state.metadata_summary.get("profile") == "offline_fixture_model":
            readiness = ModelReadiness.READY
            reason = "explicit offline fixture model"
            mode = "fixture"
        elif "senior_row_index" in state.metadata_summary or state.existing_model_result:
            readiness = ModelReadiness.RESTRICTED
            reason = "legacy OOF replay only; prospective inference is not validated"
            mode = "legacy_oof_replay"
        else:
            readiness = ModelReadiness.BLOCKED
            reason = "no validated deployable model for this case state"
            mode = "unavailable"
        return _result(
            readiness_manifest,
            ToolStatus.SUCCESS,
            {"readiness": readiness.value, "reason": reason, "model_mode": mode},
        )

    registry.register(FunctionTool(readiness_manifest, check_model_readiness))

    classifier_manifest = _manifest(
        "run_asd_classifier",
        category="model",
        output_fields=["predicted_label", "score", "model_count", "source_type"],
        allowlisted=["predicted_label", "score", "model_count", "source_type"],
        implementation=ImplementationKind.RESTRICTED_REAL,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY],
        limitations=[
            "Legacy OOF replay and fixture outputs are not prospective clinical inference.",
        ],
        unsupported_claims=["ASD diagnosis", "validated probability", "clinical severity"],
    )

    def run_asd_classifier(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        if state.model_readiness is ModelReadiness.BLOCKED:
            return _result(
                classifier_manifest,
                ToolStatus.BLOCKED,
                errors=["model readiness is blocked"],
            )
        if state.metadata_summary.get("profile") == "offline_fixture_model":
            alpha = float(state.feature_table.get("alpha_power", 0.0) or 0.0)
            score = round(min(0.95, max(0.05, 0.5 + alpha)), 6)
            return _result(
                classifier_manifest,
                ToolStatus.SUCCESS,
                {
                    "predicted_label": "demo_review_required",
                    "score": score,
                    "model_count": 1,
                    "source_type": "fixture",
                },
                limitations=["Synthetic fixture output; no participant inference."],
            )
        if "senior_row_index" in state.metadata_summary:
            replay = senior_oof.replay(int(state.metadata_summary["senior_row_index"]))
            return _result(classifier_manifest, ToolStatus.SUCCESS, replay)
        if state.existing_model_result:
            return _result(
                classifier_manifest,
                ToolStatus.SUCCESS,
                dict(state.existing_model_result),
                warnings=["Existing model result was reused; the model was not re-run."],
            )
        return _result(classifier_manifest, ToolStatus.BLOCKED, errors=["no replay source"])

    registry.register(FunctionTool(classifier_manifest, run_asd_classifier))

    medical_manifest = _manifest(
        "generate_medical_analysis",
        category="language",
        output_fields=["analysis_summary", "analysis_mode"],
        allowlisted=["analysis_summary", "analysis_mode"],
        implementation=ImplementationKind.FIXTURE,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY],
        limitations=["Fixture medical wording is not a clinical interpretation."],
        unsupported_claims=["diagnosis", "exclusion of ASD", "treatment decision"],
    )

    def generate_medical_analysis(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        summary = (
            "现有工作流结果可用于科研演示性汇总；由于模型和数据限制，"
            "不能形成个体临床诊断。"
        )
        if state.existing_report:
            summary = "已存在报告被作为受限输入保留；系统未验证其临床正确性。"
        return _result(
            medical_manifest,
            ToolStatus.SUCCESS,
            {"analysis_summary": summary, "analysis_mode": "offline_fixture"},
        )

    registry.register(FunctionTool(medical_manifest, generate_medical_analysis))

    severity_manifest = _manifest(
        "predict_severity",
        category="language",
        output_fields=["severity", "severity_mode"],
        allowlisted=["severity", "severity_mode"],
        implementation=ImplementationKind.FIXTURE,
        goals=[RequestedGoal.FULL_ASSESSMENT, RequestedGoal.REPORT_ONLY],
        limitations=["Severity is a demo label and must not be used clinically."],
        unsupported_claims=["clinical severity", "treatment priority"],
    )

    def predict_severity(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        severity = "insufficient_for_clinical_grading"
        return _result(
            severity_manifest,
            ToolStatus.SUCCESS,
            {"severity": severity, "severity_mode": "offline_fixture"},
        )

    registry.register(FunctionTool(severity_manifest, predict_severity))

    knowledge_manifest = _manifest(
        "retrieve_asd_knowledge",
        category="knowledge",
        output_fields=["item_count", "items", "knowledge_source"],
        allowlisted=["item_count", "items", "knowledge_source"],
        implementation=ImplementationKind.RESTRICTED_REAL,
        goals=REPORT_GOALS,
        limitations=["Offline senior KG is a project artifact, not a curated clinical guideline."],
        unsupported_claims=["clinical guideline recommendation", "causal evidence"],
    )

    def retrieve_asd_knowledge(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        query = state.user_request or "ASD social communication"
        try:
            items = senior_kg.retrieve_asd(query)
            papers = literature.search(query)
        except (FileNotFoundError, OSError) as exc:
            return _result(
                knowledge_manifest,
                ToolStatus.BLOCKED,
                errors=[f"offline KG unavailable: {type(exc).__name__}"],
            )
        return _result(
            knowledge_manifest,
            ToolStatus.SUCCESS,
            {
                "item_count": len(items) + len(papers),
                "items": items + papers,
                "knowledge_source": "senior_offline_kg_and_literature",
            },
        )

    registry.register(FunctionTool(knowledge_manifest, retrieve_asd_knowledge))

    intervention_manifest = _manifest(
        "retrieve_intervention_knowledge",
        category="knowledge",
        output_fields=["item_count", "items", "knowledge_source"],
        allowlisted=["item_count", "items", "knowledge_source"],
        implementation=ImplementationKind.RESTRICTED_REAL,
        goals=[RequestedGoal.INTERVENTION],
        limitations=["Retrieved tasks are knowledge candidates, not treatment prescriptions."],
    )

    def retrieve_intervention_knowledge(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolResult:
        try:
            items = senior_kg.retrieve_intervention(state.user_request)
        except (FileNotFoundError, OSError) as exc:
            return _result(
                intervention_manifest,
                ToolStatus.BLOCKED,
                errors=[f"offline KG unavailable: {type(exc).__name__}"],
            )
        state.metadata_summary["intervention_items"] = items
        return _result(
            intervention_manifest,
            ToolStatus.SUCCESS,
            {"item_count": len(items), "items": items, "knowledge_source": "senior_offline_kg"},
        )

    registry.register(FunctionTool(intervention_manifest, retrieve_intervention_knowledge))

    candidate_manifest = _manifest(
        "generate_intervention_candidates",
        category="knowledge",
        output_fields=["candidate_count", "candidates", "candidate_mode"],
        allowlisted=["candidate_count", "candidates", "candidate_mode"],
        implementation=ImplementationKind.RESTRICTED_REAL,
        goals=[RequestedGoal.INTERVENTION],
        limitations=["Candidates require qualified human review and are not prescriptions."],
    )

    def generate_intervention_candidates(
        state: AssessmentCaseState, _: dict[str, object]
    ) -> ToolResult:
        items = list(state.metadata_summary.get("intervention_items", []))
        candidates = [
            {
                "name": str(item.get("name", "候选任务")),
                "rationale": "来自离线知识图谱的评估任务节点，需由专业人员筛选。",
            }
            for item in items[:5]
            if isinstance(item, dict)
        ]
        return _result(
            candidate_manifest,
            ToolStatus.SUCCESS,
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "candidate_mode": "knowledge_candidate",
            },
        )

    registry.register(FunctionTool(candidate_manifest, generate_intervention_candidates))

    validation_manifest = _manifest(
        "validate_report_spec",
        category="reporting",
        output_fields=["schema_valid", "finding_count", "section_count"],
        allowlisted=["schema_valid", "finding_count", "section_count"],
        goals=REPORT_GOALS,
    )

    def validate_report_spec(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        raw = state.metadata_summary.get("report_spec")
        try:
            report = FinalAssessmentReportSpec.model_validate(raw)
        except Exception as exc:
            return _result(
                validation_manifest,
                ToolStatus.FAILURE,
                {"schema_valid": False},
                errors=[f"report schema validation failed: {type(exc).__name__}"],
            )
        return _result(
            validation_manifest,
            ToolStatus.SUCCESS,
            {
                "schema_valid": True,
                "finding_count": len(report.findings),
                "section_count": len(report.sections),
            },
        )

    registry.register(FunctionTool(validation_manifest, validate_report_spec))

    render_manifest = _manifest(
        "render_report",
        category="reporting",
        output_fields=["rendered", "format", "content"],
        allowlisted=["rendered", "format"],
        goals=REPORT_GOALS,
    )

    def render_report(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        report = FinalAssessmentReportSpec.model_validate(state.metadata_summary["report_spec"])
        content = MarkdownRenderer().render(report)
        return _result(
            render_manifest,
            ToolStatus.SUCCESS,
            {"rendered": True, "format": "markdown", "content": content},
        )

    registry.register(FunctionTool(render_manifest, render_report))

    review_manifest = _manifest(
        "request_human_review",
        category="safety",
        output_fields=["review_required", "reason"],
        allowlisted=["review_required", "reason"],
        goals=ALL_GOALS,
        retry_limit=0,
    )

    def request_human_review(state: AssessmentCaseState, _: dict[str, object]) -> ToolResult:
        return _result(
            review_manifest,
            ToolStatus.SUCCESS,
            {
                "review_required": True,
                "reason": "request unsupported, input incomplete, or model evidence restricted",
            },
        )

    registry.register(FunctionTool(review_manifest, request_human_review))
    return registry
