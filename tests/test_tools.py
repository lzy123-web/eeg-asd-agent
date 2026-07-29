from eeg_asd_agent.schemas import AssessmentCaseState, InputKind, ModelReadiness, ToolStatus
from eeg_asd_agent.tools.builtin import build_default_registry


def test_default_registry_has_all_planned_tools() -> None:
    names = build_default_registry().names()
    assert len(names) >= 17
    assert {
        "inspect_case",
        "extract_eeg_features",
        "run_asd_classifier",
        "retrieve_asd_knowledge",
        "validate_report_spec",
        "render_report",
    } <= names


def test_synthetic_feature_extraction_is_deterministic() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="synthetic",
        input_kind=InputKind.RAW_EEG,
        raw_files=["synthetic://same"],
    )
    first = registry.execute("extract_eeg_features", state)
    second = registry.execute("extract_eeg_features", state)
    assert first.data["features"] == second.data["features"]
    assert first.data["synthetic"] is True


def test_model_readiness_blocks_unknown_new_case() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="new",
        input_kind=InputKind.FEATURE_TABLE,
        feature_table={"f1": 0.1},
    )
    result = registry.execute("check_model_readiness", state)
    assert result.data["readiness"] == ModelReadiness.BLOCKED.value


def test_senior_oof_replay_is_restricted() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="legacy",
        input_kind=InputKind.FEATURE_TABLE,
        feature_table={"f1": 0.1},
        metadata_summary={"senior_row_index": 0},
        model_readiness=ModelReadiness.RESTRICTED,
    )
    result = registry.execute("run_asd_classifier", state)
    assert result.status is ToolStatus.SUCCESS
    assert result.data["source_type"] == "legacy_oof_replay"
    assert result.implementation_kind.value == "restricted_real"


def test_offline_kg_returns_provenance_labels() -> None:
    registry = build_default_registry()
    state = AssessmentCaseState(
        case_id="kg",
        input_kind=InputKind.KNOWLEDGE_ONLY,
        user_request="social interaction",
    )
    result = registry.execute("retrieve_asd_knowledge", state)
    assert result.status is ToolStatus.SUCCESS
    assert result.data["item_count"] > 0
    assert all(str(item["source"]).startswith("senior_") for item in result.data["items"])
