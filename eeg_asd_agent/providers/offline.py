from __future__ import annotations

import json
from typing import Any, cast


class OfflineProvider:
    """Deterministic fixture provider.

    It validates orchestration and structured-output handling. It is not evidence
    about the quality of a real language model.
    """

    name = "offline_fixture"

    def generate_json(self, purpose: str, payload: dict[str, object]) -> str:
        if purpose == "planning":
            case = dict(cast(dict[str, object], payload["case"]))
            allowed = list(cast(list[object], payload.get("allowed_tool_names", [])))
            names = self._offline_plan(str(case["input_kind"]), str(payload["goal"]), allowed)
            steps: list[dict[str, Any]] = []
            for index, name in enumerate(names, start=1):
                steps.append(
                    {
                        "step_id": f"s{index:02d}",
                        "tool_name": name,
                        "arguments": {},
                        "depends_on": [f"s{index - 1:02d}"] if index > 1 else [],
                        "condition": None,
                        "on_success": "continue",
                        "on_failure": "continue_limited",
                        "max_retries": 1,
                    }
                )
            return json.dumps(
                {
                    "schema_version": "1.0",
                    "case_id": case["case_id"],
                    "goal": payload["goal"],
                    "planner_kind": "offline_fixture",
                    "steps": steps,
                    "stop_conditions": ["max_steps", "blocked_critical_tool"],
                    "rationale": ["offline fixture plan; not an LLM efficacy result"],
                },
                ensure_ascii=False,
            )
        if purpose == "report":
            return json.dumps(self._offline_report(payload), ensure_ascii=False)
        if purpose == "repair":
            return str(payload.get("raw_output", "{}"))
        raise ValueError(f"unsupported purpose: {purpose}")

    @staticmethod
    def _offline_plan(input_kind: str, goal: str, allowed: list[object]) -> list[str]:
        allowed_names = {str(item) for item in allowed}
        if goal == "unsupported":
            requested = ["inspect_case", "request_human_review"]
        elif goal == "quality_check":
            requested = ["inspect_case", "inspect_eeg", "check_data_quality"]
        elif goal == "feature_extraction":
            requested = [
                "inspect_case",
                "inspect_eeg",
                "check_data_quality",
                "preprocess_eeg",
                "extract_eeg_features",
            ]
        elif goal == "knowledge_search":
            requested = ["inspect_case", "retrieve_asd_knowledge"]
        elif goal == "intervention":
            requested = [
                "inspect_case",
                "retrieve_intervention_knowledge",
                "generate_intervention_candidates",
            ]
        elif input_kind == "raw_eeg":
            requested = [
                "inspect_case",
                "inspect_eeg",
                "check_data_quality",
                "preprocess_eeg",
                "extract_eeg_features",
                "check_model_readiness",
                "run_asd_classifier",
                "generate_medical_analysis",
                "predict_severity",
                "retrieve_asd_knowledge",
            ]
        elif input_kind == "feature_table":
            requested = [
                "inspect_case",
                "check_data_quality",
                "check_model_readiness",
                "run_asd_classifier",
                "generate_medical_analysis",
                "predict_severity",
                "retrieve_asd_knowledge",
            ]
        elif input_kind == "medical_report":
            requested = ["inspect_case", "retrieve_asd_knowledge"]
        else:
            requested = ["inspect_case", "retrieve_asd_knowledge"]
        return [name for name in requested if name in allowed_names]

    @staticmethod
    def _offline_report(payload: dict[str, object]) -> dict[str, object]:
        case_id = str(payload["case_id"])
        results = list(cast(list[object], payload.get("visible_results", [])))
        provenance: list[dict[str, object]] = []
        limitations = [
            "本报告是科研软件的辅助性输出，不能替代临床评估或诊断。",
            "OfflineReportGenerator 仅验证结构化流程，不代表真实语言模型边界效果。",
        ]
        model_summary = "未执行或没有可白名单化的模型结果。"
        model_status = "unavailable"
        model_tools: list[str] = []
        knowledge_summary = "知识检索未执行或没有返回可白名单化条目。"
        knowledge_status = "unavailable"
        evidence_ids: list[str] = []
        for raw in results:
            result = dict(cast(dict[str, object], raw))
            tool_name = str(result["tool_name"])
            provenance.append(
                {
                    "tool_name": tool_name,
                    "observation_id": str(
                        result.get(
                            "observation_id",
                            cast(list[object], result.get("provenance_labels", ["legacy"]))[0],
                        )
                    ),
                    "implementation_kind": str(result["implementation_kind"]),
                }
            )
            result_limitations = [
                str(item)
                for item in cast(list[object], result.get("limitations", []))
            ]
            limitations.extend(result_limitations)
            status = str(result["status"])
            if status != "success":
                continue
            visible = dict(cast(dict[str, object], result.get("visible_facts", {})))
            if tool_name in {
                "research_replay_classifier",
                "inspect_existing_model_result",
            }:
                model_status = str(visible.get("readiness_status", "restricted"))
                pattern = str(visible.get("predicted_pattern", "未提供"))
                meaning = str(visible.get("output_meaning", "仅限研究解释"))
                model_summary = f"历史回放模式为 {pattern}；{meaning}"
                model_tools.append(tool_name)
            elif tool_name == "check_model_readiness":
                model_status = str(visible.get("readiness_status", model_status))
            elif tool_name == "retrieve_asd_knowledge":
                raw_items = cast(list[object], visible.get("items", []))
                evidence_ids = [
                    str(cast(dict[str, object], item).get("evidence_id", ""))
                    for item in raw_items
                    if isinstance(item, dict)
                ]
                evidence_ids = [item for item in evidence_ids if item]
                knowledge_status = "available" if evidence_ids else "unavailable"
                if evidence_ids:
                    knowledge_summary = "检索到可追踪的离线知识条目；仅用于背景解释。"
        unique_limitations = list(dict.fromkeys(limitations))
        return {
            "schema_version": "1.0",
            "case_id": case_id,
            "report_type": "non_diagnostic_research_assessment",
            "request_summary": str(payload.get("user_request", "EEG-ASD 辅助工作流请求")),
            "executed_workflows": [
                str(item) for item in cast(list[object], payload.get("executed_workflows", []))
            ],
            "data_summary": {
                "status": "available" if results else "insufficient",
                "summary": "仅汇总已执行工具经白名单投影后的结构化结果。",
                "source_tools": [
                    str(cast(dict[str, object], item).get("tool_name", ""))
                    for item in results
                    if isinstance(item, dict)
                ],
            },
            "model_assessment": {
                "status": model_status,
                "summary": model_summary,
                "source_tools": model_tools,
            },
            "knowledge_interpretation": {
                "status": knowledge_status,
                "summary": knowledge_summary,
                "evidence_ids": evidence_ids,
            },
            "unavailable_sections": [
                section
                for section, unavailable in (
                    ("model_assessment", model_status == "unavailable"),
                    ("knowledge_interpretation", knowledge_status == "unavailable"),
                )
                if unavailable
            ],
            "limitations": unique_limitations,
            "conclusion": "当前结果只能用于研究工作流辅助与人工复核，不支持临床确诊。",
            "human_review_required": True,
            "provenance": provenance,
        }

    @staticmethod
    def _fact_statement(tool_name: str, visible: dict[str, object]) -> str:
        if tool_name == "check_data_quality":
            return f"数据质量状态为 {visible.get('quality_status', 'unknown')}。"
        if tool_name == "check_model_readiness":
            return f"模型就绪状态为 {visible.get('readiness', 'unknown')}。"
        if tool_name == "run_asd_classifier":
            label = visible.get("predicted_label", "未提供")
            return f"受限模型输出标签为 {label}；该结果不是临床诊断。"
        if tool_name == "predict_severity":
            return f"演示性严重程度输出为 {visible.get('severity', 'unknown')}。"
        if tool_name.startswith("retrieve_"):
            count = visible.get("item_count", 0)
            return f"离线知识检索返回 {count} 条可追踪条目。"
        if tool_name == "extract_eeg_features":
            return f"确定性特征提取产生 {visible.get('feature_count', 0)} 个汇总特征。"
        if tool_name == "generate_intervention_candidates":
            return f"生成 {visible.get('candidate_count', 0)} 条知识型候选建议。"
        return ""


class OfflineReportGenerator(OfflineProvider):
    """Explicit deterministic provider for offline ReportSpec structure validation."""

    name = "offline_report_generator"
