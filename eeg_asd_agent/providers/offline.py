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
        findings: list[dict[str, object]] = []
        provenance: list[str] = []
        limitations = [
            "本报告是科研软件的辅助性输出，不能替代临床评估或诊断。",
            "OfflineProvider 仅验证结构化工作流，不代表真实语言模型效果。",
        ]
        for index, raw in enumerate(results, start=1):
            result = dict(cast(dict[str, object], raw))
            tool_name = str(result["tool_name"])
            provenance.append(tool_name)
            result_limitations = [
                str(item)
                for item in cast(list[object], result.get("limitations", []))
            ]
            limitations.extend(result_limitations)
            status = str(result["status"])
            if status != "success":
                continue
            visible = dict(cast(dict[str, object], result.get("visible_facts", {})))
            statement = OfflineProvider._fact_statement(tool_name, visible)
            if not statement:
                continue
            findings.append(
                {
                    "finding_id": f"f{index:02d}",
                    "title": tool_name.replace("_", " "),
                    "statement": statement,
                    "support_status": str(result.get("implementation_kind", "fixture")),
                    "source_workflows": [tool_name],
                    "source_tools": [tool_name],
                    "limitations": result_limitations,
                }
            )
        if not findings:
            findings.append(
                {
                    "finding_id": "f01",
                    "title": "信息不足",
                    "statement": "当前执行结果不足以形成个体层面的 ASD 辅助评估结论。",
                    "support_status": "insufficient",
                    "source_workflows": [],
                    "source_tools": [],
                    "limitations": ["需要补充可用输入或人工复核。"],
                }
            )
        unique_limitations = list(dict.fromkeys(limitations))
        return {
            "schema_version": "1.0",
            "case_id": case_id,
            "title": "EEG-ASD 工作流辅助评估报告",
            "summary": "系统按当前 Case State 执行了允许的工具，并仅汇总白名单结果。",
            "sections": [
                {
                    "section_id": "executive_summary",
                    "heading": "执行摘要",
                    "paragraphs": [
                        "报告内容来自已执行工具的投影结果；未执行能力不构成证据。"
                    ],
                    "finding_ids": [str(f["finding_id"]) for f in findings],
                },
                {
                    "section_id": "limitations",
                    "heading": "限制",
                    "paragraphs": unique_limitations,
                    "finding_ids": [],
                },
            ],
            "findings": findings,
            "limitations": unique_limitations,
            "safety_notice": "不得将本输出用于临床确诊、排除诊断或自动治疗决策。",
            "provenance": list(dict.fromkeys(provenance)),
            "evaluation_mode": "fixture",
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
