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
        if purpose == "agent_turn":
            return json.dumps(self._offline_agent_turn(payload), ensure_ascii=False)
        if purpose == "next_action":
            return json.dumps(payload["fixture_fallback_action"], ensure_ascii=False)
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
    def _offline_agent_turn(payload: dict[str, object]) -> dict[str, object]:
        """Manifest-driven Agent fixture used when no real provider is configured.

        This exercises the same plan/act/observe/replan protocol but is explicitly
        not evidence about LLM planning quality.
        """
        request = str(payload.get("user_request", "")).lower()
        state = dict(cast(dict[str, object], payload["case_state"]))
        inventory = dict(cast(dict[str, object], state["asset_inventory"]))
        raw_eeg_count = inventory.get("raw_eeg_count", 0)
        has_raw_eeg = isinstance(raw_eeg_count, int) and raw_eeg_count > 0
        allowed = [str(item) for item in cast(list[object], payload["allowed_next_actions"])]
        completed = {
            str(item) for item in cast(list[object], state.get("completed_actions", []))
        }
        observations = list(cast(list[object], payload.get("previous_observations", [])))
        observation_ids = [
            str(cast(dict[str, object], item).get("observation_id", ""))
            for item in observations
            if isinstance(item, dict)
        ]
        observation_ids = [item for item in observation_ids if item]

        readiness_only = any(
            word in request
            for word in (
                "模型是否具备",
                "模型就绪",
                "执行条件",
                "readiness only",
            )
        )
        quality_only = not readiness_only and any(
            word in request for word in ("只检查", "只看质量", "quality only")
        )
        feature_only = any(word in request for word in ("只提取", "仅提取", "feature only"))
        knowledge_only = any(
            word in request
            for word in ("只检索", "仅检索", "重新检索", "knowledge only")
        )
        explain_existing = (
            any(word in request for word in ("解释已有", "解释当前", "interpret existing"))
            and not any(word in request for word in ("验证", "重跑", "重新", "verify", "rerun"))
        )
        verify_existing = any(
            word in request
            for word in (
                "验证",
                "重跑",
                "重新分析",
                "重新运行",
                "复现",
                "verify",
                "rerun",
            )
        )
        rag_first = any(word in request for word in ("先检索", "先查", "rag first"))
        wants_report = not any(
            word in request
            for word in ("不生成报告", "不要报告", "无需报告", "no report")
        ) and not quality_only and not feature_only
        current_run_actions = {
            str(cast(dict[str, object], item).get("tool_name"))
            for item in cast(list[object], payload.get("action_history", []))
            if isinstance(item, dict)
        }
        force_repeat: set[str] = set()
        if knowledge_only or any(
            word in request
            for word in ("重新检索", "更新知识", "刷新知识", "refresh knowledge")
        ):
            force_repeat.update({"build_asd_rag_query", "retrieve_asd_knowledge"})
        if explain_existing:
            force_repeat.add("inspect_existing_model_result")
        if verify_existing:
            force_repeat.update({"check_model_readiness", "run_asd_baseline_model"})
        if wants_report and (
            force_repeat
            or any(
                word in request
                for word in ("重新生成报告", "更新报告", "refresh report")
            )
        ):
            force_repeat.update({"generate_report_spec", "render_report"})

        if readiness_only:
            preferred = [
                "inspect_case",
                "inspect_eeg" if has_raw_eeg else "check_data_quality",
                "check_eeg_quality" if has_raw_eeg else "check_data_quality",
                "preprocess_eeg",
                "extract_events",
                "create_epochs",
                "extract_eeg_features",
                "check_model_readiness",
            ]
            finish_after = {"check_model_readiness"}
        elif quality_only:
            preferred = [
                "inspect_case",
                "inspect_eeg" if has_raw_eeg else "check_data_quality",
                "check_eeg_quality" if has_raw_eeg else "check_data_quality",
            ]
            finish_after = {"check_eeg_quality", "check_data_quality"}
        elif feature_only:
            preferred = [
                "inspect_case",
                "inspect_eeg",
                "check_eeg_quality",
                "preprocess_eeg",
                "extract_events",
                "create_epochs",
                "extract_eeg_features",
                "check_data_quality",
            ]
            finish_after = {"extract_eeg_features", "check_data_quality"}
        elif knowledge_only:
            preferred = [
                "inspect_case",
                "build_asd_rag_query",
                "retrieve_asd_knowledge",
                "generate_report_spec",
                "render_report",
            ]
            finish_after = {"render_report"}
        elif explain_existing:
            preferred = [
                "inspect_case",
                "inspect_existing_model_result",
                "build_asd_rag_query",
                "retrieve_asd_knowledge",
                "generate_report_spec",
                "render_report",
            ]
            finish_after = {"render_report"}
        else:
            signal_chain = [
                "inspect_eeg",
                "check_eeg_quality",
                "preprocess_eeg",
                "extract_events",
                "create_epochs",
                "extract_eeg_features",
                "build_eeg_asd_profile",
            ]
            if not has_raw_eeg:
                signal_chain = ["check_data_quality"]
            # A descriptive EEG assessment report must not depend on an ASD classifier.
            # Model tools remain available when a supplied result is being inspected or
            # the request explicitly asks for model verification.
            model_chain: list[str] = []
            if verify_existing:
                model_chain = ["check_model_readiness", "run_asd_baseline_model"]
            if explain_existing and not verify_existing:
                model_chain = ["inspect_existing_model_result"]
            knowledge_chain = ["build_asd_rag_query", "retrieve_asd_knowledge"]
            if rag_first:
                preferred = ["inspect_case", *knowledge_chain, *signal_chain, *model_chain]
            else:
                preferred = ["inspect_case", *signal_chain, *model_chain, *knowledge_chain]
            if wants_report:
                preferred.extend(["generate_report_spec", "render_report"])
            finish_after = {"render_report"} if wants_report else {
                "run_asd_baseline_model",
                "inspect_existing_model_result",
            }

        def is_pending(name: str) -> bool:
            if name in force_repeat:
                return name not in current_run_actions
            return name not in completed and name not in current_run_actions

        pending_preferred = [name for name in preferred if is_pending(name)]
        selected = next((name for name in pending_preferred if name in allowed), None)
        goal_satisfied = bool(finish_after & completed) and selected is None

        if selected is None and not goal_satisfied:
            non_control = [
                name
                for name in allowed
                if not name.startswith("stop_with_") and name not in completed
            ]
            selected = non_control[0] if non_control else None
            goal_satisfied = selected is None

        current_plan = payload.get("current_plan")
        revision = 1
        if isinstance(current_plan, dict):
            revision = int(current_plan.get("revision", 0)) + 1
        remaining = [name for name in preferred if is_pending(name)]
        plan = {
            "revision": revision,
            "objective": str(payload.get("user_request", "Complete the EEG-ASD task")),
            "strategy": "offline manifest-driven Agent simulation; real LLM required for semantic planning claims",
            "steps": [
                {
                    "step_id": f"p{index:02d}",
                    "objective": name.replace("_", " "),
                    "status": "in_progress" if name == selected else "pending",
                    "candidate_tools": [name],
                    "depends_on": [f"p{index - 1:02d}"] if index > 1 else [],
                }
                for index, name in enumerate(remaining, start=1)
            ],
            "completion_criteria": [
                "The explicit user goal is satisfied by executed observations.",
                "No unexecuted capability is represented as successful.",
            ],
        }
        return {
            "plan": plan,
            "action": {
                "tool_name": selected,
                "arguments": {},
                "reason": (
                    f"Offline Agent fixture selected {selected} from the current legal manifests."
                    if selected is not None
                    else "Executed observations satisfy the offline task contract."
                ),
                "terminal": selected is None,
                "based_on_observations": observation_ids,
            },
            "goal_satisfied": selected is None and goal_satisfied,
            "assistant_message": (
                "Offline Agent simulation completed the requested EEG-ASD task."
                if selected is None
                else None
            ),
            "replan_reason": "Updated after the latest case state and observation.",
        }

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
        data_status = "insufficient"
        data_summary = "尚未形成可用于报告的 EEG 结构化评估摘要。"
        data_details: dict[str, object] = {}
        data_tools: list[str] = []
        knowledge_summary = "知识检索未执行或没有返回可白名单化条目。"
        knowledge_status = "unavailable"
        evidence_ids: list[str] = []
        evidence_titles: list[str] = []
        retrieval_mode = "unavailable"
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
            data_tools.append(tool_name)
            visible = dict(cast(dict[str, object], result.get("visible_facts", {})))
            if tool_name in {
                "run_asd_baseline_model",
                "research_replay_classifier",
                "inspect_existing_model_result",
            }:
                model_status = str(visible.get("readiness_status", "restricted"))
                pattern = str(visible.get("predicted_pattern", "未提供"))
                meaning = str(visible.get("output_meaning", "仅限研究解释"))
                model_summary = f"受限研究模型输出模式为 {pattern}；{meaning}"
                model_tools.append(tool_name)
            elif tool_name == "build_eeg_asd_profile":
                data_status = str(visible.get("profile_status", "available"))
                model_status = "available"
                model_summary = (
                    "已生成非诊断性的 EEG-ASD 研究性描述档案；该档案不是分类器，"
                    "不能用于诊断、排除诊断、严重程度或能力估计。"
                )
                model_tools.append(tool_name)
                quality = str(visible.get("quality_status", "unknown"))
                recording = visible.get("recording_summary", {})
                spectral = visible.get("spectral_summary", {})
                temporal = visible.get("temporal_summary", {})
                asymmetry = visible.get("asymmetry_summary", {})
                connectivity = visible.get("connectivity_summary", {})
                subject_context = visible.get("subject_context", {})
                quality_summary = visible.get("quality_summary", {})
                highlights = visible.get("research_highlights", [])
                data_details = {
                    "patient_information": {
                        "case_id": case_id,
                        **(
                            dict(subject_context)
                            if isinstance(subject_context, dict)
                            else {}
                        ),
                    },
                    "history": {
                        "reason_for_eeg": str(payload.get("user_request", "")),
                        "relevant_clinical_information": "未提供",
                    },
                    "technical_description": recording,
                    "eeg_description": {
                        "status": data_status,
                        "background_activity": highlights,
                        "artifacts_and_quality": quality_summary,
                        "spectral_analysis": {
                            "status": "available",
                            "summary": "已计算全局与脑区相对频带功率及窗口稳定性。",
                            "values": spectral,
                        },
                        "regional_analysis": {
                            "status": "available",
                            "summary": "已按前额、额、中央、顶、颞和枕区汇总相对频带功率。",
                            "values": (
                                dict(spectral).get("regional_relative_band_power", {})
                                if isinstance(spectral, dict)
                                else {}
                            ),
                        },
                        "laterality": {
                            "status": "available",
                            "summary": "已计算同源左右通道的描述性对数功率差异。",
                            "values": asymmetry,
                        },
                        "connectivity": {
                            "status": "available",
                            "summary": "已计算描述性通道相关与指定频带相干性摘要。",
                            "values": connectivity,
                        },
                        "abnormal_findings": (
                            "未执行临床异常判读；当前工具仅输出描述性定量特征。"
                        ),
                    },
                    "impression": {
                        "summary": "研究性 EEG 特征档案已生成。",
                        "diagnostic_status": "不提供 ASD 诊断、排除诊断或严重程度判断。",
                    },
                    "clinical_correlation": {
                        "interpretation": visible.get("interpretation_scope", ""),
                        "recommendation": "结合临床量表、发育史及专业人员复核。",
                    },
                    "quality": quality_summary,
                    "recording": recording,
                    "spectral": spectral,
                    "temporal": temporal,
                    "asymmetry": asymmetry,
                    "connectivity": connectivity,
                    "research_highlights": highlights,
                    "subject_context": subject_context,
                    "missing_context": visible.get("missing_context", []),
                    "interpretation_scope": visible.get("interpretation_scope", ""),
                }
                data_summary = (
                    f"已生成非诊断性的 EEG 研究特征档案；数据质量状态为 {quality}。"
                    "频谱、脑区、时域、左右差异和连接性结果均来自确定性信号处理工具。"
                )
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
                evidence_titles = [
                    str(cast(dict[str, object], item).get("title", ""))
                    for item in raw_items
                    if isinstance(item, dict)
                ]
                evidence_titles = [item for item in evidence_titles if item]
                retrieval_mode = str(visible.get("mode", "unknown"))
                knowledge_status = "available" if evidence_ids else "unavailable"
                if evidence_ids:
                    title_summary = "、".join(evidence_titles[:3])
                    knowledge_summary = (
                        f"师兄 RAG 以 {retrieval_mode} 模式返回 {len(evidence_ids)} 条可追踪背景证据"
                        + (f"，包括：{title_summary}" if title_summary else "")
                        + "；各结论必须以 Evidence ID 回溯，不能作为个案临床证据。"
                    )
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
                "status": data_status,
                "summary": data_summary,
                "source_tools": list(dict.fromkeys(data_tools)),
                "details": data_details,
            },
            "model_assessment": {
                "status": model_status,
                "summary": model_summary,
                "source_tools": model_tools,
                "details": {},
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
