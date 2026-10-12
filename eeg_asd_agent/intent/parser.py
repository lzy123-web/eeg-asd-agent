from eeg_asd_agent.schemas import RequestedGoal


class RuleBasedIntentParser:
    """Deterministic multilingual goal parser used by the offline profile."""

    def parse(self, request: str) -> RequestedGoal:
        normalized = request.strip().lower()
        analysis_negated = any(
            phrase in normalized
            for phrase in ("不重新分析", "不要分析", "无需分析", "without analysis")
        )
        full_requested = any(
            word in normalized for word in ("评估", "分析", "assessment", "analy", "完整")
        ) and not analysis_negated
        if any(word in normalized for word in ("确诊", "diagnose", "clinical diagnosis", "处方")):
            return RequestedGoal.UNSUPPORTED
        if (
            any(word in normalized for word in ("具备执行条件", "模型是否具备", "readiness"))
            and any(word in normalized for word in ("检查", "check", "只"))
        ):
            return RequestedGoal.READINESS_CHECK
        # A full report commonly asks to include signal quality.  Do not let
        # that requested section collapse the whole goal to a quality-only
        # run; quality_check is reserved for requests without a broader
        # assessment/analysis intent.
        if (
            any(word in normalized for word in ("质量", "quality"))
            and not full_requested
        ):
            return RequestedGoal.QUALITY_CHECK
        if (
            any(word in normalized for word in ("特征", "feature"))
            and any(word in normalized for word in ("只", "only", "extract", "提取"))
        ):
            return RequestedGoal.FEATURE_EXTRACTION
        knowledge_requested = any(
            word in normalized for word in ("知识", "检索", "knowledge", "retrieval", "rag")
        )
        knowledge_negated = any(
            phrase in normalized
            for phrase in ("不检索", "不要检索", "无需检索", "without retrieval", "no rag")
        )
        if any(word in normalized for word in ("干预", "训练建议", "intervention")):
            return RequestedGoal.INTERVENTION
        if knowledge_requested and not knowledge_negated and not full_requested:
            return RequestedGoal.KNOWLEDGE_SEARCH
        if (
            "report only" in normalized
            or ("报告" in normalized and "只" in normalized)
            or ("report" in normalized and "only" in normalized)
        ):
            return RequestedGoal.REPORT_ONLY
        if full_requested:
            return RequestedGoal.FULL_ASSESSMENT
        if knowledge_requested:
            return RequestedGoal.KNOWLEDGE_SEARCH
        if any(word in normalized for word in ("asd", "autism", "eeg")):
            return RequestedGoal.FULL_ASSESSMENT
        return RequestedGoal.UNSUPPORTED
