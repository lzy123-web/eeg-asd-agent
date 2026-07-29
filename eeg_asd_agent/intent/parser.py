from eeg_asd_agent.schemas import RequestedGoal


class RuleBasedIntentParser:
    """Deterministic multilingual goal parser used by the offline profile."""

    def parse(self, request: str) -> RequestedGoal:
        normalized = request.strip().lower()
        if any(word in normalized for word in ("确诊", "diagnose", "clinical diagnosis", "处方")):
            return RequestedGoal.UNSUPPORTED
        if any(word in normalized for word in ("干预", "训练建议", "intervention")):
            return RequestedGoal.INTERVENTION
        if any(word in normalized for word in ("质量", "quality")):
            return RequestedGoal.QUALITY_CHECK
        if (
            any(word in normalized for word in ("特征", "feature"))
            and any(word in normalized for word in ("只", "only", "extract", "提取"))
        ):
            return RequestedGoal.FEATURE_EXTRACTION
        if any(word in normalized for word in ("知识", "检索", "knowledge", "retrieval", "rag")):
            return RequestedGoal.KNOWLEDGE_SEARCH
        if (
            "report only" in normalized
            or ("报告" in normalized and "只" in normalized)
            or ("report" in normalized and "only" in normalized)
        ):
            return RequestedGoal.REPORT_ONLY
        if any(
            word in normalized
            for word in ("评估", "分析", "asd", "autism", "eeg", "完整", "assessment")
        ):
            return RequestedGoal.FULL_ASSESSMENT
        return RequestedGoal.UNSUPPORTED
