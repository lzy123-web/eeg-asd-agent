from __future__ import annotations

from dataclasses import dataclass

from eeg_asd_agent.schemas import (
    AssessmentCaseState,
    ImplementationKind,
    RequestedGoal,
    StateUpdate,
    ToolCapabilityManifest,
    ToolExecutionOutcome,
)
from eeg_asd_agent.tools.base import FunctionTool
from eeg_asd_agent.tools.core_workflow import (
    KnowledgeAdapter,
    ReportProvider,
    build_core_registry,
)
from eeg_asd_agent.tools.registry import ToolRegistry, unavailable_result


@dataclass(frozen=True)
class DomainCapabilityStub:
    name: str
    category: str
    description: str
    outputs: tuple[str, ...]
    use_when: tuple[str, ...]


DOMAIN_CAPABILITY_STUBS = (
    DomainCapabilityStub(
        "analyze_eeg_patterns",
        "eeg_analysis",
        "Analyze ASD-relevant EEG patterns beyond the current basic feature extractor.",
        ("pattern_summary", "method_version", "limitations"),
        ("The user asks for an EEG pattern or biomarker-oriented analysis.",),
    ),
    DomainCapabilityStub(
        "estimate_asd_severity",
        "assessment",
        "Run a versioned research-only ASD severity estimator when a validated model is available.",
        ("severity_estimate", "model_version", "limitations"),
        ("The user explicitly requests a research severity estimate.",),
    ),
    DomainCapabilityStub(
        "estimate_ability_dimensions",
        "assessment",
        "Estimate research ability dimensions using a dedicated validated EEG-ASD model.",
        ("ability_profile", "model_version", "limitations"),
        ("The user requests an EEG-derived ability profile and a compatible model exists.",),
    ),
    DomainCapabilityStub(
        "compare_assessment_results",
        "assessment",
        "Compare imported and newly generated EEG-ASD results without silently replacing either.",
        ("comparison", "agreements", "differences", "limitations"),
        ("The user asks to validate, reproduce, audit, or compare an existing result.",),
    ),
    DomainCapabilityStub(
        "parse_assessment_report",
        "document",
        "Extract a typed, source-linked summary from an existing assessment report.",
        ("report_summary", "extracted_findings", "limitations"),
        ("An existing report is relevant to the current user goal.",),
    ),
    DomainCapabilityStub(
        "conduct_structured_assessment",
        "assessment",
        "Conduct an optional structured ASD-related assessment as an enrichment tool.",
        ("assessment_result", "instrument_version", "limitations"),
        ("The user requests an additional structured assessment; it is never required for EEG-only flow.",),
    ),
    DomainCapabilityStub(
        "visualize_eeg_analysis",
        "visualization",
        "Render deterministic EEG quality, spectrum, feature, and comparison figures.",
        ("figure_artifacts", "renderer_version"),
        ("The user asks for visual EEG-ASD analysis artifacts.",),
    ),
    DomainCapabilityStub(
        "train_asd_research_model",
        "model_research",
        "Train and evaluate a reproducible EEG-ASD research model under subject-level splitting.",
        ("model_artifact", "evaluation", "split_manifest", "limitations"),
        ("The user asks for a model-development task rather than current-case assessment.",),
    ),
)


def register_domain_capability_stubs(registry: ToolRegistry) -> ToolRegistry:
    goals = list(RequestedGoal)
    for item in DOMAIN_CAPABILITY_STUBS:
        manifest = ToolCapabilityManifest(
            name=item.name,
            version="architecture-v1",
            category=item.category,
            description=item.description,
            use_when=list(item.use_when),
            avoid_when=["The capability is not enabled in the current implementation profile."],
            tags=["eeg-asd", "domain-capability", item.category],
            produced_outputs=list(item.outputs),
            supported_goals=goals,
            unsupported_claims=[
                "clinical ASD diagnosis",
                "treatment prescription",
                "validated prospective performance without a frozen model contract",
            ],
            limitations=[
                "Architecture placeholder only; this capability has not been implemented."
            ],
            allowlisted_fields={goal.value: [] for goal in goals},
            retry_limit=0,
            implementation_reference=(
                f"eeg_asd_agent.tools.domain_catalog:{item.name}"
            ),
            real_or_fake=ImplementationKind.STUB,
            enabled=False,
            repeatable=True,
        )

        def unavailable(
            _state: AssessmentCaseState,
            _arguments: dict[str, object],
            *,
            active_manifest: ToolCapabilityManifest = manifest,
        ) -> ToolExecutionOutcome:
            result = unavailable_result(
                active_manifest,
                "Capability is declared by the EEG-ASD Agent architecture but not implemented.",
            )
            return ToolExecutionOutcome(
                result=result,
                state_update=StateUpdate(
                    failed_workflows_add=[active_manifest.name],
                    limitations_add=result.limitations,
                ),
            )

        registry.register(FunctionTool(manifest, unavailable))
    return registry


def build_agent_registry(
    *,
    report_provider: ReportProvider | None = None,
    knowledge_adapter: KnowledgeAdapter | None = None,
) -> ToolRegistry:
    registry = build_core_registry(
        report_provider=report_provider,
        knowledge_adapter=knowledge_adapter,
    )
    return register_domain_capability_stubs(registry)
