"""
Unit tests for StructuralConstraintGuard, EvidencePreservingRAGCompressor, and EconomicNetBenefitModel.
"""

from src.guards.structural_guard import StructuralConstraintGuard
from src.rag.evidence_compressor import EvidencePreservingRAGCompressor
from src.economics.economic_model import EconomicNetBenefitModel
from src.utils import calculate_constrained_reward


def test_structural_guard_protects_negation_and_code():
    prompt_with_negation = "Users must NOT delete records older than 30 days without permission."
    entities = StructuralConstraintGuard.extract_critical_entities(prompt_with_negation)

    assert "not" in entities["negations"]
    assert "without" in entities["negations"]
    assert any("30" in num for num in entities["numerics"])

    # If an aggressive compression drops 'not', verify verification fails and fallbacks
    corrupted_prompt = "Users delete records older than 30 days permission."
    is_valid, reason = StructuralConstraintGuard.verify_constraint_preservation(prompt_with_negation, corrupted_prompt)
    assert is_valid is False
    assert "negation" in reason


def test_structural_guard_safe_code_fallback():
    code_prompt = """```python
def solve(x):
    if x <= 0:
        return 0
    return x + 1
```"""
    # Verify valid AST
    assert StructuralConstraintGuard.validate_code_blocks(code_prompt) is True

    # Bad syntax code
    broken_code = "def solve(x): if x <= 0 return 0"
    assert StructuralConstraintGuard.validate_code_blocks(broken_code) is False


def test_constrained_reward_penalizes_failure():
    # Success with 40% savings
    r_success = calculate_constrained_reward(100, 60, task_success=True, quality_score=0.95)
    assert r_success > 0.0

    # Failure with 40% savings should be severely penalized (-3.0)
    r_fail = calculate_constrained_reward(100, 60, task_success=False)
    assert r_fail == -3.0


def test_evidence_preserving_rag():
    rag_comp = EvidencePreservingRAGCompressor()
    query = "What are the isolation levels in PostgreSQL?"
    doc = (
        "PostgreSQL supports Read Committed, Repeatable Read, and Serializable isolation levels. [Doc 1] "
        "The weather in Seattle is rainy and cloudy today. "
        "HOT updates reduce write amplification."
    )

    compressed_doc, telemetry = rag_comp.compress_document(query, doc, keep_ratio=0.7)
    assert "PostgreSQL" in compressed_doc
    assert "isolation levels" in compressed_doc
    assert telemetry["savings_pct"] >= 0


def test_economic_net_benefit_model():
    eval_res = EconomicNetBenefitModel.evaluate_net_cost(
        model_name="gpt-4o",
        original_tokens=2000,
        compressed_tokens=1200,
        compression_latency_ms=0.08,
        task_success_rate=0.99,
        is_cache_hit=False,
    )
    assert eval_res["is_net_profitable"] is True
    assert eval_res["net_savings_usd"] > 0
