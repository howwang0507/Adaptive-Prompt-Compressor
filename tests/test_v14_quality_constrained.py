"""
Unit tests for StructuralConstraintGuard, EvidencePreservingRAGCompressor, and EconomicNetBenefitModel.
Strictly verifies counter-examples, strict barrier reward, mandatory citation retention, and net benefit.
"""

from src.guards.structural_guard import StructuralConstraintGuard
from src.rag.evidence_compressor import EvidencePreservingRAGCompressor
from src.economics.economic_model import EconomicNetBenefitModel
from src.utils import calculate_constrained_reward


def test_structural_guard_counter_examples():
    """Validates that all critical counter-examples across releases are strictly caught and rejected."""
    counter_examples = [
        # Release 1.4.1 counter-examples
        ("不得刪除資料", "刪除資料"),
        ("Keep x <= 30 kg", "Keep x > 30 g"),
        ("Keep 30 days", "Keep 300 days"),
        ("Do not delete records", "Do nothing; delete records"),
        # Release 1.4.2 counter-examples
        ("保留30天", "保留300天"),
        ("Pay $100", "Pay €100"),
        ("Keep x = -30", "Keep x = 30"),
        ("Do not delete A. Delete B.", "Delete A. Do not delete B."),
        ("不得刪除A。可以刪除B。", "可以刪除A。不得刪除B。"),
    ]

    for orig, comp in counter_examples:
        is_valid, reason = StructuralConstraintGuard.verify_constraint_preservation(orig, comp)
        assert is_valid is False, f"Failed on: {orig} -> {comp}"
        assert len(reason) > 0

    valid_preservations = [
        ("保留30天，費用 $100", "保留30天，費用 $100"),
        ("Keep x = -30 kg", "Keep x = -30 kg"),
        ("Please do not delete record A. Also, ensure you delete record B.", "Do not delete record A. Delete record B."),
    ]
    for orig, comp in valid_preservations:
        is_valid, reason = StructuralConstraintGuard.verify_constraint_preservation(orig, comp)
        assert is_valid is True, f"Valid case rejected: {orig} -> {comp} ({reason})"


def test_structural_guard_safe_code_fallback():
    code_prompt = """```python
def solve(x):
    if x <= 0:
        return 0
    return x + 1
```"""
    assert StructuralConstraintGuard.validate_code_blocks(code_prompt) is True

    # Bad syntax code
    broken_code = "def solve(x): if x <= 0 return 0"
    assert StructuralConstraintGuard.validate_code_blocks(broken_code) is False


def test_strict_barrier_constrained_reward():
    """
    Verifies that quality violation strictly receives a NEGATIVE reward,
    even if token savings are extremely high (e.g. 80%).
    """
    # 80% savings, but quality drops to 0.80 below 0.90 tolerance:
    # Must be strictly negative (no positive reward loophole)
    r_violation = calculate_constrained_reward(
        base_tokens=100,
        comp_tokens=20,
        task_success=True,
        quality_score=0.80,
        baseline_quality=1.0,
        quality_tolerance=0.90,
    )
    assert r_violation < 0.0, f"Violation must receive negative reward, got: {r_violation}"

    # Success with 40% savings and quality satisfying tolerance (0.95 >= 0.90)
    r_success = calculate_constrained_reward(
        base_tokens=100,
        comp_tokens=60,
        task_success=True,
        quality_score=0.95,
        baseline_quality=1.0,
        quality_tolerance=0.90,
    )
    assert r_success > 0.0

    # Hard task failure (e.g. execution crash)
    r_fail = calculate_constrained_reward(100, 60, task_success=False)
    assert r_fail <= -5.0


def test_evidence_preserving_rag_mandatory_citations():
    rag_comp = EvidencePreservingRAGCompressor()
    query = "What are the isolation levels in PostgreSQL?"
    doc = (
        "PostgreSQL supports Read Committed, Repeatable Read, and Serializable isolation levels. [Doc 1] "
        "The weather in Seattle is rainy and cloudy today. "
        "HOT updates reduce write amplification. [Doc 2]"
    )

    # Even with aggressive 40% keep ratio, sentences with citations [Doc 1] and [Doc 2] MUST be retained!
    compressed_doc, telemetry = rag_comp.compress_document(query, doc, keep_ratio=0.4)
    assert "[Doc 1]" in compressed_doc
    assert "[Doc 2]" in compressed_doc
    assert "PostgreSQL" in compressed_doc


def test_economic_net_benefit_model():
    eval_res = EconomicNetBenefitModel.evaluate_net_cost(
        model_name="gpt-4o",
        original_input_tokens=2000,
        compressed_input_tokens=1200,
        output_tokens=150,
        compression_latency_ms=0.08,
        verification_latency_ms=0.02,
        task_success_rate=0.99,
        cache_hit_ratio=0.5,
    )
    assert eval_res["is_net_profitable"] is True
    assert eval_res["net_savings_usd"] > 0
    assert "baseline_total_cost_usd" in eval_res
    assert "total_effective_cost_usd" in eval_res


def test_evidence_preserving_rag_segmentation_robustness():
    """
    Verifies that sentence segmentation never drops trailing unpunctuated sentences
    and never splits floating numbers (3.14) or URLs (https://example.com/report).
    """
    # 1. Trailing unpunctuated sentence preservation
    text1 = "First. Second. Third. Final evidence without punctuation"
    sents1 = EvidencePreservingRAGCompressor.segment_sentences(text1)
    assert len(sents1) == 4
    assert sents1[0] == "First."
    assert sents1[1] == "Second."
    assert sents1[2] == "Third."
    assert sents1[3] == "Final evidence without punctuation"

    # 2. Decimals and URLs protected from splitting
    text2 = "Value is 3.14. See https://example.com/report for details. Release v1.4.2 is live."
    sents2 = EvidencePreservingRAGCompressor.segment_sentences(text2)
    assert len(sents2) == 3
    assert sents2[0] == "Value is 3.14."
    assert sents2[1] == "See https://example.com/report for details."
    assert sents2[2] == "Release v1.4.2 is live."

    # 3. Multilingual Chinese mixed with citations
    text3 = "這是第一句 [Doc 1]。第二句數值是 3.14。最後是總結沒有標點"
    sents3 = EvidencePreservingRAGCompressor.segment_sentences(text3)
    assert len(sents3) == 3
    assert "[Doc 1]" in sents3[0]
    assert "3.14" in sents3[1]
    assert sents3[2] == "最後是總結沒有標點"

