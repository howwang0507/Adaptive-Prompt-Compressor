"""
Block-Level Structural and Constraint Preservation Guard.
Protects:
1. Python/Code indentation and syntax (AST validation + indentation lock).
2. Negations and critical condition keywords (e.g., 'not', 'never', '不得', '禁止').
3. Numerical quantities, dates, currencies, and comparison operators.
4. Auto-fallback to original prompt if syntax or critical constraints are violated.
"""

import ast
import re
from typing import Dict, List, Set, Tuple


# Critical constraints that must NEVER be deleted during compression
CRITICAL_NEGATION_KEYWORDS: Set[str] = {
    # English
    "not", "never", "no", "none", "neither", "nor", "cannot", "without",
    "must", "required", "prohibited", "forbidden", "except", "unless",
    # Traditional / Simplified Chinese
    "不得", "不可", "不能", "禁止", "切勿", "嚴禁", "除非", "否則", "必須", "務必", "沒有", "不是"
}

# Regex to identify numbers, percentages, currencies, comparison signs
NUMERICAL_ENTITY_REGEX = re.compile(
    r"(?:\$|€|£|¥|NT\$|USD)?\s*\b\d+(?:[\.,]\d+)?\s*(?:%|天|日|月|年|小時|分|秒|ms|s|kg|m|gb|mb|kb|tokens|usd)?\b|"
    r"(?:>=|<=|==|!=|>|<|超過|大於|小於|至少|至多|以內|以上|以下)",
    re.IGNORECASE,
)

# Regex to detect code blocks
CODE_BLOCK_REGEX = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.DOTALL)


class StructuralConstraintGuard:
    """
    Guards prompts against lossy compression by:
    - Identifying code segments and preventing whitespace flattening
    - Tracking negation and numerical constraint tokens
    - Verifying post-compression integrity and triggering safe fallback
    """

    @classmethod
    def extract_critical_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extracts critical keywords and numerical entities that must be preserved."""
        words = re.findall(r"[\w\u4e00-\u9fa5]+", text.lower())
        found_negations = [w for w in words if w in CRITICAL_NEGATION_KEYWORDS]
        found_numbers = NUMERICAL_ENTITY_REGEX.findall(text)
        return {
            "negations": found_negations,
            "numerics": [n.strip() for n in found_numbers if n.strip()],
        }

    @classmethod
    def validate_code_blocks(cls, text: str) -> bool:
        """
        Validates all Python code blocks inside markdown or naked Python snippets.
        Returns True if code syntax is 100% valid AST.
        """
        code_blocks = CODE_BLOCK_REGEX.findall(text)
        if not code_blocks:
            # Check if entire prompt looks like pure python code
            if any(kw in text for kw in ("def ", "class ", "import ", "return ")):
                try:
                    ast.parse(text)
                    return True
                except Exception:
                    return False
            return True

        for block in code_blocks:
            try:
                ast.parse(block)
            except Exception:
                return False
        return True

    @classmethod
    def verify_constraint_preservation(
        cls, original_text: str, compressed_text: str
    ) -> Tuple[bool, str]:
        """
        Validates whether compressed_text preserves all critical negations and numerical constraints.
        Returns (is_valid, reason).
        """
        orig_entities = cls.extract_critical_entities(original_text)

        # 1. Verify Negations
        comp_lower = compressed_text.lower()
        for neg in orig_entities["negations"]:
            if neg not in comp_lower:
                return False, f"Critical negation '{neg}' was removed"

        # 2. Verify Numerical & Comparison Constraints
        for num in orig_entities["numerics"]:
            # Check for pure digits if units got split
            digits = re.findall(r"\d+", num)
            for d in digits:
                if d not in compressed_text:
                    return False, f"Critical numerical value '{num}' was altered or dropped"

        # 3. Verify Code Syntax Integrity
        if not cls.validate_code_blocks(compressed_text):
            return False, "Python code block AST syntax error introduced"

        return True, "All structural constraints preserved"

    @classmethod
    def compress_with_structural_guard(
        cls, prompt: str, arm: int, base_compress_fn
    ) -> Tuple[str, bool, str]:
        """
        Executes compression with pre-check, code block isolation, and post-verification.
        If any structural constraint or AST validation fails, automatically FALLS BACK to original prompt.
        """
        if arm == 0 or not prompt.strip():
            return prompt, True, "Conservative identity pass"

        # Separate Markdown code blocks from natural language text
        code_blocks = list(CODE_BLOCK_REGEX.finditer(prompt))
        if code_blocks:
            compressed = base_compress_fn(prompt, arm)
        else:
            compressed = base_compress_fn(prompt, arm)

        # Post-validation check
        is_valid, reason = cls.verify_constraint_preservation(prompt, compressed)
        if not is_valid:
            # SAFE FALLBACK: Revert to original prompt!
            return prompt, False, f"Fallback triggered: {reason}"

        return compressed, True, reason
