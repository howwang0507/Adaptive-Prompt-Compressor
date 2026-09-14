"""
Block-Level Structural and Constraint Preservation Guard.
Solves critical false-positive and leakage vulnerabilities:
1. Exact Word-Boundary Matching: prevents 'not' from being satisfied by 'nothing' or 'notice'.
2. CJK / Chinese Substring Matching: accurately detects Chinese negations like '不得', '禁止', '切勿' regardless of spacing.
3. Strict Numerical & Comparison Tuple Binding: matches (operator, number, unit) as an unbroken entity (e.g. '<= 30 kg' vs '> 30 g', '30 days' vs '300 days').
4. Genuine Code Block Isolation: extracts markdown code blocks (```...```) into immutable placeholders,
   compresses only the prose, and restores the exact indented code byte-for-byte.
5. Strict AST & Invariant Validation: triggers automatic safe fallback to original prompt on any violation.
"""

import ast
import re
from typing import Any, Dict, Set, Tuple


# Critical constraints that must NEVER be deleted during compression
CRITICAL_NEGATION_KEYWORDS_EN: Set[str] = {
    "not", "never", "no", "none", "neither", "nor", "cannot", "without",
    "must", "required", "prohibited", "forbidden", "except", "unless"
}

CRITICAL_NEGATION_KEYWORDS_ZH: Set[str] = {
    "不得", "不可", "不能", "禁止", "切勿", "嚴禁", "除非", "否則", "必須", "務必", "沒有", "不是"
}

# Regex to identify comparison operators, numbers, and bound units as atomic entities
BOUND_CONSTRAINT_REGEX = re.compile(
    r"(?P<op>>=|<=|==|!=|>|<|超過|大於|小於|至少|至多|以內|以上|以下)?\s*"
    r"(?P<curr>\$|€|£|¥|NT\$|USD)?\s*"
    r"\b(?P<num>\d+(?:[\.,]\d+)?)\b\s*"
    r"(?P<unit>%|天|日|月|年|小時|分|秒|ms|s|kg|g|m|gb|mb|kb|tokens|usd|days)?",
    re.IGNORECASE,
)

# Regex to detect code blocks
CODE_BLOCK_REGEX = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.DOTALL)


class StructuralConstraintGuard:
    """
    Guards prompts against lossy compression with zero-tolerance for constraint violation.
    """

    @classmethod
    def extract_critical_entities(cls, text: str) -> Dict[str, Any]:
        """
        Extracts negations and bound constraint tuples (op, curr, num, unit).
        """
        # 1. English negations using word boundaries
        found_en_negations = []
        for kw in CRITICAL_NEGATION_KEYWORDS_EN:
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                found_en_negations.append(kw.lower())

        # 2. Chinese negations using direct character presence
        found_zh_negations = []
        for kw in CRITICAL_NEGATION_KEYWORDS_ZH:
            if kw in text:
                found_zh_negations.append(kw)

        # 3. Bound Numerical Constraints
        bound_entities = []
        for match in BOUND_CONSTRAINT_REGEX.finditer(text):
            op = (match.group("op") or "").strip()
            curr = (match.group("curr") or "").strip()
            num = (match.group("num") or "").strip()
            unit = (match.group("unit") or "").strip().lower()
            if num:
                bound_entities.append({
                    "raw": match.group(0).strip(),
                    "op": op,
                    "curr": curr,
                    "num": num,
                    "unit": unit,
                })

        return {
            "en_negations": found_en_negations,
            "zh_negations": found_zh_negations,
            "bound_entities": bound_entities,
        }

    @classmethod
    def validate_code_blocks(cls, text: str) -> bool:
        """
        Validates Python code blocks or raw code snippets.
        Returns True if code syntax is 100% valid AST.
        """
        code_blocks = CODE_BLOCK_REGEX.findall(text)
        if not code_blocks:
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
        Validates whether compressed_text strictly preserves all critical negations,
        bound constraints (numbers, operators, units), and code syntax.
        """
        orig_entities = cls.extract_critical_entities(original_text)

        # 1. Verify English Negations with Word Boundaries
        for neg in orig_entities["en_negations"]:
            if not re.search(rf"\b{re.escape(neg)}\b", compressed_text, re.IGNORECASE):
                return False, f"Critical English negation '{neg}' was removed or altered"

        # 2. Verify Chinese Negations
        for neg in orig_entities["zh_negations"]:
            if neg not in compressed_text:
                return False, f"Critical Chinese negation '{neg}' was removed"

        # 3. Verify Bound Numerical & Comparison Constraints
        comp_entities = cls.extract_critical_entities(compressed_text)["bound_entities"]
        for orig in orig_entities["bound_entities"]:
            # Check if there is an exact matching bound entity in compressed text
            matched = False
            for comp in comp_entities:
                # Number must match exactly (prevent 30 becoming 300)
                if orig["num"] == comp["num"]:
                    # If operator was specified, operator must match
                    if orig["op"] and orig["op"] != comp["op"]:
                        continue
                    # If unit was specified, unit must match (prevent kg becoming g)
                    if orig["unit"] and orig["unit"] != comp["unit"]:
                        continue
                    matched = True
                    break
            if not matched:
                return False, f"Bound constraint '{orig['raw']}' was corrupted or dropped"

        # 4. Verify Code Syntax Integrity
        if not cls.validate_code_blocks(compressed_text):
            return False, "Python code block AST syntax error introduced"

        return True, "All structural constraints preserved"

    @classmethod
    def compress_with_structural_guard(
        cls, prompt: str, arm: int, base_compress_fn
    ) -> Tuple[str, bool, str]:
        """
        Executes compression with genuine code block isolation and strict post-verification.
        If any structural constraint or AST validation fails, automatically FALLS BACK to original prompt.
        """
        if arm == 0 or not prompt.strip():
            return prompt, True, "Conservative identity pass"

        # Isolate Markdown Code Blocks via Placeholders
        code_blocks = list(CODE_BLOCK_REGEX.finditer(prompt))
        if code_blocks:
            placeholders = {}
            temp_prompt = prompt
            for idx, match in enumerate(code_blocks):
                token = f"__CODE_BLOCK_PLACEHOLDER_{idx}__"
                placeholders[token] = match.group(0)
                temp_prompt = temp_prompt.replace(match.group(0), token)

            # Compress only prose surrounding the placeholders
            compressed_temp = base_compress_fn(temp_prompt, arm)

            # Restore exact code blocks byte-for-byte
            for token, code in placeholders.items():
                compressed_temp = compressed_temp.replace(token, code)
            compressed = compressed_temp
        else:
            compressed = base_compress_fn(prompt, arm)

        # Post-validation check
        is_valid, reason = cls.verify_constraint_preservation(prompt, compressed)
        if not is_valid:
            # SAFE FALLBACK: Revert to original prompt!
            return prompt, False, f"Fallback triggered: {reason}"

        return compressed, True, reason
