"""
JSON Mode & Structured Outputs Integrity Guard.
Ensures that prompt compression preserves JSON schema structure, enum values,
property keys, and type constraints when targeting structured output LLMs.
"""

import json
import re
from typing import Any, Dict, List, Tuple


class JSONSchemaGuard:
    """
    Protects JSON schemas and Structured Output directives from lossy compression.
    Identifies JSON blocks or schema fragments, isolates keys, types, and enums,
    and applies compression only to verbose descriptive text.
    """

    # Regular expression to extract code blocks or raw JSON structures
    JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

    @classmethod
    def extract_json_blocks(cls, text: str) -> List[Tuple[str, int, int]]:
        """Finds markdown-wrapped JSON objects."""
        matches = []
        for match in cls.JSON_BLOCK_REGEX.finditer(text):
            matches.append((match.group(1), match.start(), match.end()))
        return matches

    @classmethod
    def is_valid_json(cls, text: str) -> bool:
        """Validates if a text slice is valid JSON."""
        try:
            json.loads(text.strip())
            return True
        except (ValueError, TypeError):
            return False

    @classmethod
    def sanitize_schema_descriptions(cls, schema: Dict[str, Any], compress_fn) -> Dict[str, Any]:
        """
        Recursively traverses a JSON Schema dictionary.
        Leaves structural keys ('properties', 'required', 'type', 'enum') untouched,
        and safely compresses 'description' fields using the bandit compressor.
        """
        if not isinstance(schema, dict):
            return schema

        updated_schema = {}
        for key, val in schema.items():
            if key == "description" and isinstance(val, str):
                # Safely compress description text
                compressed_desc, _, _ = compress_fn(val)
                updated_schema[key] = compressed_desc.strip()
            elif isinstance(val, dict):
                updated_schema[key] = cls.sanitize_schema_descriptions(val, compress_fn)
            elif isinstance(val, list):
                updated_schema[key] = [
                    cls.sanitize_schema_descriptions(item, compress_fn) if isinstance(item, dict) else item
                    for item in val
                ]
            else:
                updated_schema[key] = val

        return updated_schema

    @classmethod
    def compress_structured_prompt(cls, prompt: str, compress_fn) -> Tuple[str, bool]:
        """
        Safely compresses a prompt containing embedded JSON schemas or example payloads.
        Returns:
            (processed_prompt, was_json_detected)
        """
        json_blocks = cls.extract_json_blocks(prompt)
        if not json_blocks:
            # Check if entire prompt is a raw JSON string
            if cls.is_valid_json(prompt):
                data = json.loads(prompt)
                if isinstance(data, dict):
                    guarded = cls.sanitize_schema_descriptions(data, compress_fn)
                    return json.dumps(guarded, indent=2), True
            return prompt, False

        processed = prompt
        offset = 0
        for block_content, start, end in json_blocks:
            if cls.is_valid_json(block_content):
                data = json.loads(block_content)
                guarded = cls.sanitize_schema_descriptions(data, compress_fn)
                replacement = "```json\n" + json.dumps(guarded, indent=2) + "\n```"
                actual_start = start + offset
                actual_end = end + offset
                processed = processed[:actual_start] + replacement + processed[actual_end:]
                offset += len(replacement) - (end - start)

        return processed, True
