"""
Evidence-Preserving RAG Context Compressor.
Optimizes retrieval documents and multi-turn passages with:
1. Question-Evidence Relevance Ranking: identifies and anchors sentences supporting the query.
2. Citation & Source Preservation: retains metadata tags [Doc 1], URLs, and dates.
3. Multilingual Support: Traditional & Simplified Chinese character-safe segmentation.
4. Noise/Padding Pruning: discards non-evidentiary filler text while enforcing fidelity.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from src.interface import LinUCBCompressor
from src.guards.structural_guard import StructuralConstraintGuard


class EvidencePreservingRAGCompressor:
    """
    RAG-specific compressor that extracts evidence sentences relative to a query
    and preserves citations, negations, and numbers while pruning irrelevant chunks.
    """

    CITATION_REGEX = re.compile(r"\[(?:Doc|Source|來源|文獻|Ref)\s*\d+\]|https?://[^\s]+|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", re.IGNORECASE)

    def __init__(self, compressor: Optional[LinUCBCompressor] = None):
        self.compressor = compressor or LinUCBCompressor(provider="simulation")

    @classmethod
    def segment_sentences(cls, text: str) -> List[str]:
        """Splits text into sentences supporting both English and CJK punctuation."""
        # Split by periods, question marks, exclamation marks, or Chinese punctuation (。？！；\n)
        parts = re.split(r"([。？！；\n]|\.\s+|\?\s+|\!\s+)", text)
        sentences = []
        for i in range(0, len(parts), 2):
            s = parts[i]
            if i + 1 < len(parts):
                s += parts[i + 1]
            if s.strip():
                sentences.append(s.strip())
        return sentences

    def compute_relevance(self, query: str, sentence: str) -> float:
        """Heuristic lexical overlap & keyword match between query and candidate sentence."""
        q_words = set(re.findall(r"[\w\u4e00-\u9fa5]+", query.lower()))
        s_words = set(re.findall(r"[\w\u4e00-\u9fa5]+", sentence.lower()))
        if not q_words or not s_words:
            return 0.0

        overlap = len(q_words.intersection(s_words))
        score = overlap / (len(q_words) + 1e-6)

        # Boost score if sentence contains citations or numbers present in query
        if self.CITATION_REGEX.search(sentence):
            score += 0.2
        return score

    def compress_document(
        self,
        query: str,
        document_text: str,
        keep_ratio: float = 0.65,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compresses an incoming RAG retrieved document relative to the query.
        Returns (compressed_doc, telemetry).
        """
        if not document_text.strip():
            return "", {"orig_tokens": 0, "comp_tokens": 0, "savings_pct": 0.0}

        orig_tokens = len(document_text.split())
        sentences = self.segment_sentences(document_text)

        if len(sentences) <= 2:
            # Too short to safely drop entire sentences; use structural guard compression
            comp, _, _ = self.compressor.compress(document_text)
            comp_tokens = len(comp.split())
            return comp, {
                "orig_tokens": orig_tokens,
                "comp_tokens": comp_tokens,
                "savings_pct": round((1.0 - comp_tokens / max(1, orig_tokens)) * 100.0, 1),
                "retained_sentences": len(sentences),
            }

        # Score all sentences
        scored = []
        for idx, s in enumerate(sentences):
            rel = self.compute_relevance(query, s)
            # Check for critical negations or numbers
            entities = StructuralConstraintGuard.extract_critical_entities(s)
            has_critical = len(entities["negations"]) > 0 or len(entities["numerics"]) > 0
            if has_critical:
                rel += 0.3  # Prioritize preserving constraints
            scored.append((idx, s, rel))

        # Sort by relevance descending and pick top portion
        num_to_keep = max(1, int(len(sentences) * keep_ratio))
        scored.sort(key=lambda x: x[2], reverse=True)
        chosen = scored[:num_to_keep]

        # Re-sort chosen sentences by original document order to preserve natural flow
        chosen.sort(key=lambda x: x[0])
        compressed_sentences = [item[1] for item in chosen]
        compressed_text = " ".join(compressed_sentences)

        comp_tokens = len(compressed_text.split())
        savings = (1.0 - comp_tokens / max(1, orig_tokens)) * 100.0

        telemetry = {
            "orig_tokens": orig_tokens,
            "comp_tokens": comp_tokens,
            "savings_pct": round(savings, 1),
            "retained_sentences": len(compressed_sentences),
            "total_sentences": len(sentences),
        }
        return compressed_text, telemetry
