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
        """
        Splits text into sentences supporting English and CJK punctuation,
        protecting URLs, decimals, and versions, while preserving trailing unpunctuated text
        and binding citations (e.g. '[Doc 1]') to their respective sentences.
        """
        if not text or not text.strip():
            return []

        urls = []

        def url_rep(m):
            url = m.group(0)
            trailing = ""
            while url and url[-1] in ".,;:!?)":
                trailing = url[-1] + trailing
                url = url[:-1]
            urls.append(url)
            return f"__URL_{len(urls)-1}__{trailing}"

        s = re.sub(r"https?://\S+", url_rep, text)

        decimals = []

        def dec_rep(m):
            decimals.append(m.group(0))
            return f"__DEC_{len(decimals)-1}__"

        s = re.sub(r"(?i)\b(?:v\d+(?:\.\d+)+|\d+(?:\.\d+)+)\b", dec_rep, s)

        pattern = re.compile(
            r"([^。！？；\.\?\!\n]+(?:[。！？；\.\?\!]+(?:\s*\[(?:Doc|Source|來源|文獻|Ref)\s*\d+\])?|\s*$))",
            re.IGNORECASE,
        )
        matches = [m.group(0).strip() for m in pattern.finditer(s) if m.group(0).strip()]

        res = []
        for c in matches:
            for i, u in enumerate(urls):
                c = c.replace(f"__URL_{i}__", u)
            for i, d in enumerate(decimals):
                c = c.replace(f"__DEC_{i}__", d)
            res.append(c)
        return res

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

        # Partition sentences into:
        # 1. Mandatory Retention Set (Sentences containing explicit citations, doc tags, or URLs)
        # 2. Candidate Sentences (Ranked by query relevance)
        mandatory_indices = set()
        candidate_scored = []

        for idx, s in enumerate(sentences):
            has_citation = bool(self.CITATION_REGEX.search(s))
            if has_citation:
                mandatory_indices.add(idx)
            else:
                rel = self.compute_relevance(query, s)
                entities = StructuralConstraintGuard.extract_critical_entities(s)
                if entities["en_negations"] or entities["zh_negations"] or entities["bound_entities"]:
                    rel += 0.25
                candidate_scored.append((idx, s, rel))

        # Calculate remaining quota
        target_total_to_keep = max(len(mandatory_indices), int(len(sentences) * keep_ratio))
        remaining_slots = max(0, target_total_to_keep - len(mandatory_indices))

        candidate_scored.sort(key=lambda x: x[2], reverse=True)
        chosen_candidates = candidate_scored[:remaining_slots]

        # Combine mandatory sentences + top candidates
        chosen_indices = mandatory_indices.union({item[0] for item in chosen_candidates})

        # Re-sort in original sequential document order
        sorted_chosen_indices = sorted(list(chosen_indices))
        compressed_sentences = [sentences[i] for i in sorted_chosen_indices]
        compressed_text = " ".join(compressed_sentences)

        # Post-compression constraint validation & fallback
        is_valid, reason = StructuralConstraintGuard.verify_constraint_preservation(document_text, compressed_text)
        if not is_valid:
            # Safe Fallback: return original document if constraints broken
            return document_text, {
                "orig_tokens": orig_tokens,
                "comp_tokens": orig_tokens,
                "savings_pct": 0.0,
                "retained_sentences": len(sentences),
                "total_sentences": len(sentences),
                "fallback_triggered": True,
                "fallback_reason": reason,
            }

        comp_tokens = len(compressed_text.split())
        savings = (1.0 - comp_tokens / max(1, orig_tokens)) * 100.0

        telemetry = {
            "orig_tokens": orig_tokens,
            "comp_tokens": comp_tokens,
            "savings_pct": round(savings, 1),
            "retained_sentences": len(compressed_sentences),
            "total_sentences": len(sentences),
            "fallback_triggered": False,
        }
        return compressed_text, telemetry

