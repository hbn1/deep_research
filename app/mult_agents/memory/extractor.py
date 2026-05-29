"""
LLM-driven memory extractor.

Replaces keyword-based extraction with structured LLM extraction.
Includes rule-based fallback for when LLM is unavailable.
"""

import json as _json
import logging as _logging
import re as _re
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

_logger = _logging.getLogger("mult_agents.memory")


# Unicode character constants for Chinese punctuation
C_PERIOD = "。"   # ?
C_EXCL = "！"     # ?
C_QUEST = "？"    # ?
C_SEMIC = "；"    # ?
C_IS = "是"       # ?
C_HAS = "有"      # ?
C_LOCATED = "位于"  # ??
C_RESP = "负责"     # ??
C_USE = "使用"      # ??
C_CALLED = "叫"         # ?
C_DO = "做"             # ?
C_STUDY = "学"          # ?
C_AT = "在"             # ?
C_LIKE = "喜欢"     # ??
C_PREF = "偏好"     # ??
C_WANT = "想要"     # ??
C_HOPE = "希望"     # ??
C_DISLIKE = "不喜欢"  # ???
C_HATE = "讨厌"     # ??
C_ALWAYS_F = "以后都"  # ???
C_ALWAYS = "总是"   # ??
C_EVERY = "每次"    # ??
C_DONT = "不要"     # ??
C_DONT2 = "别"          # ?
C_REMEMBER = "记住" # ??
C_LATER_U = "以后你"  # ???
C_YOUR_PREF = "你的偏好"  # ????
C_YOU_WANT = "你要" # ??
C_ANS_PREF = "回答偏好"  # ????
C_MY_NAME = "我叫"  # ??
C_I_AM = "我是"     # ??
C_MY_NAME2 = "我的名字"  # ????
C_WHAT = "什么"     # ??
C_MA = "吗"             # ?

SENTENCE_SPLIT = "[" + C_PERIOD + C_EXCL + C_QUEST + "!?\n" + C_SEMIC + ";]"


class RuleBasedExtractor:
    """Rule-based fallback extractor."""

    FACT_INDICATORS = [
        C_IS, C_HAS, C_LOCATED, C_RESP, C_USE, C_CALLED, C_DO, C_STUDY, C_AT,
        "my name is", "i am", "i'm", "i work", "i study",
    ]
    PREF_INDICATORS = [
        C_LIKE, C_PREF, C_WANT, C_HOPE, C_DISLIKE, C_HATE,
        "prefer", "i like", "i love", "i hate", "style",
    ]
    CONSTRAINT_INDICATORS = [
        C_ALWAYS_F, C_ALWAYS, C_EVERY, C_DONT, C_DONT2, C_REMEMBER,
        C_LATER_U, C_YOUR_PREF, C_YOU_WANT, C_ANS_PREF,
        "always", "never", "remember",
    ]

    def extract(self, query, answer=""):
        combined = "{} {}".format(query, answer[:500])
        return {
            "facts": self._extract_by_indicators(combined, self.FACT_INDICATORS),
            "preferences": self._extract_by_indicators(combined, self.PREF_INDICATORS),
            "constraints": self._extract_by_indicators(combined, self.CONSTRAINT_INDICATORS),
            "procedural": [],
            "importance": self._estimate_importance(combined),
        }

    def _extract_by_indicators(self, text, indicators):
        lowered = text.lower()
        results = []
        for indicator in indicators:
            if indicator.lower() in lowered:
                sentences = [s.strip() for s in _re.split(SENTENCE_SPLIT, text) if s.strip()]
                for sent in sentences:
                    sent_lower = sent.lower()
                    if indicator.lower() in sent_lower and len(sent) > 3:
                        if any(q in sent_lower for q in ["?", C_QUEST, C_WHAT, C_MA, "how", "what"]):
                            continue
                        if sent not in results:
                            results.append(sent)
        return results[:5]

    def _estimate_importance(self, text):
        lowered = text.lower()
        high = [C_MY_NAME, C_I_AM, C_MY_NAME2, "my name is", "i am", C_REMEMBER]
        medium = [C_LIKE, C_PREF, C_HOPE, "prefer", "i like"]
        if any(ind in lowered for ind in high):
            return 0.85
        if any(ind in lowered for ind in medium):
            return 0.6
        return 0.3


class MemoryExtractor:
    """LLM-driven memory extractor with rule-based fallback."""

    def __init__(self, llm=None):
        self._llm = llm
        self._fallback = RuleBasedExtractor()

    def extract_from_turn(self, query, answer, existing_profile=None, existing_procedural=None):
        if self._llm:
            try:
                return self._extract_with_llm(query, answer, existing_profile, existing_procedural)
            except Exception as exc:
                _logger.warning("LLM extraction failed, fallback to rules: %s", exc)
        return self._fallback.extract(query, answer)

    def _extract_with_llm(self, query, answer, existing_profile, existing_procedural):
        profile_str = _json.dumps(existing_profile, ensure_ascii=False) if existing_profile else "none"
        procedural_str = _json.dumps(existing_procedural or [], ensure_ascii=False) if existing_procedural else "none"
        prompt = (
            "You are a memory extraction engine. From the following conversation, "
            "extract structured long-term memories as JSON.\n"
            "Existing profile: {}\n".format(profile_str) +
            "Existing patterns: {}\n".format(procedural_str) +
            "User: {}\n".format(str(query)[:2000]) +
            "Assistant: {}\n".format(str(answer)[:2000]) +
            "Extract: facts[], preferences[], constraints[], "
            "procedural[{\"trigger\",\"action\",\"context\"}], importance(float 0-1). "
            "Only extract explicitly stated information."
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        text = str(response.content).strip()
        return self._parse_llm_response(text, query, answer)

    def _parse_llm_response(self, text, query, answer):
        try:
            m = _re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, _re.DOTALL)
            if m:
                text = m.group(1).strip()
            else:
                s = text.find("{")
                e = text.rfind("}")
                if s >= 0 and e > s:
                    text = text[s:e + 1]
            result = _json.loads(text)
            return {
                "facts": self._clean_list(result.get("facts", [])),
                "preferences": self._clean_list(result.get("preferences", [])),
                "constraints": self._clean_list(result.get("constraints", [])),
                "procedural": self._clean_procedural(result.get("procedural", [])),
                "importance": float(result.get("importance", 0.5)),
            }
        except (_json.JSONDecodeError, ValueError):
            _logger.warning("LLM JSON parse failed, fallback to rules")
            return self._fallback.extract(query, answer)

    @staticmethod
    def _clean_list(items):
        if not isinstance(items, list):
            return []
        return [str(i).strip() for i in items if str(i).strip()]

    @staticmethod
    def _clean_procedural(items):
        if not isinstance(items, list):
            return []
        cleaned = []
        for item in items:
            if isinstance(item, dict):
                cleaned.append({
                    "trigger": str(item.get("trigger", "")),
                    "action": str(item.get("action", "")),
                    "context": str(item.get("context", "")),
                })
        return cleaned


def extract_memory_from_messages(messages, extract_facts=True, extract_preferences=True):
    """Compatibility wrapper for old API."""
    extractor = RuleBasedExtractor()
    combined = " ".join(str(m.content) for m in messages)
    result = extractor.extract(combined)
    return {
        "facts": result["facts"] if extract_facts else [],
        "preferences": result["preferences"] if extract_preferences else [],
    }
