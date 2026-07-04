"""
Response normalizer: unify thinking/reasoning content across models.

Some models (DeepSeek, Kimi, Qwen3) return reasoning in a separate
``reasoning_content`` field. Others inline it inside ``content`` with
tags like `<think>...</think>` or ``【思考】...【回答】``.

This module normalizes both non-streaming and streaming responses to a
consistent format: reasoning in ``reasoning_content``, clean answer in
``content``.
"""

import re
import json
from enum import Enum
from typing import Optional


class ThinkingMode(str, Enum):
    passthrough = "passthrough"  # return raw model output as-is
    normalize = "normalize"      # extract inline thinking into reasoning_content
    strip = "strip"              # remove all reasoning, only keep content


# Patterns that capture inline thinking blocks in content.
INLINE_PATTERNS = [
    # DeepSeek / Qwen / GLM style:  response ...  response
    re.compile(
        r"<think[^>]*>(.*?)</think>",
        re.DOTALL,
    ),
    # Chinese bracket style: 【思考】...【回答】 or 【思考过程】...【回答】
    re.compile(
        r"【思考[^】]*】(.*?)【回答】",
        re.DOTALL,
    ),
    # Alternative:  ...  (without closing, ends at next tag)
    re.compile(
        r"<thinking>\s*(.*?)\s*</thinking>",
        re.DOTALL,
    ),
]


def _extract_inline_thinking(content: str) -> tuple[Optional[str], str]:
    """Extract inline thinking tags from content.

    Returns (reasoning, clean_content). reasoning is None if no
    thinking content was found.
    """
    if not content:
        return None, content

    all_reasoning: list[str] = []
    cleaned = content

    for pattern in INLINE_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            all_reasoning.extend(m.strip() for m in matches)
            cleaned = pattern.sub("", cleaned)

    cleaned = cleaned.strip()

    if not all_reasoning:
        return None, content.strip()

    return "\n\n".join(all_reasoning), cleaned


def _has_reasoning_field(choice: dict) -> bool:
    """Check if the choice already has a reasoning_content field."""
    msg = choice.get("message") or choice.get("delta") or {}
    return "reasoning_content" in msg


def normalize_response(response: dict, mode: ThinkingMode) -> dict:
    """Normalize a non-streaming chat completion response."""
    if mode == ThinkingMode.passthrough:
        return response

    choices = response.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""

        if _has_reasoning_field(choice):
            if mode == ThinkingMode.strip:
                msg.pop("reasoning_content", None)
            continue

        reasoning, clean_content = _extract_inline_thinking(content)
        if reasoning is not None:
            if mode == ThinkingMode.strip:
                msg["content"] = clean_content
            else:
                msg["reasoning_content"] = reasoning
                msg["content"] = clean_content

    return response


# ── streaming parser ──────────────────────────────────────────────────


class StreamNormalizer:
    """Stateful SSE parser that normalizes thinking content on the fly.

    Accumulates content across chunks, detects inline thinking tags,
    and re-emits as separate reasoning_content / content delta events.
    """

    def __init__(self, mode: ThinkingMode):
        self.mode = mode
        self._buffer: list[str] = []
        self._in_thinking = False
        self._seen_reasoning = False  # model already uses reasoning_content
        self._finished = False

    def feed(self, line: str) -> list[str]:
        """Process one SSE line, return a list of SSE lines to emit.

        Each returned string is a line (without trailing \\n). An empty
        string represents a blank line separator between SSE events.
        """
        if self.mode == ThinkingMode.passthrough:
            return [line] if line is not None else [""]

        if not line:
            return [""]

        if line.startswith(":"):
            return [line]

        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                self._finished = True
                flushed = self._flush_buffer()
                if flushed:
                    return flushed + [line]
                return [line]

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                return [line]

            return self._process_chunk(data)

        return [line]

    def _process_chunk(self, data: dict) -> list[str]:
        choices = data.get("choices", [])
        if not choices:
            return self._make_sse(data)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "")
        reasoning = delta.get("reasoning_content", "")

        if reasoning:
            self._seen_reasoning = True
            if self.mode == ThinkingMode.strip:
                delta.pop("reasoning_content", None)
                return self._make_sse(data) if delta else [""]
            return self._make_sse(data)

        if content is None:
            return self._make_sse(data)

        if self._seen_reasoning:
            return self._make_sse(data)

        self._buffer.append(content)
        full = "".join(self._buffer)

        result = self._try_split(full)
        if result is None:
            return [""]

        reasoning_part, content_part = result
        events = []

        if reasoning_part:
            r_data = self._clone_with_delta(data, reasoning_content=reasoning_part)
            events.extend(self._make_sse(r_data))

        if content_part:
            c_data = self._clone_with_delta(data, content=content_part)
            events.extend(self._make_sse(c_data))

        self._buffer = []
        return events

    def _try_split(self, full: str) -> Optional[tuple[Optional[str], Optional[str]]]:
        """Try to split full text into (reasoning, content)."""
        if not full:
            return None

        for pattern in INLINE_PATTERNS:
            m = pattern.search(full)
            if m:
                reasoning = m.group(1).strip()
                after = full[m.end():].strip()
                return reasoning, after

        open_tag = re.search(r"<think[^>]*>", full)
        if open_tag and not re.search(r"</think>", full):
            return None

        open_tag2 = re.search(r"【思考[^】]*】", full)
        if open_tag2 and not re.search(r"【回答】", full):
            return None

        return None, full

    def _flush_buffer(self) -> list[str]:
        if self._buffer and not self._seen_reasoning:
            full = "".join(self._buffer)
            reasoning, clean = _extract_inline_thinking(full)
            if reasoning and self.mode != ThinkingMode.strip:
                r_line = f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': reasoning}, 'index': 0}]}, ensure_ascii=False)}"
                c_line = f"data: {json.dumps({'choices': [{'delta': {'content': clean}, 'index': 0}]}, ensure_ascii=False)}"
                self._buffer = []
                return [r_line, "", c_line]
            elif self.mode == ThinkingMode.strip and reasoning:
                c_line = f"data: {json.dumps({'choices': [{'delta': {'content': clean}, 'index': 0}]}, ensure_ascii=False)}"
                self._buffer = []
                return [c_line]
        return []

    def _make_sse(self, data: dict) -> list[str]:
        line = f"data: {json.dumps(data, ensure_ascii=False)}"
        return [line, ""]

    def _clone_with_delta(self, data: dict, **delta_kw) -> dict:
        result = {
            **data,
            "choices": [{
                **data["choices"][0],
                "delta": {**data["choices"][0].get("delta", {}), **delta_kw},
            }],
        }
        if "reasoning_content" in delta_kw:
            result["choices"][0]["delta"].pop("content", None)
        if "content" in delta_kw:
            result["choices"][0]["delta"].pop("reasoning_content", None)
        return result