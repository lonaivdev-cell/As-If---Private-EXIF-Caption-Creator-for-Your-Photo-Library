"""The model's output contract (§8 of docs/DESIGN.md) and the prompt that asks for it.

Everything a model says is untrusted text. It is validated here, on the
orchestrator side, before it gets anywhere near a file: the JSON must match
the schema, and every string is cleaned of control characters so it can't
smuggle a newline into exiftool's argument stream.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import cache
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROMPT_VERSION = "caption_v1"
CAPTION_MAX = 240
KEYWORD_MAX = 40
KEYWORDS_MIN_COUNT = 3
KEYWORDS_MAX_COUNT = 15

# Bidi overrides and isolates can make text display differently from what it
# is ("Trojan Source"). Nobody needs them in a photo caption.
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩﻿")
_WHITESPACE_RUN = re.compile(r"\s+")


def _is_noncharacter(ch: str) -> bool:
    """U+FDD0..U+FDEF and the last two code points of every plane.

    XML forbids U+FFFE and U+FFFF outright (exiftool writes them as "?"), and
    none of them is ever legitimate text.
    """
    cp = ord(ch)
    return 0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE


class SchemaError(ValueError):
    """The model's answer didn't match the schema. Retried, then quarantined."""


def clean_text(value: str, max_len: int) -> str:
    """Normalise model text to one safe line.

    NFC-normalises; turns control characters, line/paragraph separators,
    bidi controls and Unicode noncharacters into spaces; collapses
    whitespace; trims; cuts to ``max_len`` characters. Zero-width joiners
    stay, because emoji and several scripts (Persian, Hindi) need them.
    """
    text = unicodedata.normalize("NFC", value)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        drop = cat in ("Cc", "Cs", "Zl", "Zp") or ch in _BIDI_CONTROLS or _is_noncharacter(ch)
        if drop:
            out.append(" ")
        else:
            out.append(ch)
    collapsed = _WHITESPACE_RUN.sub(" ", "".join(out)).strip()
    return collapsed[:max_len].rstrip()


class CaptionResult(BaseModel):
    """A validated, cleaned model answer.

    Types are strict (no coercion: ``"3"`` is not a count, ``1`` is not
    ``true``). Strings are cleaned and duplicate keywords merged, so an answer
    can have fewer keywords than the model sent, but never none.
    ``parse_caption_json`` additionally holds the model's raw answer to the
    JSON schema it was constrained to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    caption: str = Field(min_length=1, max_length=CAPTION_MAX)
    keywords: tuple[str, ...] = Field(min_length=1, max_length=KEYWORDS_MAX_COUNT)
    has_text: bool
    people_count: int = Field(ge=0, le=50)
    setting: Literal["indoor", "outdoor", "unclear"]

    @field_validator("caption", mode="before")
    @classmethod
    def _clean_caption(cls, value: Any) -> Any:
        return clean_text(value, CAPTION_MAX) if isinstance(value, str) else value

    @field_validator("keywords", mode="before")
    @classmethod
    def _clean_keywords(cls, value: Any) -> Any:
        if not isinstance(value, list | tuple):
            return value
        seen: set[str] = set()
        kept: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("every keyword must be a string")
            word = clean_text(item, KEYWORD_MAX)
            key = word.casefold()
            if word and key not in seen:
                seen.add(key)
                kept.append(word)
        return tuple(kept[:KEYWORDS_MAX_COUNT])


def _check_wire(data: dict[str, Any]) -> None:
    """The parts of schema.json that cleaning would otherwise paper over.

    The model was constrained to the schema, so an answer that breaks it
    means something is wrong (a worker bug, a server ignoring the grammar).
    Better to retry than to quietly trim it into shape.
    """
    caption = data.get("caption")
    if isinstance(caption, str) and len(caption) > CAPTION_MAX:
        raise SchemaError(f"caption: longer than {CAPTION_MAX} characters")
    keywords = data.get("keywords")
    if isinstance(keywords, list):
        if not KEYWORDS_MIN_COUNT <= len(keywords) <= KEYWORDS_MAX_COUNT:
            raise SchemaError(
                f"keywords: expected {KEYWORDS_MIN_COUNT} to {KEYWORDS_MAX_COUNT}, "
                f"got {len(keywords)}"
            )
        if any(isinstance(k, str) and len(k) > KEYWORD_MAX for k in keywords):
            raise SchemaError(f"keywords: one is longer than {KEYWORD_MAX} characters")


def parse_caption_json(raw: str) -> CaptionResult:
    """Parse and validate the model's raw text. Raises ``SchemaError``."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SchemaError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError("expected a JSON object")
    _check_wire(data)
    try:
        return CaptionResult.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise SchemaError(problems) from exc


@cache
def _schema_text() -> str:
    return resources.files("asif.infer").joinpath("schema.json").read_text(encoding="utf-8")


def caption_schema() -> dict[str, Any]:
    """The JSON schema llama-server constrains generation to (a fresh copy each call)."""
    schema: dict[str, Any] = json.loads(_schema_text())
    return schema


# Names help small models more than bare tags do. Unlisted tags fall back to the tag.
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "no": "Norwegian",
    "nb": "Norwegian Bokmål",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt-BR": "Brazilian Portuguese",
    "pt-PT": "European Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-Hans": "Simplified Chinese",
    "zh-Hant": "Traditional Chinese",
}


def language_name(tag: str) -> str:
    """ "zh-Hant-TW" -> "Traditional Chinese": the longest known prefix wins."""
    parts = tag.split("-")
    for n in range(len(parts), 0, -1):
        name = _LANGUAGE_NAMES.get("-".join(parts[:n]))
        if name:
            return name
    return _LANGUAGE_NAMES.get(parts[0].lower(), tag)


def load_prompt(lang: str, version: str = PROMPT_VERSION) -> str:
    """The system prompt for ``version``, filled in for ``lang``."""
    template = (
        resources.files("asif.infer")
        .joinpath("prompts", f"{version}.txt")
        .read_text(encoding="utf-8")
    )
    return template.replace("{language}", language_name(lang)).replace("{lang}", lang)
