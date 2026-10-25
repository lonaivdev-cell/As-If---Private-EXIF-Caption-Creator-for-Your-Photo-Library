"""The model's output contract: schema, validation, cleaning, and the prompt."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from asif.infer.schema import (
    CAPTION_MAX,
    KEYWORD_MAX,
    KEYWORDS_MAX_COUNT,
    PROMPT_VERSION,
    CaptionResult,
    SchemaError,
    caption_schema,
    clean_text,
    language_name,
    load_prompt,
    parse_caption_json,
)

GOOD: dict[str, Any] = {
    "caption": "A brown dog runs along a sandy beach at sunset.",
    "keywords": ["dog", "beach", "sunset"],
    "has_text": False,
    "people_count": 0,
    "setting": "outdoor",
}


def parse(**changes: Any) -> CaptionResult:
    return parse_caption_json(json.dumps({**GOOD, **changes}))


def rejects(**changes: Any) -> str:
    with pytest.raises(SchemaError) as info:
        parse(**changes)
    return str(info.value)


# -- the JSON schema llama-server is constrained to -------------------------------------


def test_schema_matches_the_design() -> None:
    schema = caption_schema()
    props = schema["properties"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(props) == set(GOOD)
    assert props["caption"] == {"type": "string", "minLength": 1, "maxLength": CAPTION_MAX}
    assert props["keywords"]["minItems"] == 3
    assert props["keywords"]["maxItems"] == KEYWORDS_MAX_COUNT
    assert props["keywords"]["items"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": KEYWORD_MAX,
    }
    assert props["has_text"] == {"type": "boolean"}
    assert props["people_count"] == {"type": "integer", "minimum": 0, "maximum": 50}
    assert props["setting"]["enum"] == ["indoor", "outdoor", "unclear"]


def test_the_biggest_answer_the_schema_allows_is_accepted() -> None:
    result = parse(
        caption="c" * CAPTION_MAX,
        keywords=[f"{i:02d}" + "k" * (KEYWORD_MAX - 2) for i in range(KEYWORDS_MAX_COUNT)],
        people_count=50,
        has_text=True,
        setting="unclear",
    )
    assert len(result.caption) == CAPTION_MAX
    assert len(result.keywords) == KEYWORDS_MAX_COUNT
    assert all(len(k) == KEYWORD_MAX for k in result.keywords)


# -- parsing and validation ---------------------------------------------------------------


def test_a_good_answer_parses() -> None:
    result = parse()
    assert result.caption == GOOD["caption"]
    assert result.keywords == ("dog", "beach", "sunset")
    assert (result.has_text, result.people_count, result.setting) == (False, 0, "outdoor")


def test_results_are_immutable() -> None:
    with pytest.raises(ValidationError):
        parse().caption = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "{'single': 'quotes'}",
        '{"caption": "unterminated',
        "[1, 2, 3]",
        "42",
        "null",
    ],
)
def test_non_objects_are_schema_errors(raw: str) -> None:
    with pytest.raises(SchemaError):
        parse_caption_json(raw)


def test_a_non_string_raw_is_a_schema_error() -> None:
    with pytest.raises(SchemaError):
        parse_caption_json(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", sorted(GOOD))
def test_every_field_is_required(field: str) -> None:
    data = {k: v for k, v in GOOD.items() if k != field}
    with pytest.raises(SchemaError, match=field):
        parse_caption_json(json.dumps(data))


def test_extra_fields_are_rejected() -> None:
    assert "confidence" in rejects(confidence=0.99)


@pytest.mark.parametrize(
    "changes",
    [
        {"caption": ""},
        {"caption": "   "},
        {"caption": "\n\t\r\x00"},
        {"caption": "\u202e\u2028"},
        {"caption": 123},
        {"caption": None},
        {"keywords": []},
        {"keywords": "dog, beach"},
        {"keywords": ["dog", 3, "beach"]},
        {"keywords": ["", "  ", "\n"]},
        {"has_text": "maybe"},
        {"has_text": None},
        {"people_count": -1},
        {"people_count": 51},
        {"people_count": 2.5},
        {"people_count": True},
        {"people_count": False},
        {"setting": "beach"},
        {"setting": "Outdoor"},
        {"setting": None},
    ],
)
def test_invalid_values_are_rejected(changes: dict[str, Any]) -> None:
    rejects(**changes)


def test_people_count_bounds_are_inclusive() -> None:
    assert parse(people_count=0).people_count == 0
    assert parse(people_count=50).people_count == 50


def test_error_messages_never_contain_the_models_text() -> None:
    # Errors end up in logs and terminals; captions must not.
    message = rejects(caption="Grandma's secret garden", keywords="secret-keyword", setting=7)
    assert "secret" not in message.lower()
    with pytest.raises(SchemaError) as info:
        parse_caption_json('{"caption": "secret garden", "keywords": [')
    assert "secret" not in str(info.value)


# -- cleaning -----------------------------------------------------------------------------


def test_captions_become_one_safe_line() -> None:
    # A newline could smuggle an extra option into exiftool's argument stream.
    result = parse(caption="A dog.\n-XMP:all=\r\n\tOn a beach.")
    assert result.caption == "A dog. -XMP:all= On a beach."
    assert "\n" not in result.caption and "\r" not in result.caption


def test_text_that_only_cleaning_makes_too_long_is_cut_to_the_limit() -> None:
    # 240 raw characters pass the wire check; NFC and whitespace cleaning can
    # only shorten them, so the result is within the limits and trimmed.
    result = parse(caption=" " + "word " * 47 + "end", keywords=["k" * 40, "b", "c"])
    assert len(result.caption) <= CAPTION_MAX
    assert all(len(k) <= KEYWORD_MAX for k in result.keywords)
    assert result.caption == result.caption.strip()
    assert result.caption.endswith("end")


def test_constructing_directly_cuts_overlong_text() -> None:
    # Callers that build a CaptionResult themselves (not from the model) get
    # cleaned, cut text rather than an error.
    result = CaptionResult(
        caption="word " * 100,
        keywords=("k" * 100, "b", "c"),
        has_text=False,
        people_count=0,
        setting="unclear",
    )
    assert len(result.caption) <= CAPTION_MAX
    assert all(len(k) <= KEYWORD_MAX for k in result.keywords)


def test_keywords_are_cleaned_and_deduplicated_keeping_the_first_spelling() -> None:
    result = parse(
        keywords=["Dog", " dog ", "DOG", "beach\nparty", "", "sun\u202eset", "beach party"]
    )
    assert result.keywords == ("Dog", "beach party", "sun set")


@pytest.mark.parametrize("count", [0, 1, 2, 16, 40])
def test_keyword_count_outside_the_schema_is_rejected(count: int) -> None:
    # The model was constrained to 3-15; anything else means something is off.
    assert "keywords" in rejects(keywords=[f"kw{i}" for i in range(count)])


def test_overlong_raw_text_is_rejected_not_trimmed() -> None:
    assert "caption" in rejects(caption="a" * 241)
    assert "keywords" in rejects(keywords=["ok", "fine", "k" * 41])


def test_duplicates_may_leave_fewer_than_three_keywords() -> None:
    result = parse(keywords=["Dog", "dog", "DOG "])
    assert result.keywords == ("Dog",)


@pytest.mark.parametrize(
    "changes",
    [
        {"people_count": "3"},
        {"people_count": 3.0},
        {"people_count": True},
        {"has_text": 1},
        {"has_text": "true"},
        {"caption": 42},
        {"keywords": "dog, cat, sky"},
        {"keywords": ["dog", "cat", 7]},
    ],
)
def test_types_are_never_coerced(changes: dict[str, Any]) -> None:
    rejects(**changes)


def test_caption_schema_is_a_fresh_copy_each_time() -> None:
    from asif.infer.schema import caption_schema

    first = caption_schema()
    first["properties"].clear()
    assert caption_schema()["properties"], "mutating one copy must not change the next"


def test_noncharacters_are_removed() -> None:
    from asif.infer.schema import clean_text

    assert clean_text("a\ufffeb\uffffc\ufdd0d", 40) == "a b c d"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\nb\tc\rd\x00e\x7fz", "a b c d e z"),
        ("line\u2028para\u2029end", "line para end"),
        ("left\u202eright\u2066x\u2069\ufeff", "left right x"),
        ("  lots   of\n\n space  ", "lots of space"),
        ("cafe\u0301", "caf\u00e9"),  # NFC
        ("family: \U0001f469\u200d\U0001f467", "family: \U0001f469\u200d\U0001f467"),  # ZWJ kept
        (
            "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645",
            "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645",
        ),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw, 100) == expected


def test_clean_text_cuts_then_trims() -> None:
    assert clean_text("abc def", 4) == "abc"
    assert clean_text("abcdef", 3) == "abc"


def test_lone_surrogates_from_json_escapes_are_removed() -> None:
    raw = json.dumps({**GOOD, "caption": "PLACEHOLDER"}).replace("PLACEHOLDER", "a\\ud800b")
    assert parse_caption_json(raw).caption == "a b"


def test_non_ascii_text_survives() -> None:
    result = parse(
        caption="Cachorro na praia ao pôr do sol", keywords=["cachorro", "praia", "pôr do sol"]
    )
    assert result.caption == "Cachorro na praia ao pôr do sol"
    assert "pôr do sol" in result.keywords


# -- the prompt ---------------------------------------------------------------------------


def test_prompt_is_filled_in_for_the_language() -> None:
    prompt = load_prompt("pt-BR")
    assert "Brazilian Portuguese" in prompt
    assert "(language tag: pt-BR)" in prompt
    assert "{" not in prompt and "}" not in prompt
    assert "JSON" in prompt
    assert "identity" in prompt  # never guess who people are


def test_prompt_version_has_a_file() -> None:
    assert load_prompt("en", PROMPT_VERSION) == load_prompt("en")
    with pytest.raises(FileNotFoundError):
        load_prompt("en", "caption_v0_does_not_exist")


@pytest.mark.parametrize(
    ("tag", "name"),
    [
        ("en", "English"),
        ("pt-BR", "Brazilian Portuguese"),
        ("pt-PT", "European Portuguese"),
        ("pt-AO", "Portuguese"),
        ("de-CH", "German"),
        ("zh-Hant", "Traditional Chinese"),
        ("zh-Hant-TW", "Traditional Chinese"),
        ("zh-Hans-CN", "Simplified Chinese"),
        ("PT-br", "Portuguese"),
        ("xx", "xx"),
        ("tlh-KL", "tlh-KL"),
    ],
)
def test_language_names(tag: str, name: str) -> None:
    assert language_name(tag) == name
