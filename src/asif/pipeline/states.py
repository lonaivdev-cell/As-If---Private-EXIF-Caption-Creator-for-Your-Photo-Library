"""Per-image states (§5 of docs/DESIGN.md).

Only part of the machine is used before M7 (batch scheduler, resume). The full
set is defined now so the names never drift between milestones.
"""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    DISCOVERED = "discovered"
    STAGED = "staged"
    OPENED = "opened"
    DESCRIBED = "described"
    WRITTEN = "written"
    VERIFIED = "verified"
    UPLOADED = "uploaded"
    CONFIRMED = "confirmed"
    CLEAN = "clean"

    FAILED_SCHEMA = "failed_schema"
    FAILED_GATE = "failed_gate"
    FAILED_UPLOAD = "failed_upload"
    SKIPPED_CONFLICT = "skipped_conflict"
    QUARANTINE = "quarantine"

    # Not in the design's diagram: items that need no work at all.
    SKIPPED_DONE = "skipped_done"
    """Already captioned by an earlier run; nothing to do."""
    SKIPPED_FORMAT = "skipped_format"
    """A format As If doesn't write yet (HEIC/AVIF, RAW, video)."""


TERMINAL = frozenset(
    {
        State.CLEAN,
        State.FAILED_GATE,
        State.SKIPPED_CONFLICT,
        State.QUARANTINE,
        State.SKIPPED_DONE,
        State.SKIPPED_FORMAT,
    }
)
