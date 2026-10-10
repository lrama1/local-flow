"""Small voice lookups shared by the board, the Cast library and audio generation (no service imports)."""

from __future__ import annotations

from sqlmodel import Session, col, select

from localflow.errors import bad_request
from localflow.models import Voice


def check_voice(s: Session, voice_id: str | None) -> None:
    """Validates a character's `voice_id` (Section 12.2)."""
    if voice_id is not None and s.get(Voice, voice_id) is None:
        raise bad_request("invalid_voice", f"Unknown voice '{voice_id}'. Pick one from the voice list.")


def voices_by_id(s: Session, ids: set[str]) -> dict[str, Voice]:
    if not ids:
        return {}
    return {v.id: v for v in s.exec(select(Voice).where(col(Voice.id).in_(ids)))}
