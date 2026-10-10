"""Character templates (Section 11.7.1): read-only built-ins from `templates.yaml` + user templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, not_found
from localflow.models import CharacterTemplate, Ingredient, utcnow
from localflow.schemas.board import VoiceHintsIn
from localflow.schemas.characters import (
    TEMPLATE_CATEGORIES,
    TemplateCategory,
    TemplateCreate,
    TemplateOut,
    TemplateUpdate,
)
from localflow.schemas.story import slugify
from localflow.services.board import unique_key

BUILTIN_FILE = Path(__file__).with_name("templates.yaml")
THUMBNAILS_DIR = Path(__file__).with_name("thumbnails")
THUMB_SIZE = 384


class BuiltinTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    name: str
    category: TemplateCategory
    visual_prompt: str = Field(min_length=10)
    negative_prompt: str = ""
    character_info: str = ""
    body_outfit_prompt: str = ""
    voice_hints: VoiceHintsIn = Field(default_factory=VoiceHintsIn)


@lru_cache(maxsize=1)
def builtin_templates() -> tuple[BuiltinTemplate, ...]:
    raw = yaml.safe_load(BUILTIN_FILE.read_text(encoding="utf-8")) or []
    items = tuple(BuiltinTemplate.model_validate(x) for x in raw)
    keys = [t.key for t in items]
    if len(set(keys)) != len(keys):
        raise ValueError("templates.yaml has duplicate keys")
    return items


def square_thumb(src: Path, dst: Path, size: int = THUMB_SIZE) -> None:
    """A square JPEG crop biased toward the top of the image, where faces usually are."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        thumb = ImageOps.fit(im.convert("RGB"), (size, size), Image.Resampling.LANCZOS, centering=(0.5, 0.3))
    thumb.save(dst, "JPEG", quality=86)


class TemplateService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    @property
    def thumbs_dir(self) -> Path:
        return self.settings.library_dir / "templates"

    # --- reads -----------------------------------------------------------------------------------
    def list(self) -> list[TemplateOut]:
        order = {c: n for n, c in enumerate(TEMPLATE_CATEGORIES)}
        builtins = sorted(builtin_templates(), key=lambda t: order.get(t.category, 99))
        with self.db.session() as s:
            users = sorted(s.exec(select(CharacterTemplate)).all(), key=lambda t: t.name.lower())
        return [self._builtin_out(t) for t in builtins] + [self._user_out(u) for u in users]

    def find(self, key: str) -> TemplateOut:
        for t in builtin_templates():
            if t.key == key:
                return self._builtin_out(t)
        with self.db.session() as s:
            row = s.exec(select(CharacterTemplate).where(CharacterTemplate.key == key)).first()
        if row is None:
            raise not_found("Template", key)
        return self._user_out(row)

    def thumbnail(self, key: str) -> Path:
        for t in builtin_templates():
            if t.key == key:
                path = THUMBNAILS_DIR / f"{key}.jpg"
                break
        else:
            with self.db.session() as s:
                row = s.exec(select(CharacterTemplate).where(CharacterTemplate.key == key)).first()
            if row is None or not row.thumbnail_path:
                raise not_found("Template thumbnail", key)
            path = self.settings.data_dir / row.thumbnail_path
        if not path.is_file():
            raise not_found("Template thumbnail", key)
        return path

    def _builtin_out(self, t: BuiltinTemplate) -> TemplateOut:
        has_thumb = (THUMBNAILS_DIR / f"{t.key}.jpg").is_file()
        return TemplateOut(
            id=None,
            key=t.key,
            name=t.name,
            category=t.category,
            builtin=True,
            visual_prompt=t.visual_prompt,
            negative_prompt=t.negative_prompt,
            character_info=t.character_info,
            body_outfit_prompt=t.body_outfit_prompt,
            voice_hints=t.voice_hints.model_dump(),
            thumbnail_url=f"/api/character-templates/{t.key}/thumbnail" if has_thumb else None,
        )

    def _user_out(self, row: CharacterTemplate) -> TemplateOut:
        version = int(row.updated_at.timestamp())
        return TemplateOut(
            id=row.id,
            key=row.key,
            name=row.name,
            category=row.category,
            builtin=False,
            visual_prompt=row.visual_prompt,
            negative_prompt=row.negative_prompt,
            character_info=row.character_info,
            body_outfit_prompt=row.body_outfit_prompt,
            voice_hints=dict(row.voice_hints_json),
            thumbnail_url=(
                f"/api/character-templates/{row.key}/thumbnail?v={version}" if row.thumbnail_path else None
            ),
        )

    # --- user templates --------------------------------------------------------------------------
    def create(self, body: TemplateCreate, source: Ingredient | None, primary: Path | None) -> TemplateOut:
        """ "Save as template" (with `source`) or a template written from scratch."""
        name = body.name or (source.name if source else "")
        with self.db.session() as s:
            taken = {t.key for t in builtin_templates()} | {
                t.key for t in s.exec(select(CharacterTemplate)).all()
            }
            row = CharacterTemplate(
                key=unique_key(slugify(name, "template"), taken),
                name=name,
                category=body.category,
                visual_prompt=_pick(body.visual_prompt, source.visual_prompt if source else ""),
                negative_prompt=_pick(body.negative_prompt, source.negative_prompt if source else ""),
                character_info=_pick(body.character_info, source.character_info if source else ""),
                body_outfit_prompt=_pick(
                    body.body_outfit_prompt, source.body_outfit_prompt if source else ""
                ),
                voice_hints_json=(
                    body.voice_hints.model_dump()
                    if body.voice_hints
                    else dict(source.voice_hints_json if source else {})
                ),
            )
            if primary is not None and primary.is_file():
                rel = f"library/templates/{row.id}.jpg"
                square_thumb(primary, self.settings.data_dir / rel)
                row.thumbnail_path = rel
            s.add(row)
            s.commit()
            s.refresh(row)
        return self._user_out(row)

    def _user_row(self, s: Session, template_id: str) -> CharacterTemplate:
        if any(t.key == template_id for t in builtin_templates()):
            raise AppError(
                409, "builtin_template", "Built-in templates are read-only. Save a copy as your own template."
            )
        row = s.get(CharacterTemplate, template_id)
        if row is None:
            raise not_found("Template", template_id)
        return row

    def update(self, template_id: str, body: TemplateUpdate) -> TemplateOut:
        with self.db.session() as s:
            row = self._user_row(s, template_id)
            for k, v in body.model_dump(exclude_unset=True).items():
                if k == "voice_hints":
                    row.voice_hints_json = dict(v or {})
                elif v is not None:
                    setattr(row, k, v)
            row.updated_at = utcnow()
            s.add(row)
            s.commit()
            s.refresh(row)
        return self._user_out(row)

    def delete(self, template_id: str) -> None:
        with self.db.session() as s:
            row = self._user_row(s, template_id)
            if row.thumbnail_path:
                (self.settings.data_dir / row.thumbnail_path).unlink(missing_ok=True)
            s.delete(row)
            s.commit()


def _pick(explicit: str | None, fallback: str | None) -> str:
    return explicit if explicit is not None else (fallback or "")
