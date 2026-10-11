from __future__ import annotations

from sqlmodel import Session, col, select

from localflow.db import Database
from localflow.errors import bad_request, not_found
from localflow.models import Project, Setting, Voice, utcnow
from localflow.schemas.core import ProjectCreate, ProjectUpdate
from localflow.services.assets import AssetService

PLAYGROUND_KEY = "playground_project_id"


class ProjectService:
    def __init__(self, db: Database, assets: AssetService) -> None:
        self.db = db
        self.assets = assets

    def list_projects(self, include_archived: bool = False) -> list[Project]:
        with self.db.session() as s:
            q = select(Project)
            if not include_archived:
                q = q.where(col(Project.archived).is_(False))
            return list(s.exec(q.order_by(col(Project.updated_at).desc())).all())

    def get(self, project_id: str) -> Project:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                raise not_found("Project", project_id)
            return p

    def create(self, data: ProjectCreate) -> Project:
        p = Project(**data.model_dump())
        with self.db.session() as s:
            self._check_voice(s, p.narrator_voice_id)
            s.add(p)
            s.commit()
            s.refresh(p)
        self.assets.project_dir(p.id).mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _check_voice(s: Session, voice_id: str | None) -> None:
        if voice_id is not None and s.get(Voice, voice_id) is None:
            raise bad_request("invalid_voice", "narrator_voice_id isn't a voice in the library.")

    def update(self, project_id: str, data: ProjectUpdate) -> Project:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                raise not_found("Project", project_id)
            changes = data.model_dump(exclude_unset=True)
            if "narrator_voice_id" in changes:
                self._check_voice(s, changes["narrator_voice_id"])
            for k, v in changes.items():
                setattr(p, k, v)
            p.updated_at = utcnow()
            s.add(p)
            s.commit()
            s.refresh(p)
            return p

    def delete(self, project_id: str) -> None:
        self.get(project_id)
        with self.db.session() as s:
            p = s.get(Project, project_id)
            s.delete(p)
            pg = s.get(Setting, PLAYGROUND_KEY)
            if pg is not None and pg.value_json == project_id:
                s.delete(pg)
            s.commit()
        self.assets.delete_project_files(project_id)

    def ensure_playground(self) -> Project:
        """Playground generations need a project for their assets; create a dedicated one on first use."""
        with self.db.session() as s:
            setting = s.get(Setting, PLAYGROUND_KEY)
            if setting is not None and isinstance(setting.value_json, str):
                p = s.get(Project, setting.value_json)
                if p is not None:
                    return p
        p = self.create(
            ProjectCreate(name="Playground", description="Standalone generations from the Playground")
        )
        with self.db.session() as s:
            row = s.get(Setting, PLAYGROUND_KEY) or Setting(key=PLAYGROUND_KEY)
            row.value_json = p.id
            s.add(row)
            s.commit()
        return p
