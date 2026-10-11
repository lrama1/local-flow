"""Collections: user-defined groups of a project's assets (Section 13). Deleting a collection keeps its
assets; deleting an asset drops it from every collection (FK cascade)."""

from __future__ import annotations

from sqlalchemy import delete
from sqlmodel import Session, col, func, select

from localflow.db import Database
from localflow.errors import bad_request, conflict, not_found
from localflow.models import Asset, Collection, CollectionAsset, Project, utcnow
from localflow.schemas.collections import CollectionCreate, CollectionOut, CollectionUpdate
from localflow.services.events import EventBus


class CollectionService:
    def __init__(self, db: Database, bus: EventBus) -> None:
        self.db = db
        self.bus = bus

    def _get(self, s: Session, collection_id: str) -> Collection:
        c = s.get(Collection, collection_id)
        if c is None:
            raise not_found("Collection", collection_id)
        return c

    def _out(self, s: Session, c: Collection) -> CollectionOut:
        n = s.exec(
            select(func.count()).select_from(CollectionAsset).where(CollectionAsset.collection_id == c.id)
        ).one()
        cover = s.exec(
            select(Asset)
            .join(CollectionAsset, col(CollectionAsset.asset_id) == col(Asset.id))
            .where(CollectionAsset.collection_id == c.id, col(Asset.thumb_path).is_not(None))
            .order_by(col(CollectionAsset.id).desc())
            .limit(1)
        ).first()
        return CollectionOut(
            id=c.id,
            project_id=c.project_id,
            name=c.name,
            description=c.description,
            asset_count=int(n),
            cover_thumb_url=f"/api/thumbs/{c.project_id}/{cover.id}" if cover else None,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )

    def _changed(self, project_id: str) -> None:
        self.bus.publish("collections.updated", project_id=project_id)

    def _check_name(self, s: Session, project_id: str, name: str, exclude: str | None = None) -> str:
        name = name.strip()
        clash = s.exec(
            select(Collection).where(Collection.project_id == project_id, Collection.name == name)
        ).first()
        if clash is not None and clash.id != exclude:
            raise conflict("collection_exists", f"There's already a collection named '{name}'.")
        return name

    def for_project(self, project_id: str) -> list[CollectionOut]:
        with self.db.session() as s:
            if s.get(Project, project_id) is None:
                raise not_found("Project", project_id)
            rows = s.exec(
                select(Collection).where(Collection.project_id == project_id).order_by(col(Collection.name))
            ).all()
            return [self._out(s, c) for c in rows]

    def create(self, project_id: str, body: CollectionCreate) -> CollectionOut:
        with self.db.session() as s:
            if s.get(Project, project_id) is None:
                raise not_found("Project", project_id)
            c = Collection(
                project_id=project_id,
                name=self._check_name(s, project_id, body.name),
                description=body.description.strip(),
            )
            s.add(c)
            s.flush()
            self._add(s, c, body.asset_ids)
            s.commit()
            s.refresh(c)
            out = self._out(s, c)
        self._changed(project_id)
        return out

    def update(self, collection_id: str, body: CollectionUpdate) -> CollectionOut:
        with self.db.session() as s:
            c = self._get(s, collection_id)
            if body.name is not None:
                c.name = self._check_name(s, c.project_id, body.name, exclude=c.id)
            if body.description is not None:
                c.description = body.description.strip()
            c.updated_at = utcnow()
            s.add(c)
            s.commit()
            s.refresh(c)
            out = self._out(s, c)
        self._changed(out.project_id)
        return out

    def delete(self, collection_id: str) -> None:
        with self.db.session() as s:
            c = self._get(s, collection_id)
            project_id = c.project_id
            s.exec(delete(CollectionAsset).where(col(CollectionAsset.collection_id) == c.id))
            s.delete(c)
            s.commit()
        self._changed(project_id)

    def _add(self, s: Session, c: Collection, asset_ids: list[str]) -> None:
        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return
        assets = {a.id: a for a in s.exec(select(Asset).where(col(Asset.id).in_(ids)))}
        bad = [i for i in ids if i not in assets or assets[i].project_id != c.project_id]
        if bad:
            raise bad_request("invalid_asset", f"Not an asset of this project: {', '.join(bad[:3])}")
        have = set(
            s.exec(
                select(CollectionAsset.asset_id).where(
                    CollectionAsset.collection_id == c.id, col(CollectionAsset.asset_id).in_(ids)
                )
            )
        )
        for i in ids:
            if i not in have:
                s.add(CollectionAsset(collection_id=c.id, asset_id=i))

    def add_assets(self, collection_id: str, asset_ids: list[str]) -> CollectionOut:
        with self.db.session() as s:
            c = self._get(s, collection_id)
            self._add(s, c, asset_ids)
            c.updated_at = utcnow()
            s.add(c)
            s.commit()
            s.refresh(c)
            out = self._out(s, c)
        self._changed(out.project_id)
        return out

    def remove_assets(self, collection_id: str, asset_ids: list[str]) -> CollectionOut:
        with self.db.session() as s:
            c = self._get(s, collection_id)
            for m in s.exec(
                select(CollectionAsset).where(
                    CollectionAsset.collection_id == c.id, col(CollectionAsset.asset_id).in_(asset_ids)
                )
            ):
                s.delete(m)
            c.updated_at = utcnow()
            s.add(c)
            s.commit()
            s.refresh(c)
            out = self._out(s, c)
        self._changed(out.project_id)
        return out

    def of_asset(self, asset_id: str) -> list[str]:
        """Ids of the collections an asset is in."""
        with self.db.session() as s:
            return list(
                s.exec(select(CollectionAsset.collection_id).where(CollectionAsset.asset_id == asset_id))
            )
