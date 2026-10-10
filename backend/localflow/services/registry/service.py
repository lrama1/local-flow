"""Model registry (Section 9): catalog entries, HF/Civitai installs, access checks, deletes, license acks."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from sqlmodel import col, select

from localflow.config import Settings
from localflow.db import Database
from localflow.errors import AppError, bad_request, conflict, not_found
from localflow.models import Job, ModelEntry, utcnow
from localflow.schemas.models import (
    AdapterOption,
    CivitaiFileOut,
    CivitaiInstallBody,
    CivitaiModelOut,
    CivitaiRef,
    CivitaiSearchOut,
    CivitaiVersionOut,
    DeleteModelOut,
    HfFileOut,
    HfInspectOut,
    HfInstallBody,
    HubAccount,
    ModelEntryOut,
    ModelFileOut,
    ModelOptions,
    ModelPartOut,
    SuggestionOut,
)
from localflow.services.events import EventBus
from localflow.services.jobs.service import TERMINAL, JobService
from localflow.services.keys import KeyService
from localflow.services.registry import policy
from localflow.services.registry.catalog import load_catalog, sync_catalog
from localflow.services.registry.civitai import CivitaiClient
from localflow.services.registry.hf import (
    FetchRequest,
    HfBackend,
    HubAccessError,
    HubClient,
    RepoFile,
    RepoInfo,
    model_url,
    worst_access,
)
from localflow.services.registry.infer import (
    CIVITAI_KINDS,
    KNOWN_ADAPTERS,
    capabilities_for,
    suggest_civitai,
    suggest_hf,
)
from localflow.services.secrets import SecretStore
from localflow.worker.spec import LoraSpec, ModelSpec, PartSpec

log = logging.getLogger("localflow.registry")

Report = Callable[[int, int], None]
_ACTIVE = ("installed", "downloading")
_COMPONENT_KINDS = ("vae", "text_encoder")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ ()\-]+")
_TAG = re.compile(r"<[^>]+>")


def normalize_repo_id(value: str) -> str:
    v = value.strip()
    v = re.sub(r"^https?://(www\.)?huggingface\.co/", "", v)
    parts = [p for p in v.split("/") if p]
    if len(parts) < 2:
        raise bad_request("invalid_repo_id", f"'{value}' isn't a Hugging Face repo id (org/name).")
    return f"{parts[0]}/{parts[1]}"


def civitai_url(model_id: int, version_id: int | None = None) -> str:
    base = f"https://civitai.com/models/{model_id}"
    return f"{base}?modelVersionId={version_id}" if version_id else base


def _strip_html(text: str | None, limit: int = 2000) -> str | None:
    if not text:
        return None
    out = html.unescape(_TAG.sub(" ", text))
    out = re.sub(r"\s+", " ", out).strip()
    return out[:limit] + ("…" if len(out) > limit else "")


class RegistryService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        bus: EventBus,
        jobs: JobService,
        secrets: SecretStore,
        keys: KeyService,
        *,
        hf: HfBackend | None = None,
        civitai: CivitaiClient | None = None,
        adapters: Callable[[], list[dict[str, Any]] | None] = lambda: None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.jobs = jobs
        self.secrets = secrets
        self.keys = keys
        self.hf: HfBackend = hf or HubClient()
        self.civitai = civitai or CivitaiClient()
        self._adapters = adapters
        self._catalog_order = {e.key: i for i, e in enumerate(load_catalog())}

    # --- paths / tokens --------------------------------------------------------------------------
    @property
    def hf_repos_dir(self) -> Path:
        return self.settings.resolved_hf_home / "repos"

    def hf_local_dir(self, repo_id: str) -> Path:
        return self.hf_repos_dir / repo_id.replace("/", "--")

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.settings.data_dir.resolve()).as_posix()
        except ValueError:
            return str(p)

    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.settings.data_dir / p

    def _token(self, name: str) -> str | None:
        r = self.secrets.resolve(name)
        return r.value if r else None

    def _account(self, name: str) -> HubAccount:
        st = self.keys.status(name)
        username = (st.account or {}).get("username") if st.account else None
        return HubAccount(configured=st.configured, username=username, source=st.source)

    # --- queries ---------------------------------------------------------------------------------
    def get(self, entry_id: str) -> ModelEntry:
        with self.db.session() as s:
            e = s.get(ModelEntry, entry_id)
            if e is None:
                raise not_found("Model", entry_id)
            return e

    def list_entries(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        capability: str | None = None,
        catalog: bool | None = None,
    ) -> list[ModelEntry]:
        with self.db.session() as s:
            q = select(ModelEntry)
            if status:
                q = q.where(ModelEntry.status == status)
            if source:
                q = q.where(ModelEntry.source == source)
            if catalog is True:
                q = q.where(col(ModelEntry.catalog_key).is_not(None))
            elif catalog is False:
                q = q.where(col(ModelEntry.catalog_key).is_(None))
            rows = list(s.exec(q).all())
        if capability:
            rows = [r for r in rows if capability in (r.capabilities_json or [])]
        order = self._catalog_order
        return sorted(rows, key=lambda r: (order.get(r.catalog_key or "", 10_000), r.name.lower()))

    def to_out(self, e: ModelEntry) -> ModelEntryOut:
        meta = e.meta_json or {}
        worker = {a.get("adapter_key"): a for a in (self._adapters() or [])}
        known = KNOWN_ADAPTERS.get(e.adapter_key or "")
        parts: list[ModelPartOut] = []
        for p in e.files_json or []:
            repo = p.get("repo_id")
            url: str | None
            if p.get("source") == "civitai":
                files = [ModelFileOut(path=str(p.get("name") or ""), size=p.get("size"))]
                url = civitai_url(int(p.get("model_id") or 0), p.get("version_id"))
            else:
                files = [ModelFileOut(path=f["path"], size=f.get("size")) for f in p.get("files") or []]
                url = model_url(repo) if repo else None
            parts.append(
                ModelPartOut(
                    role=str(p.get("role") or "main"),
                    source=p.get("source") or e.source,
                    repo_id=repo,
                    revision=p.get("revision"),
                    allow_patterns=list(p.get("allow_patterns") or []),
                    files=files,
                    access=p.get("access"),
                    gated=bool(p.get("gated")),
                    model_url=url,
                )
            )
        civ = meta.get("civitai")
        return ModelEntryOut(
            id=e.id,
            catalog_key=e.catalog_key,
            source=e.source,
            source_ref=e.source_ref,
            name=e.name,
            description=e.description,
            purpose=meta.get("purpose"),
            recommended=bool(meta.get("recommended")),
            kind=e.kind,
            capabilities=list(e.capabilities_json or []),
            adapter_key=e.adapter_key,
            adapter_name=known.name if known else e.adapter_key,
            adapter_available=bool((worker.get(e.adapter_key) or {}).get("available")),
            base_model=e.base_model,
            license=e.license,
            license_url=e.license_url,
            license_acknowledged=e.license_acknowledged_at is not None,
            license_acknowledged_at=e.license_acknowledged_at,
            nsfw=e.nsfw,
            parts=parts,
            local_path=e.local_path,
            size_bytes=e.size_bytes,
            approx_size_bytes=meta.get("approx_size_bytes"),
            status=e.status,
            partial=bool(meta.get("partial")) and e.status != "installed",
            error=e.error,
            default_params=dict(e.default_params_json or {}),
            vram_profiles=dict(e.vram_profiles_json or {}),
            gated=e.gated,
            access=e.access,
            access_message=e.access_message,
            access_checked_at=e.access_checked_at,
            trigger_words=list(e.trigger_words_json or []),
            preview_url=e.preview_url,
            requires_pickle=e.requires_pickle,
            pickle_blocked=e.requires_pickle and not self.settings.allow_pickle_weights,
            experimental=e.experimental,
            notes=e.notes,
            model_url=self._model_url(e),
            job_id=meta.get("job_id") if e.status == "downloading" else None,
            civitai=CivitaiRef.model_validate(civ) if isinstance(civ, dict) else None,
            installed_at=e.installed_at,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )

    def _model_url(self, e: ModelEntry) -> str | None:
        if e.source == "hf":
            return model_url(e.source_ref)
        civ = (e.meta_json or {}).get("civitai") or {}
        if e.source == "civitai" and civ.get("model_id"):
            return civitai_url(int(civ["model_id"]), civ.get("version_id"))
        return None

    def options(self) -> ModelOptions:
        worker = {a.get("adapter_key"): a for a in (self._adapters() or [])}
        return ModelOptions(
            adapters=[
                AdapterOption(
                    key=a.key,
                    name=a.name,
                    capabilities=list(a.capabilities),
                    base_models=list(a.base_models),
                    supports_lora=a.supports_lora,
                    milestone=a.milestone,
                    available=bool((worker.get(a.key) or {}).get("available")),
                )
                for a in KNOWN_ADAPTERS.values()
            ],
            kinds=["checkpoint", "lora", "vae", "text_encoder", "audio"],
            allow_pickle_weights=self.settings.allow_pickle_weights,
            allow_nsfw_models=self.settings.allow_nsfw_models,
            models_dir=str(self.settings.models_dir),
            free_bytes=policy.free_bytes(self.settings.models_dir),
            hf=self._account("hf_token"),
            civitai=self._account("civitai_api_key"),
        )

    # --- persistence helpers ---------------------------------------------------------------------
    def _save(self, e: ModelEntry, *, publish: bool = True) -> ModelEntry:
        e.updated_at = utcnow()
        with self.db.session() as s:
            merged = s.merge(e)
            s.commit()
            s.refresh(merged)
            e = merged
        if publish:
            self.bus.publish("model.status", model=self.to_out(e).model_dump(mode="json"))
        return e

    def _update(self, entry_id: str, **fields: Any) -> ModelEntry:
        e = self.get(entry_id)
        for k, v in fields.items():
            setattr(e, k, v)
        return self._save(e)

    def sync_catalog(self) -> int:
        return sync_catalog(self.db)

    def reset_interrupted(self) -> int:
        """At startup: entries stuck `downloading` without a queued job go back to `available`."""
        with self.db.session() as s:
            queued = s.exec(select(Job).where(Job.type == "download_model", Job.status == "queued")).all()
            active = {j.params_json.get("model_entry_id") for j in queued}
            stuck = s.exec(select(ModelEntry).where(ModelEntry.status == "downloading")).all()
            n = 0
            for e in stuck:
                if e.id not in active:
                    e.status, e.error = "available", "Download interrupted. Install again to resume."
                    meta = {k: v for k, v in (e.meta_json or {}).items() if k != "job_id"}
                    e.meta_json = {**meta, "partial": True}
                    s.add(e)
                    n += 1
            s.commit()
        return n

    # --- Hugging Face ----------------------------------------------------------------------------
    async def _repo_info(self, repo_id: str, revision: str | None, token: str | None) -> RepoInfo:
        return await asyncio.to_thread(self.hf.repo_info, repo_id, revision, token)

    def _find(self, source: str, source_ref: str, *, user_only: bool = True) -> ModelEntry | None:
        with self.db.session() as s:
            q = select(ModelEntry).where(ModelEntry.source == source, ModelEntry.source_ref == source_ref)
            if user_only:
                q = q.where(col(ModelEntry.catalog_key).is_(None))
            return s.exec(q).first()

    async def hf_inspect(self, repo_id: str, revision: str | None) -> HfInspectOut:
        repo_id = normalize_repo_id(repo_id)
        token = self._token("hf_token")
        info = await self._repo_info(repo_id, revision, token)
        access = await asyncio.to_thread(self.hf.access, info, token, self.keys.status("hf_token").token_type)
        allow_pickle = self.settings.allow_pickle_weights
        paths = [f.path for f in info.files]
        selected = set(policy.default_selection(paths, allow_pickle))
        files = [
            HfFileOut(
                path=f.path,
                size=f.size,
                file_class=policy.classify(f.path),
                blocked_reason=policy.blocked_reason(f.path, allow_pickle),
                default_selected=f.path in selected,
            )
            for f in sorted(info.files, key=lambda f: f.path)
        ]
        sug = suggest_hf(
            repo_id,
            {
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "base_models": info.base_models,
                "files": paths,
            },
        )
        existing = self._find("hf", repo_id)
        account = self._account("hf_token").username
        message = self._access_message(repo_id, access.state, access.message, account)
        return HfInspectOut(
            repo_id=repo_id,
            revision=revision,
            sha=info.sha,
            gated=bool(info.gated),
            gated_mode=info.gated,
            private=info.private,
            access=access.state,
            access_message=message,
            model_url=model_url(repo_id),
            license=info.license_name or info.license,
            license_url=info.license_url,
            pipeline_tag=info.pipeline_tag,
            library_name=info.library_name,
            tags=info.tags[:40],
            files=files,
            total_size=sum(f.size or 0 for f in info.files),
            default_selection=sorted(selected),
            selected_size=sum(f.size or 0 for f in info.files if f.path in selected),
            suggestion=SuggestionOut(
                kind=sug.kind,
                base_model=sug.base_model,
                adapter_key=sug.adapter_key,
                capabilities=sug.capabilities,
                supported=sug.supported,
                reason=sug.reason,
            ),
            existing_entry_id=existing.id if existing else None,
            account=account,
        )

    @staticmethod
    def _access_message(repo_id: str, state: str, message: str | None, account: str | None) -> str | None:
        if state == "license_not_accepted" and account:
            return (
                f"{repo_id} is gated. You're signed in as {account}, but haven't accepted its license. "
                "Open the model page, accept, then click Check again."
            )
        return message

    async def hf_install(self, body: HfInstallBody) -> tuple[ModelEntry, Job]:
        repo_id = normalize_repo_id(body.repo_id)
        self._validate_adapter(body.adapter_key)
        token = self._token("hf_token")
        info = await self._repo_info(repo_id, body.revision, token)
        allow_pickle = self.settings.allow_pickle_weights
        paths = [f.path for f in info.files]
        patterns = body.allow_patterns or policy.default_selection(paths, allow_pickle)
        chosen = [
            p for p in policy.match_patterns(paths, patterns) if PurePosixPath(p).name != ".gitattributes"
        ]
        policy.validate_selection(chosen, allow_pickle)
        if not chosen:
            raise bad_request("no_files", "No files in the repository match the selection.")
        sug = suggest_hf(
            repo_id,
            {
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "base_models": info.base_models,
                "files": paths,
            },
        )
        e = self._find("hf", repo_id) or ModelEntry(
            source="hf", source_ref=repo_id, name=repo_id.split("/")[1]
        )
        if e.status in _ACTIVE:
            raise conflict(
                "already_installed" if e.status == "installed" else "already_downloading",
                f"{repo_id} is already {e.status}. Delete it first to change the selected files.",
                {"model_entry_id": e.id},
            )
        base = body.base_model or sug.base_model
        e.name = body.name or e.name
        e.kind = body.kind or sug.kind
        e.base_model = base
        e.adapter_key = body.adapter_key
        if e.kind in _COMPONENT_KINDS:
            e.capabilities_json = []  # components (VAE, text encoder) don't run jobs on their own
        else:
            e.capabilities_json = (
                body.capabilities or capabilities_for(body.adapter_key, base) or sug.capabilities
            )
        e.license = info.license_name or info.license
        e.license_url = info.license_url
        e.gated = bool(info.gated)
        e.requires_pickle = any(policy.classify(p) == "pickle" for p in chosen)
        sizes = {f.path: f.size for f in info.files}
        e.files_json = [
            {
                "role": "main",
                "source": "hf",
                "repo_id": repo_id,
                "revision": body.revision or "main",
                "allow_patterns": list(patterns),
                "files": [{"path": p, "size": sizes.get(p)} for p in chosen],
                "access": None,
                "gated": bool(info.gated),
            }
        ]
        e.size_bytes = sum(int(sizes.get(p) or 0) for p in chosen)
        e.meta_json = {
            **(e.meta_json or {}),
            "pipeline_tag": info.pipeline_tag,
            "library_name": info.library_name,
            "tags": info.tags[:20],
        }
        e.error = None
        e = self._save(e)
        return await self.install(e.id, _infos={repo_id: info})

    async def _resolve_hf(
        self, e: ModelEntry, infos: dict[str, RepoInfo] | None, *, raise_on_denied: bool
    ) -> ModelEntry:
        """Resolves each part's file list + access state (Section 9.3) and stores them on the entry."""
        token = self._token("hf_token")
        token_type = self.keys.status("hf_token").token_type
        infos = dict(infos or {})
        allow_pickle = self.settings.allow_pickle_weights
        parts = [dict(p) for p in e.files_json or []]
        messages: dict[str, str] = {}
        for p in parts:
            if p.get("source", "hf") != "hf":
                continue
            repo = str(p["repo_id"])
            rev = p.get("revision")
            try:
                info = infos.get(repo) or await self._repo_info(
                    repo, None if rev in (None, "main") else rev, token
                )
                infos[repo] = info
            except HubAccessError as err:
                p["access"] = err.access
                messages.setdefault(err.access, err.message)
                continue
            paths = {f.path: f for f in info.files}
            matched = [
                m
                for m in policy.match_patterns(paths, p.get("allow_patterns") or [])
                if policy.blocked_reason(m, allow_pickle) is None
                or (policy.classify(m) == "pickle" and e.requires_pickle and allow_pickle)
            ]
            if not matched:
                raise AppError(
                    502,
                    "catalog_files_missing",
                    f"No files in {repo} match {p.get('allow_patterns')}; the repo layout may have changed.",
                )
            p["files"] = [{"path": m, "size": paths[m].size} for m in matched]
            p["gated"] = bool(info.gated or info.private)
            p["sha"] = info.sha
            p["local_dir"] = self._rel(self.hf_local_dir(repo))
            access = await asyncio.to_thread(self.hf.access, info, token, token_type)
            p["access"] = access.state
            if access.state != "ok" and access.message:
                messages.setdefault(access.state, access.message)
        state = worst_access([p.get("access") for p in parts if p.get("source", "hf") == "hf"]) or "ok"
        denied_repo = next(
            (str(p["repo_id"]) for p in parts if p.get("access") == state and state != "ok"), None
        )
        account = self._account("hf_token").username
        e.files_json = parts
        e.access = state
        e.access_message = (
            self._access_message(denied_repo or e.source_ref, state, messages.get(state), account)
            if state != "ok"
            else None
        )
        e.access_checked_at = utcnow()
        e.gated = any(p.get("gated") for p in parts)
        if e.status not in _ACTIVE:
            e.size_bytes = sum(int(f.get("size") or 0) for p in parts for f in p.get("files") or [])
        e = self._save(e)
        if raise_on_denied and state != "ok":
            repo = denied_repo or e.source_ref
            raise HubAccessError(
                state,
                e.access_message or f"No access to {repo}.",
                {"model_entry_id": e.id, "repo_id": repo, "model_url": model_url(repo), "account": account},
            )
        return e

    # --- Civitai ---------------------------------------------------------------------------------
    def _civitai_index(self) -> dict[int, str]:
        with self.db.session() as s:
            rows = s.exec(select(ModelEntry).where(ModelEntry.source == "civitai")).all()
        out: dict[int, str] = {}
        for r in rows:
            civ = (r.meta_json or {}).get("civitai") or {}
            if civ.get("version_id"):
                out[int(civ["version_id"])] = r.id
        return out

    def _civitai_file(self, f: dict[str, Any]) -> CivitaiFileOut:
        meta = f.get("metadata") or {}
        fmt = meta.get("format")
        name = str(f.get("name") or "")
        size_kb = f.get("sizeKB")
        scan_ok = f.get("pickleScanResult") in (None, "Success") and f.get("virusScanResult") in (
            None,
            "Success",
        )
        allow_pickle = self.settings.allow_pickle_weights
        cls = policy.classify(name)
        reason: str | None = None
        if fmt == "PickleTensor" or cls == "pickle":
            reason = None if allow_pickle else policy.blocked_reason("x.ckpt", False)
        elif cls != "weight":
            reason = f"Unsupported file type ({fmt or PurePosixPath(name).suffix or 'unknown'})."
        if reason is None and not scan_ok:
            reason = "The file hasn't passed Civitai's safety scan."
        return CivitaiFileOut(
            id=int(f.get("id") or 0),
            name=name,
            size_bytes=round(float(size_kb) * 1024) if size_kb else None,
            format=fmt,
            fp=meta.get("fp"),
            file_type=f.get("type"),
            primary=bool(f.get("primary")),
            sha256=(f.get("hashes") or {}).get("SHA256"),
            scan_ok=scan_ok,
            blocked_reason=reason,
        )

    def _preview(self, images: list[dict[str, Any]]) -> str | None:
        for img in images or []:
            if img.get("type", "image") != "image":
                continue
            if not self.settings.allow_nsfw_models and int(img.get("nsfwLevel") or 1) > 1:
                continue
            if img.get("url"):
                return str(img["url"])
        return None

    def _civitai_version(self, mtype: str, v: dict[str, Any], idx: dict[int, str]) -> CivitaiVersionOut:
        sug = suggest_civitai(mtype, v.get("baseModel"))
        paid = v.get("paidAccess")
        early = bool(paid) or v.get("availability") not in (None, "Public")
        ends = paid.get("endsAt") if isinstance(paid, dict) else None
        usage = v.get("usageControl")
        downloadable = usage in (None, "Download")
        lock: str | None = None
        if not downloadable:
            lock = "The creator only allows on-site generation; downloads are disabled."
        elif early:
            lock = f"Early access until {ends[:10]}" if ends else "Paid / early access"
        return CivitaiVersionOut(
            id=int(v.get("id") or 0),
            name=str(v.get("name") or ""),
            base_model_raw=v.get("baseModel"),
            base_model=sug.base_model,
            adapter_key=sug.adapter_key,
            supported=sug.supported,
            capabilities=sug.capabilities,
            trained_words=[str(w) for w in v.get("trainedWords") or []],
            published_at=v.get("publishedAt"),
            early_access=early,
            early_access_ends_at=ends,
            downloadable=downloadable,
            lock_reason=lock,
            files=[self._civitai_file(f) for f in v.get("files") or []],
            preview_url=self._preview(v.get("images") or []),
            installed_entry_id=idx.get(int(v.get("id") or 0)),
        )

    def _civitai_model(
        self, m: dict[str, Any], idx: dict[int, str], *, detail: bool = False
    ) -> CivitaiModelOut:
        mtype = str(m.get("type") or "")
        versions = [self._civitai_version(mtype, v, idx) for v in m.get("modelVersions") or []]
        stats = m.get("stats") or {}
        return CivitaiModelOut(
            id=int(m.get("id") or 0),
            name=str(m.get("name") or ""),
            type=mtype,
            kind=CIVITAI_KINDS.get(mtype),
            nsfw=bool(m.get("nsfw")),
            creator=(m.get("creator") or {}).get("username"),
            tags=[str(t) for t in m.get("tags") or []][:12],
            description=_strip_html(m.get("description")) if detail else None,
            downloads=stats.get("downloadCount"),
            thumbs_up=stats.get("thumbsUpCount"),
            preview_url=next((v.preview_url for v in versions if v.preview_url), None),
            permissions={
                k: m.get(k)
                for k in ("allowCommercialUse", "allowDerivatives", "allowNoCredit", "allowDifferentLicense")
            },
            versions=versions,
            model_url=civitai_url(int(m.get("id") or 0)),
        )

    async def civitai_search(
        self,
        *,
        query: str | None,
        types: list[str],
        base_models: list[str],
        sort: str | None,
        cursor: str | None,
        limit: int,
        nsfw: bool,
    ) -> CivitaiSearchOut:
        allow_nsfw = self.settings.allow_nsfw_models
        params: list[tuple[str, str]] = [("limit", str(limit))]
        if query:
            params.append(("query", query))
        params += [("types", t) for t in types]
        params += [("baseModels", b) for b in base_models]
        if sort:
            params.append(("sort", sort))
        if cursor:
            params.append(("cursor", cursor))
        params.append(("nsfw", "true" if (allow_nsfw and nsfw) else "false"))
        data = await self.civitai.search(params, self._token("civitai_api_key"))
        idx = self._civitai_index()
        items: list[CivitaiModelOut] = []
        hidden = 0
        for m in data.get("items") or []:
            if m.get("nsfw") and not allow_nsfw:
                hidden += 1
                continue
            items.append(self._civitai_model(m, idx))
        next_cursor = (data.get("metadata") or {}).get("nextCursor")
        return CivitaiSearchOut(
            items=items, next_cursor=str(next_cursor) if next_cursor else None, hidden_nsfw=hidden
        )

    async def civitai_model(self, model_id: int) -> CivitaiModelOut:
        m = await self.civitai.model(model_id, self._token("civitai_api_key"))
        if m.get("nsfw") and not self.settings.allow_nsfw_models:
            raise AppError(
                403, "nsfw_blocked", "This model is marked NSFW; enable allow_nsfw_models to view it."
            )
        return self._civitai_model(m, self._civitai_index(), detail=True)

    async def civitai_install(self, body: CivitaiInstallBody) -> tuple[ModelEntry, Job]:
        token = self._token("civitai_api_key")
        v = await self.civitai.version(body.version_id, token)
        model_id = int(v.get("modelId") or (v.get("model") or {}).get("id") or 0)
        if not model_id:
            raise AppError(502, "hub_error", "Civitai returned a version without a model id.")
        m = await self.civitai.model(model_id, token)
        if (m.get("nsfw") or (v.get("model") or {}).get("nsfw")) and not self.settings.allow_nsfw_models:
            raise AppError(
                403, "nsfw_blocked", "This model is marked NSFW; enable allow_nsfw_models to install it."
            )
        mtype = str(m.get("type") or (v.get("model") or {}).get("type") or "")
        sug = suggest_civitai(mtype, v.get("baseModel"))
        if sug.kind == "other":
            raise bad_request("unsupported_type", sug.reason)
        version = self._civitai_version(mtype, v, {})
        if not version.downloadable:
            raise HubAccessError(
                "forbidden", version.lock_reason or "Downloads are disabled for this version."
            )
        files = v.get("files") or []
        raw = next((f for f in files if body.file_id and f.get("id") == body.file_id), None)
        if body.file_id and raw is None:
            raise bad_request("unknown_file", f"Version {body.version_id} has no file {body.file_id}.")
        if raw is None:
            raw = next((f for f in files if f.get("primary")), None) or next(iter(files), None)
        if raw is None:
            raise bad_request("no_files", "This version has no files.")
        fo = self._civitai_file(raw)
        if fo.blocked_reason:
            raise AppError(400, "file_blocked", f"'{fo.name}' can't be downloaded: {fo.blocked_reason}")
        adapter = body.adapter_key if body.adapter_key is not None else sug.adapter_key
        self._validate_adapter(adapter)
        ref = f"{model_id}:{body.version_id}"
        e = self._find("civitai", ref) or ModelEntry(source="civitai", source_ref=ref, name="")
        if e.status in _ACTIVE:
            raise conflict(
                "already_installed" if e.status == "installed" else "already_downloading",
                f"This version is already {e.status}.",
                {"model_entry_id": e.id},
            )
        safe_name = (
            _SAFE_NAME.sub("_", PurePosixPath(fo.name).name).strip(" .") or f"{body.version_id}.safetensors"
        )
        dest = self.settings.models_dir / "civitai" / str(model_id) / str(body.version_id) / safe_name
        download_url = str(raw.get("downloadUrl") or v.get("downloadUrl") or "")
        if not download_url.startswith("https://civitai.com/"):
            raise AppError(502, "hub_error", "Civitai returned an unexpected download URL.")
        e.name = f"{m.get('name') or 'Civitai model'} · {v.get('name') or body.version_id}"
        e.description = _strip_html(m.get("description"), 600) or ""
        e.kind = sug.kind
        e.base_model = sug.base_model
        e.adapter_key = adapter
        e.capabilities_json = (
            capabilities_for(adapter, sug.base_model) if adapter and sug.kind not in _COMPONENT_KINDS else []
        )
        e.license = "Civitai creator license (see permissions)"
        e.license_url = civitai_url(model_id, body.version_id)
        e.nsfw = bool(m.get("nsfw"))
        e.requires_pickle = policy.classify(safe_name) == "pickle"
        e.trigger_words_json = version.trained_words
        e.preview_url = version.preview_url or self._preview(
            [img for mv in m.get("modelVersions") or [] for img in mv.get("images") or []]
        )
        e.files_json = [
            {
                "role": "weights",
                "source": "civitai",
                "model_id": model_id,
                "version_id": body.version_id,
                "file_id": fo.id,
                "name": safe_name,
                "size": fo.size_bytes,
                "sha256": fo.sha256,
                "download_url": policy.safe_url(download_url),
                "local_path": self._rel(dest),
            }
        ]
        e.size_bytes = fo.size_bytes
        e.meta_json = {
            **(e.meta_json or {}),
            "civitai": CivitaiRef(
                model_id=model_id,
                version_id=body.version_id,
                file_id=fo.id,
                version_name=v.get("name"),
                type=mtype,
                base_model_raw=v.get("baseModel"),
                permissions={
                    k: m.get(k)
                    for k in (
                        "allowCommercialUse",
                        "allowDerivatives",
                        "allowNoCredit",
                        "allowDifferentLicense",
                    )
                },
            ).model_dump(),
            "creator": (m.get("creator") or {}).get("username"),
            "early_access": version.early_access,
        }
        e.error = None
        e = self._save(e)
        return await self.install(e.id)

    # --- install / download ----------------------------------------------------------------------
    def _validate_adapter(self, adapter_key: str | None) -> None:
        if adapter_key is not None and adapter_key not in KNOWN_ADAPTERS:
            raise bad_request(
                "unknown_adapter",
                f"Unknown adapter '{adapter_key}'. Known: {', '.join(sorted(KNOWN_ADAPTERS))}.",
            )

    def _expected_files(self, e: ModelEntry) -> list[tuple[Path, int | None]]:
        out: list[tuple[Path, int | None]] = []
        for p in e.files_json or []:
            if p.get("source") == "civitai":
                out.append((self._abs(str(p["local_path"])), p.get("size")))
            else:
                local = self.hf_local_dir(str(p["repo_id"]))
                out += [(local / f["path"], f.get("size")) for f in p.get("files") or []]
        return out

    def _remaining_bytes(self, e: ModelEntry) -> int:
        need = 0
        for path, size in self._expected_files(e):
            have = 0
            with contextlib.suppress(OSError):
                if path.is_file():
                    have = path.stat().st_size
                elif path.with_name(path.name + ".part").is_file():
                    have = path.with_name(path.name + ".part").stat().st_size
            need += max(0, int(size or 0) - have)
        return need

    async def install(
        self, entry_id: str, *, _infos: dict[str, RepoInfo] | None = None
    ) -> tuple[ModelEntry, Job]:
        e = self.get(entry_id)
        if e.status == "installed":
            raise conflict("already_installed", f"{e.name} is already installed.", {"model_entry_id": e.id})
        if e.status == "downloading":
            raise conflict(
                "already_downloading", f"{e.name} is already downloading.", {"model_entry_id": e.id}
            )
        if e.requires_pickle and not self.settings.allow_pickle_weights:
            raise conflict(
                "pickle_weights_blocked",
                f"{e.name} is only published as PyTorch pickle weights. Set allow_pickle_weights = true in "
                "config.toml to allow it (pickles can run code when loaded; LocalFlow loads them with "
                "weights_only=True).",
            )
        if e.source == "hf":
            e = await self._resolve_hf(e, _infos, raise_on_denied=True)
        elif e.source == "civitai":
            e = await self._check_civitai(e, raise_on_denied=True)
        policy.check_disk(self.settings.models_dir, self._remaining_bytes(e))
        job = self.jobs.create(
            type_="download_model", params={"model_entry_id": e.id, "name": e.name}, project_id=None
        )
        e = self._update(
            e.id,
            status="downloading",
            error=None,
            meta_json={**(e.meta_json or {}), "job_id": job.id},
        )
        return e, job

    async def _check_civitai(self, e: ModelEntry, *, raise_on_denied: bool) -> ModelEntry:
        part = (e.files_json or [{}])[0]
        token = self._token("civitai_api_key")
        try:
            state, message = await self.civitai.probe(str(part.get("download_url")), token), None
        except HubAccessError as err:
            state, message = err.access, err.message
        e.access, e.access_message, e.access_checked_at = state, message, utcnow()
        e.gated = state != "ok"
        e = self._save(e)
        if raise_on_denied and state != "ok":
            raise HubAccessError(
                state, message or "No access.", {"model_entry_id": e.id, "model_url": self._model_url(e)}
            )
        return e

    async def check_access(self, entry_id: str) -> ModelEntry:
        e = self.get(entry_id)
        if e.source == "hf":
            return await self._resolve_hf(e, None, raise_on_denied=False)
        if e.source == "civitai":
            return await self._check_civitai(e, raise_on_denied=False)
        return e

    async def run_download(self, entry_id: str, job_id: str, report: Report) -> dict[str, Any]:
        """Runs in the API process (Section 4.1). Tokens are resolved here, never stored in the job."""
        e = self.get(entry_id)
        if e.status == "installed":
            return {"model_entry_id": e.id, "already_installed": True}
        e = self._update(
            e.id, status="downloading", error=None, meta_json={**(e.meta_json or {}), "job_id": job_id}
        )
        if e.source == "hf":
            if not all(p.get("files") for p in e.files_json or []) or e.access != "ok":
                e = await self._resolve_hf(e, None, raise_on_denied=True)
            token = self._token("hf_token")
            parts = list(e.files_json or [])
            total = sum(int(f.get("size") or 0) for p in parts for f in p.get("files") or [])
            offset = 0
            for p in parts:
                files = [RepoFile(path=f["path"], size=f.get("size")) for f in p.get("files") or []]
                base = offset

                def on_progress(done: int, _total: int, base: int = base) -> None:
                    report(base + done, total)

                await self.hf.fetch(
                    FetchRequest(
                        repo_id=str(p["repo_id"]),
                        revision=None if p.get("revision") in (None, "main") else str(p["revision"]),
                        files=files,
                        local_dir=self.hf_local_dir(str(p["repo_id"])),
                        token=token,
                        hf_home=self.settings.resolved_hf_home,
                    ),
                    on_progress,
                )
                offset += sum(f.size or 0 for f in files)
            primary = next((p for p in parts if p.get("role") in ("base", "main")), parts[0])
            local_path = self._rel(self.hf_local_dir(str(primary["repo_id"])))
        elif e.source == "civitai":
            p = (e.files_json or [{}])[0]
            dest = self._abs(str(p["local_path"]))
            total = int(p.get("size") or 0)
            await self.civitai.download(
                str(p["download_url"]),
                dest,
                token=self._token("civitai_api_key"),
                expected_size=p.get("size"),
                sha256=p.get("sha256"),
                on_progress=lambda done, t: report(done, total or t),
            )
            local_path = self._rel(dest)
        else:
            raise bad_request("unsupported_source", f"Can't download '{e.source}' entries.")
        size = sum(path.stat().st_size for path, _ in self._expected_files(e) if path.is_file())
        meta = {k: v for k, v in (e.meta_json or {}).items() if k not in ("job_id", "partial")}
        self._update(
            e.id,
            status="installed",
            error=None,
            local_path=local_path,
            size_bytes=size,
            installed_at=utcnow(),
            access="ok",
            access_message=None,
            meta_json=meta,
        )
        return {"model_entry_id": e.id, "size_bytes": size, "local_path": local_path}

    def download_stopped(self, entry_id: str, *, interrupted: bool = False) -> None:
        with contextlib.suppress(AppError):
            e = self.get(entry_id)
            if e.status == "downloading":
                meta = {k: v for k, v in (e.meta_json or {}).items() if k != "job_id"}
                self._update(
                    e.id,
                    status="available",
                    error="Download interrupted. Install again to resume." if interrupted else None,
                    meta_json={**meta, "partial": True},
                )

    def download_failed(self, entry_id: str, err: AppError) -> None:
        with contextlib.suppress(AppError):
            e = self.get(entry_id)
            meta = {k: v for k, v in (e.meta_json or {}).items() if k != "job_id"}
            fields: dict[str, Any] = {
                "status": "error",
                "error": err.message,
                "meta_json": {**meta, "partial": True},
            }
            if isinstance(err, HubAccessError):
                fields |= {"access": err.access, "access_message": err.message, "access_checked_at": utcnow()}
            self._update(e.id, **fields)

    # --- delete / license ------------------------------------------------------------------------
    def delete(self, entry_id: str) -> DeleteModelOut:
        e = self.get(entry_id)
        if e.status == "downloading":
            job_id = (e.meta_json or {}).get("job_id")
            job_status = None
            if job_id:
                with contextlib.suppress(AppError):
                    job_status = self.jobs.get(job_id).status
            if job_status is not None and job_status not in TERMINAL:
                raise conflict("download_in_progress", "Cancel the download before deleting the model.")
        others = [o for o in self.list_entries() if o.id != e.id and o.status in _ACTIVE]
        shared_files = {
            (p.get("repo_id"), f["path"])
            for o in others
            for p in o.files_json or []
            if p.get("source", "hf") == "hf"
            for f in p.get("files") or []
        }
        shared_repos = {
            p.get("repo_id") for o in others for p in o.files_json or [] if p.get("source", "hf") == "hf"
        }
        freed = 0
        errors: list[str] = []

        def remove(path: Path) -> None:
            nonlocal freed
            try:
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    freed += size
            except OSError as err:
                errors.append(f"{path.name}: {err.strerror or err}")

        for p in e.files_json or []:
            if p.get("source") == "civitai":
                dest = self._abs(str(p["local_path"]))
                remove(dest)
                remove(dest.with_name(dest.name + ".part"))
                _prune_empty(dest.parent, stop=self.settings.models_dir / "civitai")
                continue
            repo = str(p.get("repo_id"))
            local = self.hf_local_dir(repo)
            if repo not in shared_repos:
                if local.exists():
                    size = policy.dir_size(local)
                    try:
                        shutil.rmtree(local)
                        freed += size
                    except OSError as err:
                        errors.append(f"{local.name}: {err.strerror or err}")
                continue
            for f in p.get("files") or []:
                if (repo, f["path"]) in shared_files:
                    continue
                remove(local / f["path"])
                remove(local / ".cache" / "huggingface" / "download" / f"{f['path']}.metadata")
            _prune_empty(local, stop=self.hf_repos_dir)
        if errors:
            raise conflict(
                "delete_failed",
                f"Some files couldn't be deleted (in use?): {'; '.join(errors[:3])}",
                {"freed_bytes": freed},
            )
        if e.catalog_key is None:
            with self.db.session() as s:
                row = s.get(ModelEntry, e.id)
                if row is not None:
                    s.delete(row)
                    s.commit()
            self.bus.publish("model.status", model_entry_id=e.id, deleted=True)
            return DeleteModelOut(id=e.id, freed_bytes=freed, deleted_entry=True)
        meta = {k: v for k, v in (e.meta_json or {}).items() if k not in ("job_id", "partial")}
        expected = sum(int(f.get("size") or 0) for p in e.files_json or [] for f in p.get("files") or [])
        e = self._update(
            e.id,
            status="available",
            error=None,
            local_path=None,
            installed_at=None,
            size_bytes=expected or meta.get("approx_size_bytes"),
            meta_json=meta,
        )
        return DeleteModelOut(id=e.id, freed_bytes=freed, deleted_entry=False, entry=self.to_out(e))

    def acknowledge_license(self, entry_id: str) -> ModelEntry:
        e = self.get(entry_id)
        if e.license_acknowledged_at is not None:
            return e
        return self._update(e.id, license_acknowledged_at=utcnow())

    def require_usable(self, entry_id: str, kind: str) -> ModelEntry:
        """Checked when a generation job names a model (Section 9.5)."""
        try:
            e = self.get(entry_id)
        except AppError as err:
            raise bad_request("unknown_model", f"Model '{entry_id}' isn't in the registry.") from err
        if e.status != "installed":
            raise conflict(
                "model_not_installed", f"{e.name} isn't installed. Install it from the Models page."
            )
        if e.kind != "lora" and e.capabilities_json and kind not in e.capabilities_json:
            raise bad_request("model_capability", f"{e.name} doesn't support '{kind}'.")
        if e.license_acknowledged_at is None:
            raise conflict(
                "license_not_acknowledged",
                f"Review the license for {e.name} ({e.license or 'unknown license'}) and acknowledge it "
                "first.",
                {"model_entry_id": e.id, "license": e.license, "license_url": e.license_url},
            )
        return e

    # --- worker specs (Section 10) ---------------------------------------------------------------------
    def model_spec(self, entry_id: str) -> dict[str, Any]:
        """Resolves an installed entry to absolute paths for the worker (it never reads the database)."""
        e = self.get(entry_id)
        if e.status != "installed":
            raise conflict(
                "model_not_installed", f"{e.name} isn't installed. Install it from the Models page."
            )
        parts: list[PartSpec] = []
        single_file: str | None = None
        for p in e.files_json or []:
            if p.get("source") == "civitai":
                single_file = str(self._abs(str(p["local_path"])))
            else:
                parts.append(
                    PartSpec(
                        role=str(p.get("role") or "main"),
                        local_dir=str(self.hf_local_dir(str(p["repo_id"]))),
                        files=tuple(f["path"] for f in p.get("files") or []),
                    )
                )
        if single_file is None and e.kind == "checkpoint" and len(parts) == 1:
            files = parts[0].files
            weights = [f for f in files if policy.classify(f) == "weight"]
            if "model_index.json" not in files and len(weights) == 1:
                single_file = str(Path(parts[0].local_dir) / weights[0])  # e.g. a single-file SDXL repo
        aux: dict[str, str] = {}
        if e.adapter_key == "sdxl" and single_file:
            base = self._installed_catalog("sdxl-base")
            if base is not None:
                for p in base.files_json or []:
                    if p.get("role") == "base":
                        aux["sdxl_config_dir"] = str(self.hf_local_dir(str(p["repo_id"])))
                    elif p.get("role") == "vae":
                        aux["vae_dir"] = str(self.hf_local_dir(str(p["repo_id"])))
        return ModelSpec(
            entry_id=e.id,
            name=e.name,
            adapter_key=e.adapter_key,
            kind=e.kind,
            base_model=e.base_model,
            source=e.source,
            parts=tuple(parts),
            single_file=single_file,
            default_params=dict(e.default_params_json or {}),
            aux=aux,
        ).to_dict()

    def _installed_catalog(self, key: str) -> ModelEntry | None:
        with self.db.session() as s:
            e = s.exec(select(ModelEntry).where(ModelEntry.catalog_key == key)).first()
        return e if e is not None and e.status == "installed" else None

    def lora_spec(self, entry_id: str, weight: float, model_entry_id: str | None) -> dict[str, Any]:
        e = self.get(entry_id)
        if e.kind != "lora":
            raise bad_request("not_a_lora", f"{e.name} isn't a LoRA.")
        if e.status != "installed":
            raise conflict(
                "model_not_installed", f"{e.name} isn't installed. Install it from the Models page."
            )
        if model_entry_id:
            model = self.get(model_entry_id)
            if e.adapter_key and model.adapter_key and e.adapter_key != model.adapter_key:
                raise bad_request(
                    "lora_incompatible",
                    f"{e.name} is a {e.base_model or e.adapter_key} LoRA; "
                    f"it can't be used with {model.name}.",
                )
        path: Path | None = None
        for p in e.files_json or []:
            if p.get("source") == "civitai":
                path = self._abs(str(p["local_path"]))
            else:
                local = self.hf_local_dir(str(p["repo_id"]))
                weights = [f["path"] for f in p.get("files") or [] if str(f["path"]).endswith(".safetensors")]
                if weights:
                    path = local / weights[0]
        if path is None:
            raise bad_request("lora_file_missing", f"{e.name} has no .safetensors file.")
        return LoraSpec(entry_id=e.id, name=e.name, path=str(path), weight=weight).to_dict()

    def default_entry(
        self, kind: str, *, adapter_key: str | None, usable: set[str] | None, preferred: list[str | None]
    ) -> ModelEntry | None:
        """The model a job runs on when none is chosen: the project default, else the first installed
        catalog-ordered entry whose adapter can run `kind` (Section 11)."""
        candidates = [
            e
            for e in self.list_entries(status="installed", capability=kind)
            if e.kind not in ("lora", *_COMPONENT_KINDS)
            and e.adapter_key
            and (adapter_key is None or e.adapter_key == adapter_key)
            and (usable is None or e.adapter_key in usable)
        ]
        for pid in preferred:
            for e in candidates:
                if pid and e.id == pid:
                    return e
        return candidates[0] if candidates else None

    def render_profile(self, entry_id: str | None) -> tuple[str, dict[str, Any]] | None:
        """(name, render constraints) of a video model, for the story pipeline (Section 8.2)."""
        if not entry_id:
            return None
        with self.db.session() as s:
            e = s.get(ModelEntry, entry_id)
        if e is None:
            return None
        render = (e.meta_json or {}).get("render")
        if not isinstance(render, dict):
            caps = [c for c in e.capabilities_json or [] if c in ("t2v", "i2v", "flf2v", "ref2v")]
            if not caps:
                return None
            render = {"supported_modes": caps}
        return e.name, render


def _prune_empty(path: Path, *, stop: Path) -> None:
    """Removes empty directories below and including `path`, then empty ancestors strictly below `stop`."""
    if not path.exists():
        return
    for d in sorted((p for p in path.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            d.rmdir()
    cur = path
    while cur != stop and stop in cur.parents:
        with contextlib.suppress(OSError):
            cur.rmdir()
        cur = cur.parent
