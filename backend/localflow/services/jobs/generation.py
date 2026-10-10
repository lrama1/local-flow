"""Turns `POST /generate/{kind}` bodies into jobs, and jobs into worker requests."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from localflow.config import Settings
from localflow.errors import AppError, bad_request, conflict
from localflow.models import Job, Project
from localflow.schemas.generate import ASSET_FIELDS, COMMON_FIELDS, GenerateCommon
from localflow.services.assets import AssetService
from localflow.services.jobs.service import JobService
from localflow.services.projects import ProjectService
from localflow.worker.adapters.base import REQUEST_MODELS
from localflow.worker.messages import JobRequest

if TYPE_CHECKING:
    from localflow.services.characters.mentions import Resolved
    from localflow.services.registry.service import RegistryService

FAKE = "fake"
_VIDEO_KINDS = {"t2v", "i2v", "flf2v", "ref2v", "extend"}

# (job kind, project id, prompt) -> resolved mentions, or None if nothing to resolve; raises 422 on unknown.
MentionResolver = Callable[[str, str, str], "Resolved | None"]


class GenerationService:
    def __init__(
        self,
        settings: Settings,
        jobs: JobService,
        assets: AssetService,
        projects: ProjectService,
        adapters: Callable[[], list[dict[str, Any]] | None],
        registry: RegistryService | None = None,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.assets = assets
        self.projects = projects
        self._adapters = adapters
        self.registry = registry
        self.mention_resolver: MentionResolver | None = None  # set by the container (characters service)
        # TTS: LocalFlow voice ids -> adapter-native voices (+ picks the adapter); set by the container.
        self.voice_resolver: Callable[[Any, dict[str, Any]], None] | None = None

    # --- adapters --------------------------------------------------------------------------------
    def adapters(self) -> list[dict[str, Any]] | None:
        """The worker's adapter descriptions (AdapterInfo dicts), or None while it's starting."""
        return self._adapters()

    def usable_adapters(self, kind: str) -> set[str] | None:
        """Adapter keys the worker can run for `kind`, or None while the worker is starting."""
        adapters = self._adapters()
        if adapters is None:
            return None
        return {
            str(a["adapter_key"])
            for a in adapters
            if a.get("available") and kind in a.get("capabilities", [])
        }

    def model_adapters(self, kind: str) -> set[str] | None:
        """Adapter filter for picking installed models; None = any (worker starting, or FakeEngine)."""
        usable = self.usable_adapters(kind)
        return None if usable is None or FAKE in usable else usable

    def _check_adapter(self, kind: str, adapter_key: str) -> None:
        usable = self.usable_adapters(kind)
        if usable is not None and adapter_key not in usable:
            raise conflict("adapter_unavailable", f"Adapter '{adapter_key}' is not available for '{kind}'.")

    def _use_fake(self, kind: str, requested: str | None) -> bool:
        if requested == FAKE:
            return True
        if requested:
            return False
        usable = self.usable_adapters(kind)
        if usable is None:
            return self.settings.engine == FAKE
        return FAKE in usable

    def resolve(self, kind: str, body: GenerateCommon, request: dict[str, Any], project: Project) -> str:
        """Picks the adapter and, for real adapters, the model entry (written into `request`)."""
        entry_id = request.get("model_entry_id")
        if entry_id:
            if self.registry is None:
                raise conflict("no_registry", "Model registry unavailable.")
            entry = self.registry.require_usable(str(entry_id), kind)
            adapter_key = body.adapter_key or entry.adapter_key
            if not adapter_key:
                raise bad_request("model_has_no_adapter", f"{entry.name} has no adapter assigned.")
            usable = self.usable_adapters(kind)
            if usable is not None and adapter_key not in usable and FAKE in usable:
                return FAKE  # the FakeEngine stands in for every adapter (dev/tests)
            if adapter_key != FAKE:
                self._check_adapter(kind, adapter_key)
            return adapter_key
        if self._use_fake(kind, body.adapter_key):
            if self.usable_adapters(kind) is not None:
                self._check_adapter(kind, FAKE)
            return FAKE
        if self.registry is None:
            raise conflict("no_model_installed", f"No installed model can run '{kind}'.")
        preferred = (
            [project.default_video_model_id] if kind in _VIDEO_KINDS else [project.default_image_model_id]
        )
        default = self.registry.default_entry(
            kind, adapter_key=body.adapter_key, usable=self.usable_adapters(kind), preferred=preferred
        )
        if default is None:
            raise conflict(
                "no_model_installed",
                f"No installed model can run '{kind}'. Install one from the Models page.",
                {"kind": kind},
            )
        self.registry.require_usable(default.id, kind)
        request["model_entry_id"] = default.id
        return str(default.adapter_key)

    # --- submit ----------------------------------------------------------------------------------
    def submit(self, kind: str, body: GenerateCommon, *, chain: dict[str, Any] | None = None) -> Job:
        """`chain` is API-side follow-up work run when the job's outputs arrive (e.g. extend the new take);
        it's stored in the job params but never sent to the worker."""
        project_id = body.project_id or self.projects.ensure_playground().id
        project = self.projects.get(project_id)
        request: dict[str, Any] = body.model_dump(exclude=COMMON_FIELDS)
        prompt_raw, mention_keys = body.prompt_raw, body.mentions or []
        prompt = request.get("prompt")
        if prompt_raw is None and isinstance(prompt, str) and "@" in prompt and self.mention_resolver:
            resolved = self.mention_resolver(kind, project_id, prompt)
            if resolved is not None:
                request["prompt"], prompt_raw, mention_keys = resolved.text, resolved.raw, resolved.keys
        if kind == "tts" and self.voice_resolver is not None:
            self.voice_resolver(body, request)
        adapter_key = self.resolve(kind, body, request, project)
        if adapter_key != FAKE and self.registry is not None:
            for lora in request.get("loras") or []:
                self.registry.require_usable(str(lora["model_entry_id"]), kind)
                self.registry.lora_spec(
                    str(lora["model_entry_id"]), float(lora.get("weight", 1.0)), request.get("model_entry_id")
                )
        for field, (_path_field, expected) in ASSET_FIELDS.get(kind, {}).items():
            ids = request[field] if isinstance(request[field], list) else [request[field]]
            for asset_id in ids:
                asset = self.assets.get(asset_id)
                if asset.type != expected:
                    raise bad_request(
                        "wrong_asset_type", f"Asset '{asset_id}' is a {asset.type}, expected {expected}"
                    )
        if request.get("seed") is None:
            request["seed"] = random.randint(0, 2**32 - 1)
        params: dict[str, Any] = {"adapter_key": adapter_key, "request": request}
        if prompt_raw is not None and prompt_raw != request.get("prompt"):
            params["prompt_raw"], params["mentions"] = prompt_raw, mention_keys
        if chain:
            params["chain"] = chain
        return self.jobs.create(
            type_=kind,
            params=params,
            project_id=project_id,
            priority=body.priority,
            target_ref=body.target_ref,
        )

    # --- dispatch --------------------------------------------------------------------------------
    def build_request(self, job: Job) -> JobRequest:
        params = job.params_json
        adapter_key = str(params.get("adapter_key"))
        request = dict(params.get("request") or {})
        for field, (path_field, expected) in ASSET_FIELDS.get(job.type, {}).items():
            value = request.pop(field, None)
            if isinstance(value, list):
                request[path_field] = [self._asset_path(v, expected) for v in value]
            elif isinstance(value, str):
                request[path_field] = self._asset_path(value, expected)
        model: type[BaseModel] = REQUEST_MODELS[job.type]
        try:
            model.model_validate(request)
        except ValidationError as e:
            raise bad_request(
                "invalid_request", f"Invalid {job.type} request: {e.errors()[0].get('msg')}"
            ) from e
        entry_id = request.get("model_entry_id")
        spec: dict[str, Any] | None = None
        loras: list[dict[str, Any]] = []
        if adapter_key != FAKE and self.registry is not None:
            if entry_id:
                spec = self.registry.model_spec(str(entry_id))
            loras = [
                self.registry.lora_spec(str(lo["model_entry_id"]), float(lo.get("weight", 1.0)), entry_id)
                for lo in request.get("loras") or []
            ]
        out_dir = self.settings.tmp_dir / "jobs" / job.id
        return JobRequest(
            job_id=job.id,
            job_type=job.type,
            adapter_key=adapter_key,
            params=request,
            output_dir=str(out_dir),
            preview_dir=str(self.settings.tmp_dir / "previews"),
            seed=int(request.get("seed") or 0),
            vram_profile=self.settings.vram_profile,
            model_entry_id=entry_id,
            model=spec,
            loras=loras,
        )

    def _asset_path(self, asset_id: str, expected: str) -> str:
        asset = self.assets.get(asset_id)
        if asset.type != expected:
            raise bad_request(
                "wrong_asset_type", f"Asset '{asset_id}' is a {asset.type}, expected {expected}"
            )
        path = self.assets.file_path(asset)
        if not path.exists():
            raise AppError(410, "asset_file_missing", f"File for asset '{asset_id}' is missing on disk")
        return str(path)
