from __future__ import annotations

from dataclasses import dataclass

import httpx2
from fastapi import Request

from localflow.config import Settings
from localflow.db import Database
from localflow.models import Asset, Job
from localflow.services.assets import AssetService
from localflow.services.audiogen import AudioGenService
from localflow.services.board import BoardService
from localflow.services.characters.library import LibraryService
from localflow.services.characters.service import CharacterService
from localflow.services.characters.templates import TemplateService
from localflow.services.collections import CollectionService
from localflow.services.events import EventBus
from localflow.services.imagegen import ImageGenService
from localflow.services.jobs.generation import GenerationService
from localflow.services.jobs.service import JobService
from localflow.services.jobs.supervisor import WorkerSupervisor
from localflow.services.keys import Checker, KeyService
from localflow.services.llm.service import LLMService, ProviderFactory
from localflow.services.narration import NarrationService
from localflow.services.projects import ProjectService
from localflow.services.registry.civitai import CivitaiClient
from localflow.services.registry.downloads import DownloadRunner
from localflow.services.registry.hf import HfBackend
from localflow.services.registry.service import RegistryService
from localflow.services.review import ReviewService
from localflow.services.secrets import SecretStore, build_secret_store
from localflow.services.story.service import StoryService
from localflow.services.timeline.export import ExportRunner
from localflow.services.timeline.service import TimelineService
from localflow.services.videogen import VideoGenService
from localflow.services.voices import VoiceService


@dataclass
class Services:
    settings: Settings
    db: Database
    bus: EventBus
    jobs: JobService
    assets: AssetService
    projects: ProjectService
    generation: GenerationService
    supervisor: WorkerSupervisor
    secrets: SecretStore
    keys: KeyService
    llm: LLMService
    story: StoryService
    board: BoardService
    registry: RegistryService
    downloads: DownloadRunner
    imagegen: ImageGenService
    templates: TemplateService
    library: LibraryService
    characters: CharacterService
    videogen: VideoGenService
    timeline: TimelineService
    exports: ExportRunner
    voices: VoiceService
    audiogen: AudioGenService
    collections: CollectionService
    review: ReviewService
    narration: NarrationService


def build_services(
    settings: Settings,
    *,
    secrets: SecretStore | None = None,
    llm_factory: ProviderFactory | None = None,
    key_checker: Checker | None = None,
    hf_backend: HfBackend | None = None,
    civitai_transport: httpx2.AsyncBaseTransport | None = None,
) -> Services:
    """Test hooks: a secret store, an LLM provider factory (e.g. FakeLLMProvider), a key checker, a fake
    Hugging Face backend, and an httpx2 transport for Civitai."""
    db = Database(settings.db_path)
    bus = EventBus()
    jobs = JobService(db, bus)
    assets = AssetService(settings, db)
    projects = ProjectService(db, assets)
    holder: dict[str, WorkerSupervisor] = {}
    secrets = secrets or build_secret_store(
        settings.secret_backend, settings.data_dir, settings.secrets_from_env
    )
    keys = KeyService(db, secrets, key_checker)
    registry = RegistryService(
        settings,
        db,
        bus,
        jobs,
        secrets,
        keys,
        hf=hf_backend,
        civitai=CivitaiClient(transport=civitai_transport),
        adapters=lambda: holder["sup"].adapters if "sup" in holder else None,
    )
    generation = GenerationService(
        settings,
        jobs,
        assets,
        projects,
        adapters=lambda: holder["sup"].adapters,
        registry=registry,
    )
    supervisor = WorkerSupervisor(settings, jobs, generation, assets, bus)
    holder["sup"] = supervisor
    imagegen = ImageGenService(db, generation, registry, bus)
    downloads = DownloadRunner(jobs, registry, bus, settings.max_concurrent_downloads)
    llm = LLMService(settings, db, keys, llm_factory)
    story = StoryService(settings, db, llm, bus, video_profile=registry.render_profile)
    board = BoardService(settings, db, bus)
    videogen = VideoGenService(settings, db, generation, registry, imagegen, board, bus)
    timeline = TimelineService(db, bus)
    voices = VoiceService(settings, db, bus, generation, registry, assets, llm, generation.adapters)
    audiogen = AudioGenService(db, generation, registry, board, voices, timeline, bus)

    def on_assets(job: Job, assets: list[Asset]) -> None:
        imagegen.on_job_assets(job, assets)
        videogen.on_job_assets(job, assets)
        audiogen.on_job_assets(job, assets)
        voices.on_job_assets(job, assets)

    supervisor.on_assets = on_assets
    generation.voice_resolver = voices.resolve_tts
    templates = TemplateService(settings, db)
    library = LibraryService(settings, db, bus, assets)
    characters = CharacterService(
        settings, db, bus, assets, board, generation, registry, llm, projects, templates, voices
    )
    generation.mention_resolver = characters.resolve_for_generation
    exports = ExportRunner(settings, jobs, assets, timeline, bus)
    narration = NarrationService(db, llm, story, board, voices, generation, registry, timeline)
    return Services(
        settings,
        db,
        bus,
        jobs,
        assets,
        projects,
        generation,
        supervisor,
        secrets,
        keys,
        llm,
        story,
        board,
        registry,
        downloads,
        imagegen,
        templates,
        library,
        characters,
        videogen,
        timeline,
        exports,
        voices,
        audiogen,
        CollectionService(db, bus),
        ReviewService(settings, assets, imagegen, llm),
        narration,
    )


def services(request: Request) -> Services:
    return request.app.state.services
