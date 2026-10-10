"""Generic generation (Playground, outside the story flow): `POST /api/generate/{kind}` → job."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from localflow.schemas.core import JobOut
from localflow.schemas.generate import (
    ExtendBody,
    FLF2VBody,
    I2VBody,
    ImageEditBody,
    MusicBody,
    Ref2VBody,
    T2IBody,
    T2VBody,
    TTSBody,
    V2ABody,
)
from localflow.services.container import Services, services

router = APIRouter(prefix="/api/generate", tags=["generate"])


@router.post("/t2i", response_model=JobOut, status_code=201)
def generate_t2i(body: T2IBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("t2i", body))


@router.post("/image_edit", response_model=JobOut, status_code=201)
def generate_image_edit(body: ImageEditBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("image_edit", body))


@router.post("/t2v", response_model=JobOut, status_code=201)
def generate_t2v(body: T2VBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("t2v", body))


@router.post("/i2v", response_model=JobOut, status_code=201)
def generate_i2v(body: I2VBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("i2v", body))


@router.post("/flf2v", response_model=JobOut, status_code=201)
def generate_flf2v(body: FLF2VBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("flf2v", body))


@router.post("/ref2v", response_model=JobOut, status_code=201)
def generate_ref2v(body: Ref2VBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("ref2v", body))


@router.post("/extend", response_model=JobOut, status_code=201)
def generate_extend(body: ExtendBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("extend", body))


@router.post("/tts", response_model=JobOut, status_code=201)
def generate_tts(body: TTSBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("tts", body))


@router.post("/v2a", response_model=JobOut, status_code=201)
def generate_v2a(body: V2ABody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("v2a", body))


@router.post("/music", response_model=JobOut, status_code=201)
def generate_music(body: MusicBody, svc: Services = Depends(services)) -> JobOut:
    return JobOut.of(svc.generation.submit("music", body))
