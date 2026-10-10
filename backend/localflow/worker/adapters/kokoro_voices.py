"""Kokoro-82M built-in voices (Section 12.2). Grades are from the model card (VOICES.md); traits are
LocalFlow's short descriptions, used for suggestions and the voice designer. Japanese and Mandarin voices are
left out: they need extra G2P packages (misaki[ja] / misaki[zh])."""

from __future__ import annotations

from localflow.worker.adapters.speech import BuiltinVoice

# Kokoro's lang_code is the first letter of the voice id.
LANG_BY_PREFIX = {"a": "en-US", "b": "en-GB", "e": "es", "f": "fr", "h": "hi", "i": "it", "p": "pt-BR"}


def _v(native_id: str, name: str, quality: str, traits: str) -> BuiltinVoice:
    gender = "female" if native_id[1] == "f" else "male"
    return BuiltinVoice(native_id, name, LANG_BY_PREFIX[native_id[0]], gender, quality, traits)


KOKORO_VOICES: tuple[BuiltinVoice, ...] = (
    # American English
    _v("af_heart", "Heart", "A", "warm, expressive, natural"),
    _v("af_bella", "Bella", "A-", "bright, lively, youthful"),
    _v("af_nicole", "Nicole", "B-", "soft, breathy, intimate"),
    _v("af_aoede", "Aoede", "C+", "smooth, gentle"),
    _v("af_kore", "Kore", "C+", "clear, confident"),
    _v("af_sarah", "Sarah", "C+", "composed, mature, clear"),
    _v("af_alloy", "Alloy", "C", "neutral, even"),
    _v("af_nova", "Nova", "C", "crisp, professional"),
    _v("af_sky", "Sky", "C-", "light, airy, young"),
    _v("af_jessica", "Jessica", "D", "casual, friendly"),
    _v("af_river", "River", "D", "relaxed, low"),
    _v("am_fenrir", "Fenrir", "C+", "strong, deep, commanding"),
    _v("am_michael", "Michael", "C+", "warm, steady, trustworthy"),
    _v("am_puck", "Puck", "C+", "playful, lively, youthful"),
    _v("am_echo", "Echo", "D", "resonant, calm"),
    _v("am_eric", "Eric", "D", "confident, upbeat"),
    _v("am_liam", "Liam", "D", "young, friendly"),
    _v("am_onyx", "Onyx", "D", "deep, smooth, low"),
    _v("am_santa", "Santa", "D-", "jolly, old, hearty"),
    _v("am_adam", "Adam", "F+", "plain, flat"),
    # British English
    _v("bf_emma", "Emma", "B-", "refined, warm, mature"),
    _v("bf_isabella", "Isabella", "C", "polished, elegant"),
    _v("bf_alice", "Alice", "D", "light, proper"),
    _v("bf_lily", "Lily", "D", "soft, young"),
    _v("bm_george", "George", "C", "distinguished, older, deep"),
    _v("bm_fable", "Fable", "C", "storyteller, warm, expressive"),
    _v("bm_lewis", "Lewis", "D+", "grave, deep, serious"),
    _v("bm_daniel", "Daniel", "D", "calm, measured"),
    # Spanish, French, Hindi, Italian, Brazilian Portuguese (espeak-ng G2P)
    _v("ef_dora", "Dora", "", "warm, clear"),  # Spanish and Portuguese voices are ungraded
    _v("em_alex", "Alex", "", "friendly, clear"),
    _v("em_santa", "Santa (es)", "", "jolly, old, hearty"),
    _v("ff_siwis", "Siwis", "B-", "clear, natural"),
    _v("hf_alpha", "Alpha", "C", "clear, warm"),
    _v("hf_beta", "Beta", "C", "soft, gentle"),
    _v("hm_omega", "Omega", "C", "calm, deep"),
    _v("hm_psi", "Psi", "C", "clear, steady"),
    _v("if_sara", "Sara", "C", "warm, clear"),
    _v("im_nicola", "Nicola", "C", "friendly, clear"),
    _v("pf_dora", "Dora (pt)", "", "warm, clear"),
    _v("pm_alex", "Alex (pt)", "", "friendly, clear"),
    _v("pm_santa", "Santa (pt)", "", "jolly, old, hearty"),
)
