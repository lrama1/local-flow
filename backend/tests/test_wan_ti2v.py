"""The lean TI2V forward (wan_ti2v) matches diffusers' per-token forward on a tiny model (CPU, fp32)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from localflow.worker.adapters import wan_ti2v  # noqa: E402


def _tiny_model() -> object:
    torch.manual_seed(0)
    model = diffusers.WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=24,
        freq_dim=16,
        ffn_dim=48,
        num_layers=2,
        cross_attn_norm=True,
        rope_max_seq_len=64,
    )
    # Random non-zero modulation tables, so every shift/scale/gate path matters.
    for block in model.blocks:
        block.scale_shift_table.data.normal_()
    return model.eval()


def _run(model: object, timestep: torch.Tensor) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, 3, 8, 12, generator=g)
    text = torch.randn(1, 7, 24, generator=g)
    with torch.inference_mode():
        return model(hidden_states=x, timestep=timestep, encoder_hidden_states=text, return_dict=False)[0]


def _per_token(first: float, rest: float) -> torch.Tensor:
    mask = torch.ones(3, 4, 6)  # latent frames x (8/2) x (12/2) tokens
    mask[0] = 0
    return (mask * rest + (1 - mask) * first).flatten().unsqueeze(0)


@pytest.mark.parametrize(("first", "rest"), [(0.0, 700.0), (500.0, 500.0)])
def test_lean_forward_matches_stock(first: float, rest: float) -> None:
    model = _tiny_model()
    timestep = _per_token(first, rest)
    stock = _run(model, timestep)
    wan_ti2v.apply(model)
    lean = _run(model, timestep)
    torch.testing.assert_close(lean, stock, rtol=1e-5, atol=1e-5)


def test_default_negative_never_negates_what_the_prompt_asks_for() -> None:
    from localflow.worker.adapters.wan import DEFAULT_NEGATIVE, negative_prompt

    full = negative_prompt("A man walks along the pier", "blurry", True)
    assert full == f"blurry, {DEFAULT_NEGATIVE}" and ", static," in full
    static = negative_prompt("He sits. Static camera, a locked-off shot.", "camera shake", True)
    assert static.startswith("camera shake, bright tones")
    assert "static" not in static.split(", ") and "still picture" in static
    # Whole terms only: "images" stays even though the prompt says "image".
    assert "images" in negative_prompt("An image of a harbor", "", True).split(", ")
    assert negative_prompt("Static camera", "blurry", False) == "blurry"


def test_scalar_timestep_path_is_untouched() -> None:
    model = _tiny_model()
    timestep = torch.tensor([700.0])
    stock = _run(model, timestep)
    wan_ti2v.apply(model)
    wan_ti2v.apply(model)  # idempotent
    torch.testing.assert_close(_run(model, timestep), stock, rtol=0, atol=0)
