"""Lower-memory forward for Wan 2.2 TI2V-5B's per-token timesteps (DECISIONS M4).

TI2V conditions on the first latent frame by giving its tokens timestep 0 and every other token the step's
timestep, so diffusers embeds the timestep per token. At 1280x704 with 121 frames (27,280 tokens) that is a
(27,280, 6 x 3072) bf16 projection (1 GB), kept for the whole forward. On top of that, every block builds all
six fp32 modulation tensors at once (2 GB). There are only one or two distinct timesteps, so this embeds each
distinct value once and each block gathers a modulation tensor only when it uses it. The math is the same,
with about 3 GB less peak memory at 720p.

`apply` patches one transformer instance: its condition embedder and block forwards. The class and the
Wan 2.1 / A14B path (one timestep per sample) are untouched."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class _TokenRows:
    """Shared by one transformer's embedder and blocks: token → row of the distinct-timestep embeddings."""

    def __init__(self) -> None:
        self.index: torch.Tensor | None = None


def apply(transformer: Any) -> None:
    if getattr(transformer, "_localflow_lean_ti2v", False):
        return
    rows = _TokenRows()
    embedder = transformer.condition_embedder
    embedder.forward = _lean_embed(embedder.forward, rows)
    for block in transformer.blocks:
        block.forward = _lean_block_forward(block, rows)
    transformer._localflow_lean_ti2v = True


def _lean_embed(original: Callable[..., Any], rows: _TokenRows) -> Callable[..., Any]:
    def embed(
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ) -> Any:
        if timestep_seq_len is None:
            rows.index = None
            return original(timestep, encoder_hidden_states, encoder_hidden_states_image, None)
        per_token = timestep.unflatten(0, (-1, timestep_seq_len))  # (B, seq)
        distinct, index = torch.unique(per_token, return_inverse=True)  # (U,), (B, seq)
        temb, proj, text, image = original(distinct, encoder_hidden_states, encoder_hidden_states_image, None)
        rows.index = index
        # temb only feeds the output norm; per token it's (B, seq, dim), small.
        # proj stays per distinct timestep: (B, U, 6*dim), which the model unflattens to (B, U, 6, dim).
        return temb[index], proj.unsqueeze(0).expand(per_token.shape[0], -1, -1), text, image

    return embed


def _lean_block_forward(block: Any, rows: _TokenRows) -> Callable[..., torch.Tensor]:
    original = block.forward

    def forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Any,
    ) -> torch.Tensor:
        index = rows.index
        if temb.ndim != 4 or index is None:
            return original(hidden_states, encoder_hidden_states, temb, rotary_emb)
        table = block.scale_shift_table  # (1, 6, dim)
        batch = torch.arange(index.shape[0], device=index.device)[:, None]

        def mod(i: int) -> torch.Tensor:
            # shift, scale, gate (attention), then shift, scale, gate (feed-forward): (B, seq, dim) fp32
            return (table[:, i] + temb[:, :, i].float())[batch, index]

        x = block.norm1(hidden_states.float())
        x = x.mul_(mod(1).add_(1)).add_(mod(0)).type_as(hidden_states)
        out = block.attn1(x, None, None, rotary_emb)
        del x
        hidden_states = (hidden_states.float() + out * mod(2)).type_as(hidden_states)
        del out

        x = block.norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + block.attn2(x, encoder_hidden_states, None, None)
        del x

        x = block.norm3(hidden_states.float())
        x = x.mul_(mod(4).add_(1)).add_(mod(3)).type_as(hidden_states)
        out = block.ffn(x)
        del x
        return (hidden_states.float() + out.float() * mod(5)).type_as(hidden_states)

    return forward
