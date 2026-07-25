# SPDX-License-Identifier: Apache-2.0
"""Steering to Target Projection (StTP).

Projection-based steering along a learned trait direction. For tokens whose
projection p = <h, v_hat> falls on the "wrong side" of a gating threshold,
the component of h along v_hat is replaced so that the projection equals a
precomputed target value s, leaving the orthogonal subspace untouched:

    h' = h + (s - p) * v_hat        (only where the gate mask is true)

Direction semantics:
  * direction == "positive": steer tokens with p < threshold up to s
    (s = mu_pos + coefficient * std_pos, computed client-side)
  * direction == "negative": steer tokens with p > threshold down to s
    (s = mu_neg + coefficient * std_neg, computed client-side)

The per-layer statistics (normalized direction v_hat, gating threshold,
target s) are precomputed by the client from the positive/negative training
activation distributions and shipped as a small dict payload — the raw
[L, N, H] embedding tensors never enter vLLM.

Expected .pt payload format (dict):
    {
        "v_hat":     torch.Tensor [H]  — L2-normalized steering direction
        "threshold": float             — gating threshold in projection space
        "target":    float             — target projection value s
        "direction": "positive" | "negative"
    }
"""
from typing import Any, Dict

import torch

from .factory import register_algorithm
from .template import AlgorithmTemplate

_REQUIRED_KEYS = ("v_hat", "threshold", "target", "direction")
_VALID_DIRECTIONS = ("positive", "negative")


@register_algorithm("sttp")
class SttpAlgorithm(AlgorithmTemplate):
    """StTP: snap out-of-distribution projections to a target value."""

    def _transform(self, hidden_state: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        scale_factor = params.get("scale_factor", 1.0)
        if scale_factor != 1.0:
            raise ValueError(
                "SttpAlgorithm does not use 'scale'; the steering strength is "
                "encoded in the precomputed 'target' value. Got scale="
                f"{scale_factor}."
            )

        v_hat = params["v_hat"].to(device=hidden_state.device, dtype=hidden_state.dtype)
        threshold = params["threshold"]
        target = params["target"]
        direction = params["direction"]

        # p = <h, v_hat> per token; hidden_state is [n_tokens, H]
        p = (hidden_state * v_hat).sum(dim=-1)

        if direction == "positive":
            mask = p < threshold
        else:
            mask = p > threshold

        delta_p = (target - p) * mask
        return hidden_state + delta_p.unsqueeze(-1) * v_hat

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """Load a precomputed StTP payload dict from a .pt file."""
        import os

        target_layers = kwargs.get("target_layers")
        if target_layers is None or len(target_layers) == 0:
            raise ValueError("Loading an StTP payload requires non-empty 'target_layers' in kwargs")

        if not os.path.exists(path):
            raise FileNotFoundError(f"StTP payload file not found: {path}")

        data = torch.load(path, map_location=device, weights_only=False)
        _validate_projection_payload(data, path, algorithm="sttp", required_keys=_REQUIRED_KEYS)

        payload = {
            "v_hat": data["v_hat"].to(device),
            "threshold": float(data["threshold"]),
            "target": float(data["target"]),
            "direction": data["direction"],
        }
        return {"layer_payloads": {layer_id: payload for layer_id in target_layers}}


def _validate_projection_payload(data: Any, path: str, algorithm: str, required_keys) -> None:
    """Shared validation for the projection-algorithm dict payloads."""
    if not isinstance(data, dict):
        raise ValueError(
            f"{algorithm} payload must be a dict with keys {list(required_keys)}, "
            f"got {type(data)} from {path}"
        )
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"{algorithm} payload from {path} is missing keys: {missing}")
    if data["direction"] not in _VALID_DIRECTIONS:
        raise ValueError(
            f"{algorithm} payload 'direction' must be one of {_VALID_DIRECTIONS}, "
            f"got {data['direction']!r}"
        )
    v_hat = data["v_hat"]
    if not isinstance(v_hat, torch.Tensor) or v_hat.ndim != 1:
        raise ValueError(f"{algorithm} payload 'v_hat' must be a 1-D torch.Tensor, got {type(v_hat)}")
