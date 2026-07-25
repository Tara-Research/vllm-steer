# SPDX-License-Identifier: Apache-2.0
"""Steering to Mirror Projection (StMP).

Mirror-based steering along a learned trait direction. For tokens whose
projection p = <h, v_hat> falls on the "wrong side" of a gating threshold,
the projection is reflected across the threshold, interpolated by a
coefficient, and clamped at a cap:

    p_mirror = 2 * threshold - p
    p'       = p + coefficient * (p_mirror - p)
    p'       = min(p', cap)
    h'       = h + (p' - p) * v_hat      (only where the gate mask is true)

Coefficient semantics: 0.0 = no change, 0.5 = halfway to the mirror image,
1.0 = full reflection, >1.0 = overshoot.

Direction semantics:
  * direction == "positive": mirror tokens with p < threshold upward
  * direction == "negative": mirror tokens with p > threshold downward

The per-layer statistics (normalized direction v_hat, gating threshold, cap)
are precomputed by the client from the positive/negative training activation
distributions and shipped as a small dict payload — the raw [L, N, H]
embedding tensors never enter vLLM.

Expected .pt payload format (dict):
    {
        "v_hat":       torch.Tensor [H]  — L2-normalized steering direction
        "threshold":   float             — gating/mirror threshold in projection space
        "coefficient": float             — mirror interpolation coefficient
        "cap":         float             — upper clamp for p' (mu_pos + 40 * std_pos)
        "direction":   "positive" | "negative"
    }
"""
from typing import Any, Dict

import torch

from .factory import register_algorithm
from .sttp import _validate_projection_payload
from .template import AlgorithmTemplate

_REQUIRED_KEYS = ("v_hat", "threshold", "coefficient", "cap", "direction")


@register_algorithm("stmp")
class StmpAlgorithm(AlgorithmTemplate):
    """StMP: reflect out-of-distribution projections across the threshold."""

    def _transform(self, hidden_state: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        scale_factor = params.get("scale_factor", 1.0)
        if scale_factor != 1.0:
            raise ValueError(
                "StmpAlgorithm does not use 'scale'; the steering strength is "
                "encoded in the payload 'coefficient'. Got scale="
                f"{scale_factor}."
            )

        v_hat = params["v_hat"].to(device=hidden_state.device, dtype=hidden_state.dtype)
        threshold = params["threshold"]
        coefficient = params["coefficient"]
        cap = params["cap"]
        direction = params["direction"]

        # p = <h, v_hat> per token; hidden_state is [n_tokens, H]
        p = (hidden_state * v_hat).sum(dim=-1)

        if direction == "positive":
            mask = p < threshold
        else:
            mask = p > threshold

        mirror_target = 2.0 * threshold - p
        p_prime = p + coefficient * (mirror_target - p)
        p_prime = torch.clamp(p_prime, max=cap)
        delta_p = (p_prime - p) * mask
        delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)

        return hidden_state + delta_p.unsqueeze(-1) * v_hat

    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> Dict[str, Any]:
        """Load a precomputed StMP payload dict from a .pt file."""
        import os

        target_layers = kwargs.get("target_layers")
        if target_layers is None or len(target_layers) == 0:
            raise ValueError("Loading an StMP payload requires non-empty 'target_layers' in kwargs")

        if not os.path.exists(path):
            raise FileNotFoundError(f"StMP payload file not found: {path}")

        data = torch.load(path, map_location=device, weights_only=False)
        _validate_projection_payload(data, path, algorithm="stmp", required_keys=_REQUIRED_KEYS)

        payload = {
            "v_hat": data["v_hat"].to(device),
            "threshold": float(data["threshold"]),
            "coefficient": float(data["coefficient"]),
            "cap": float(data["cap"]),
            "direction": data["direction"],
        }
        return {"layer_payloads": {layer_id: payload for layer_id in target_layers}}
