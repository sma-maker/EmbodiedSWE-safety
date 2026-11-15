"""Small helpers shared by the deformable-suite preventions (Newton backend)."""

from __future__ import annotations


def t(x):
    """A torch tensor from a Newton/Isaac data array (`.torch` on warp arrays) or a tensor."""
    return getattr(x, "torch", x)


def wxyz(q_xyzw):
    """Newton poses store quaternions as xyzw; the shared math uses wxyz."""
    return q_xyzw[..., [3, 0, 1, 2]]


class Reporter:
    """Latching incident reporter: one record per env per onset, sent to the safety layer too."""

    def __init__(self, env) -> None:
        self.env, self.incidents, self._latched, self.step = env, [], {}, 0

    def __call__(self, kind: str, mask, info=lambda e: {}) -> None:
        import torch

        prev = self._latched.get(kind)
        prev = torch.zeros_like(mask) if prev is None else prev.to(mask.device)
        new = mask & ~prev
        self._latched[kind] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "step": self.step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"what": kind})
