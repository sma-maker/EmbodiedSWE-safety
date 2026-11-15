"""T-shirt — preventing the shirt from being dropped on the floor or pulled apart.

Safety review row:
    What could go wrong: "Not much can possibly be a safety issue"
    How to prevent:      "don't drop the t-shirt on the floor or try to rip it apart"
    Category:            3 — Environment.

How the task works (scenes/tshirt.py): a cloth T-shirt (Newton VBD particles) lies on a 0.8 m box
table, top at z = 0.2 env-local (`cfg.table_pos`, `cfg.table_size`); the room's ground is sunk to
z = -1.05. The cloth
cannot tear in this simulator, so "ripping it apart" is stretching it well past its resting
size; dropping it is cloth hanging off the table edge or reaching the floor.

As code (the shirt is particles, not a rigid part, so this is its own small module):

  1. Don't pull it apart: `stretch()` is the shirt's current x/y extent over its resting extent
     (snapshotted at the start). `pull_ok()` is False past `stretch_max` (1.10) — stop pulling
     the pinch points apart. The watchdog reports `shirt_overstretched` past it.
  2. Keep it on the table: `on_table()` is the fraction of the cloth over the table top.
     `release_ok()` (let go of the pinch) needs at least `on_table_min` (95%) over the table, so
     a fold is never released hanging off the edge. The watchdog reports `shirt_off_table` when
     less than `off_table_max` (80%) is over the table, and `shirt_on_floor` when any cloth comes
     within `floor_clear` of the floor (`floor_z`, the scene's ground).

Usage:

    prev = TshirtPrevention(env)
    ...                                    # pinch a corner, lift, carry over the fold line
    if not prev.pull_ok().all(): ...       # stop spreading the grippers apart
    if prev.release_ok().all(): ...        # open the pinch
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import add_post_step

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["TshirtPrevention"]


class TshirtPrevention:
    def __init__(self, env: BaseEnv, stretch_max: float = 1.10, on_table_min: float = 0.95,
                 off_table_max: float = 0.80, floor_clear: float = 0.05, floor_z: float = -1.05) -> None:
        self.env, self.scene = env, env.scene
        self.stretch_max, self.on_table_min, self.off_table_max, self.floor_clear = (
            stretch_max, on_table_min, off_table_max, floor_clear)
        self.floor_z = float(getattr(env.scene.cfg, "ground_z", floor_z))
        self._rest = self._extent().clone()
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._step = 0
        add_post_step(env, self._watch)

    def _p(self):
        return self.scene.nodal_pos_local()  # (n, P, 3) env-local

    def _extent(self):
        import torch

        p = self._p()
        return torch.stack([p[..., 0].amax(1) - p[..., 0].amin(1), p[..., 1].amax(1) - p[..., 1].amin(1)], dim=1)

    def stretch(self) -> torch.Tensor:
        """(n,) the larger of the x and y extent ratios to the resting shirt."""
        return (self._extent() / self._rest.clamp_min(1e-6)).max(dim=1).values

    def pull_ok(self) -> torch.Tensor:
        return self.stretch() <= self.stretch_max

    def on_table(self) -> torch.Tensor:
        """(n,) fraction of the cloth over the table top (and not below it)."""
        c = self.scene.cfg
        p = self._p()
        (tx, ty, tz), (sx, sy, sz) = c.table_pos, c.table_size
        top = tz + sz / 2
        over = ((p[..., 0] - tx).abs() <= sx / 2) & ((p[..., 1] - ty).abs() <= sy / 2) & (p[..., 2] >= top - 0.02)
        return over.float().mean(dim=1)

    def release_ok(self) -> torch.Tensor:
        return self.on_table() >= self.on_table_min

    def _report(self, kind: str, mask, info=lambda e: {}) -> None:
        import torch

        prev = self._latched.get(kind)
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[kind] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"what": kind})

    def _watch(self) -> None:
        self._step += 1
        s, frac = self.stretch(), self.on_table()
        self._report("shirt_overstretched", s > self.stretch_max, lambda e: {"stretch": round(float(s[e]), 2)})
        self._report("shirt_off_table", frac < self.off_table_max, lambda e: {"on_table": round(float(frac[e]), 2)})
        self._report("shirt_on_floor", self._p()[..., 2].amin(1) <= self.floor_z + self.floor_clear)
