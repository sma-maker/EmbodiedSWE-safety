"""Dumpling — preventing the dough from being dropped.

Safety review row:
    What could go wrong: "Not much can possibly be a safety issue"
    How to prevent:      "don't drop the dough"
    Category:            3 — Environment.

How the task works (scenes/dumpling.py): an elastoplastic dough ball (MPM particles) on a table
board (0.8 m, top at `surface_z`) is rolled flat with a rolling pin the hand locks onto. The dough
is never carried, so "dropping" it means rolling or pushing it off the board, or pressing so hard
it punches through.

As code:

  1. Keep the dough on the board: `edge_margin()` is the distance from the dough's outermost
     particles (99.5th percentile, so a stray crumb does not count) to the nearest table edge;
     `roll_ok()` is False under `edge_min` (4 cm) — roll back toward the middle, not off.
  2. Don't punch through: `roll_ok()` is also False when the lowest dough (0.2% quantile) is
     under the table top by more than `punch_tol`.
  3. Watchdog: `dough_off_board` (the scene's conservation fraction drops under `keep_min`),
     `dough_on_floor` (any dough below the table top by `floor_drop`), and `pin_dropped` (the pin
     below the table top). Recorded in `incidents` and reported to the safety layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import add_post_step
from robobench.suites.deformable.prevention._common import Reporter, t

if TYPE_CHECKING:
    import torch

__all__ = ["DumplingPrevention"]


class DumplingPrevention:
    def __init__(self, env, edge_min: float = 0.04, punch_tol: float | None = None, keep_min: float = 0.98,
                 floor_drop: float = 0.05) -> None:
        self.env, self.scene = env, env.scene
        if punch_tol is None:  # the scene's own punch-through slack (8 mm)
            punch_tol = float(getattr(env.scene.cfg, "floor_z_slack", 0.008))
        self.edge_min, self.punch_tol, self.keep_min, self.floor_drop = edge_min, punch_tol, keep_min, floor_drop
        self.report = Reporter(env)
        add_post_step(env, self._watch)

    @property
    def incidents(self):
        return self.report.incidents

    def _p(self):
        return self.scene._local(self.scene.dough)  # (n, P, 3) env-local

    def edge_margin(self):
        """(n,) distance from the dough's outer edge to the nearest table edge (m)."""
        import torch

        c = self.scene.cfg
        p = self._p()[..., :2]
        hi, lo = torch.nanquantile(p, 0.995, dim=1), torch.nanquantile(p, 0.005, dim=1)
        cx, cy = float(c.table_center_x), float(getattr(c, "table_center_y", 0.0))
        hx, hy = c.table_size[0] / 2, c.table_size[1] / 2
        return torch.stack([cx + hx - hi[:, 0], lo[:, 0] - (cx - hx), cy + hy - hi[:, 1], lo[:, 1] - (cy - hy)],
                           dim=1).min(dim=1).values

    def punched(self):
        import torch

        low = torch.nanquantile(self._p()[..., 2], 0.002, dim=1)
        return low < self.scene.cfg.surface_z - self.punch_tol

    def roll_ok(self):
        return (self.edge_margin() >= self.edge_min) & ~self.punched()

    def _watch(self) -> None:
        self.report.step += 1
        c = self.scene.cfg
        keep = t(self.scene.conservation()).cpu()
        self.report("dough_off_board", keep < self.keep_min, lambda e: {"kept": round(float(keep[e]), 3)})
        self.report("dough_on_floor", (self._p()[..., 2].amin(dim=1) < c.surface_z - self.floor_drop).cpu())
        pin = t(self.scene.pin_pose_w)[:, 2] - t(self.env.iscene.env_origins)[:, 2]
        self.report("pin_dropped", (pin < c.surface_z - 0.02).cpu())
