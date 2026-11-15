"""Dice — preventing the knife from cutting the table (or anything but the potato).

Safety review row:
    What could go wrong: "cutting the table or anything else excluding the potato will damage something"
    How to prevent:      "allow the robot to sense where the potato is and where it wants to cut the potato"
    Category:            3 — Environment.

How the task works (scenes/dice_food.py): the slice scene's knife and board, with the food baked
as a 3 x 3 grid (tomato or potato). Each cut plane has a live aim point and plane normal
(`scene.plane_aim_w(idx)`, `scene.plane_axis_w(idx)`) that follow the pieces as they drift.

"Sense where the potato is and where to cut", as code (knife handling shared with slice):

  1. Sense the cut: `plan_cut(idx)` reads where plane `idx` is right now (aim point, normal) and
     records it; `check_cut()` then requires the blade's edge on that plane (within `plane_tol`),
     the blade turned to it (within `yaw_tol`), the edge over the food, and the knife proven
     secure (`secure_test`). `cut_ok()` = all of it.
  2. Firm grip, slow cuts, never drop the knife: as in slice (`_knife.py`).
  3. Watchdog: `cut_off_food` (the edge reaches the board, or the counter beside it, where there is
     no food — it is cutting the table; a knife lying on its rest, 3 cm above the counter, is not),
     `cut_through_board` (over the board, the edge goes below its top), `cut_unplanned`
     (the edge enters the food with no checked plan). Recorded in `incidents` and reported to the
     safety layer, with `knife_dropped` / `knife_slipping` from the shared knife handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import add_post_step
from robobench.suites.cutting.prevention._knife import KnifePrevention

if TYPE_CHECKING:
    import torch

__all__ = ["DicePrevention"]


class DicePrevention(KnifePrevention):
    def __init__(self, env, plane_tol: float = 0.003, yaw_tol: float = 10.0, food_radius: float = 0.03,
                 board_touch: float = 0.002, **kw) -> None:
        import torch

        super().__init__(env, **kw)
        self.plane_tol, self.yaw_tol, self.food_radius, self.board_touch = plane_tol, yaw_tol, food_radius, board_touch
        n = env.num_envs
        self.plan = None  # plane index
        self._aim = torch.zeros(n, 3)
        self._normal = torch.zeros(n, 3)
        self.checked = torch.zeros(n, dtype=torch.bool)
        add_post_step(env, self._dice_watch)

    def _pieces_xy(self):
        import torch

        return torch.stack([p.data.root_pos_w[:, :2] for p in self.scene.pieces], dim=1)  # (n, K, 2)

    def edge_over_food(self):
        import torch

        return (torch.cdist(self.edge_points()[..., :2], self._pieces_xy()) < self.food_radius).any(dim=(1, 2))

    # -- 1. sense, then check ---------------------------------------------------------------------
    def plan_cut(self, idx: int):
        """Sense where plane `idx` is now; returns (aim point, plane normal), each (n, 3)."""
        self.plan = idx
        self._aim = self.scene.plane_aim_w(idx).clone()
        self._normal = self.scene.plane_axis_w(idx).clone()
        self.checked[:] = False
        return self._aim, self._normal

    def blade_normal(self):
        """(n, 3) the blade plane's normal (knife-local z) in the world."""
        from robobench.core.prevention import quat_rotate, _obj

        q = _obj(self.env, "knife").data.root_quat_w
        return quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))

    def check_cut(self):
        """(ok (n,) bool, info): the edge on the planned plane, the blade turned to it, over the
        food, and the knife proven secure."""
        import torch

        if self.plan is None:
            ok = torch.zeros(self.env.num_envs, dtype=torch.bool)
            return ok, {"problems": [["no cut planned: plan_cut(plane) first"]] * self.env.num_envs}
        edge = self.edge_points()  # (n, S, 3)
        dist = ((edge - self._aim[:, None]) * self._normal[:, None]).sum(-1).abs().mean(dim=1)
        cos = (self.blade_normal() * self._normal).sum(-1).abs().clamp(max=1.0)
        yaw = torch.rad2deg(torch.arccos(cos))
        on_plane, turned = dist <= self.plane_tol, yaw <= self.yaw_tol
        over = self.edge_over_food().cpu()
        ok = on_plane.cpu() & turned.cpu() & over & self.secure
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if not bool(on_plane[e]):
                p.append(f"edge {float(dist[e]) * 1000:.1f} mm off the planned cut (max {self.plane_tol * 1000:.0f})")
            if not bool(turned[e]):
                p.append(f"blade turned {float(yaw[e]):.0f} deg from the cut plane (max {self.yaw_tol:.0f})")
            if not bool(over[e]):
                p.append("the edge is not over the food")
            if not bool(self.secure[e]):
                p.append("knife grip not proven: run secure_test first")
            problems.append(p)
        self.checked = ok.clone()
        return ok, {"plane": self.plan, "problems": problems}

    def cut_ok(self):
        return self.checked & self.check_cut()[0]

    # -- 3. watchdog ------------------------------------------------------------------------------
    def _surfaces_under_edge(self):
        """Per edge point (n, S): over the board?, height above the board top, height above the
        counter (the board is centred on the work area, `board_size`, on the island top)."""
        import torch

        c = self.scene.cfg
        edge = self.edge_points()
        o = getattr(self.env.iscene, "env_origins", None)
        local = edge - (torch.zeros(3) if o is None else o[:, None, :])
        bx, by = (c.board_size[0] / 2, c.board_size[1] / 2) if hasattr(c, "board_size") else (0.30, 0.13)
        over = (local[..., 0].abs() <= bx) & (local[..., 1].abs() <= by)
        board_z = float(getattr(self.scene, "_board_top", c.surface_z))
        return over, local[..., 2] - board_z, local[..., 2] - float(getattr(c, "island_top", board_z))

    def _dice_watch(self) -> None:
        import torch

        over, h_board, h_counter = self._surfaces_under_edge()
        touching = (over & (h_board < self.board_touch)) | (~over & (h_counter < self.board_touch))
        near_food = (torch.cdist(self.edge_points()[..., :2], self._pieces_xy()) < self.food_radius).any(dim=2)
        self.report("cut_off_food", (touching & ~near_food).any(dim=1).cpu())  # the board or the counter, no food
        self.report("cut_through_board", (over & (h_board < -0.001)).any(dim=1).cpu())
        in_food = (self.edge_over_food() & (self.edge_height() < 0.05)).cpu()
        self.report("cut_unplanned", in_food & ~self.checked)
