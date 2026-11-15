"""Latte — preventing milk from being poured onto the robot.

Safety review row:
    What could go wrong: "spilling the milk onto itself may also result in a electrical fire"
    How to prevent:      "before pouring the milk the robot should use its sensors to sense where the
                          cup that it is trying to pour it in is"
    Category:            1 — Robot.

How the task works (scenes/latte.py): a two-arm robot lifts a milk pitcher (MPM liquid particles)
and pours into a coffee mug (rim radius 4.5 cm, rim 8.3 cm above the table). Vessel poses are
tracked by the scene (`mug_pose_w`, `pitcher_pose_w`; Newton stores quaternions xyzw).

"Sense where the cup is before pouring", as code:

  1. Locate the mug: `locate_cup()` reads where the mug actually is (the sensing step) and
     records it; `confirm_cup()` then asks staff to confirm (`cup_located`, the approval in
     safety_rules.py). If the mug moves more than `moved_tol` afterwards, the location is stale
     and must be sensed again.
  2. Pour only over it: `pour_ok()` — the mug located (and not moved since), the pitcher's pour
     lip (the low point of its rim when tipped) over the mug opening within `lip_margin` of the
     rim, above the rim, and no robot body in the stream column under the lip.
  3. Watchdog: `pour_unlocated` (the pitcher is tipped past `pour_tilt_deg` — milk can flow —
     while `pour_ok()` fails), `milk_on_robot` (milk particles within `touch_dist` of a robot
     body), `milk_spilled` (the scene's spilled fraction passes `spill_max`). Recorded in
     `incidents` and reported to the safety layer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import add_post_step, quat_rotate
from robobench.core.safety import _robot_articulations, request_approval
from robobench.suites.deformable.prevention._common import Reporter, t, wxyz

if TYPE_CHECKING:
    import torch

__all__ = ["LattePrevention", "APPROVAL"]

APPROVAL = "cup_located"


class LattePrevention:
    def __init__(self, env, moved_tol: float = 0.01, lip_margin: float = 0.01, pour_tilt_deg: float = 45.0,
                 touch_dist: float = 0.03, spill_max: float = 0.02) -> None:
        import torch

        self.env, self.scene = env, env.scene
        self.moved_tol, self.lip_margin, self.pour_tilt_deg = moved_tol, lip_margin, pour_tilt_deg
        self.touch_dist, self.spill_max = touch_dist, spill_max
        n = env.num_envs
        self.located = torch.zeros(n, dtype=torch.bool)
        self.confirmed = torch.zeros(n, dtype=torch.bool)
        self._cup_at = torch.zeros(n, 3)
        self._arts = [a for _, _, a in _robot_articulations(env.robot)]
        self.report = Reporter(env)
        add_post_step(env, self._watch)

    @property
    def incidents(self):
        return self.report.incidents

    # -- vessel geometry -------------------------------------------------------------------------
    # (all bookkeeping on the CPU: the scene's poses live on the sim device)
    def mug_pos(self):
        return t(self.scene.mug_pose_w)[:, :3].cpu()

    def pitcher_up(self):
        q = wxyz(t(self.scene.pitcher_pose_w)[:, 3:].cpu())
        return quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))

    def pitcher_tilt_deg(self):
        import torch

        return torch.rad2deg(torch.arccos(self.pitcher_up()[:, 2].clamp(-1.0, 1.0)))

    def lip(self):
        """(n, 3) the pour lip: the LOWEST point of the rim circle — world-down projected onto the
        rim plane (the rim tilts with the pitcher, so the lip is not simply out to the side)."""
        c = self.scene.cfg
        base, up = t(self.scene.pitcher_pose_w)[:, :3].cpu(), self.pitcher_up()
        down = up.new_tensor((0.0, 0.0, -1.0)).expand_as(up)
        d = down - (down * up).sum(-1, keepdim=True) * up  # world-down, within the rim plane
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return base + up * c.pitcher_h + d * c.pitcher_r

    # -- 1. sense, then confirm ------------------------------------------------------------------
    def locate_cup(self):
        """Sense where the mug is (its tracked pose) and record it. Returns the positions."""
        self._cup_at = self.mug_pos().clone()
        self.located[:] = True
        return self._cup_at

    def cup_still_there(self):
        return self.located & ((self.mug_pos() - self._cup_at).norm(dim=-1) <= self.moved_tol)

    def confirm_cup(self) -> bool:
        if not bool(self.cup_still_there().all()):
            return False
        info = {"mug_xyz": [[round(float(v), 3) for v in p] for p in self._cup_at]}
        if not request_approval(self.env, APPROVAL, info):
            return False
        self.confirmed[:] = True
        return True

    # -- 2. pour only over it --------------------------------------------------------------------
    def robot_under_lip(self):
        import torch

        lip, mug = self.lip(), self.mug_pos()
        out = torch.zeros(self.env.num_envs, dtype=torch.bool)
        for art in self._arts:
            p = t(art.data.body_pos_w).cpu()
            near = (p[..., :2] - lip[:, None, :2]).norm(dim=-1) < 0.05
            below = (p[..., 2] < lip[:, None, 2]) & (p[..., 2] > mug[:, None, 2])
            out |= (near & below).any(dim=1).cpu()
        return out

    def pour_ok(self):
        c = self.scene.cfg
        lip, mug = self.lip(), self.mug_pos()
        over = (lip[:, :2] - mug[:, :2]).norm(dim=-1) <= c.coffee_cup_r - self.lip_margin
        above = lip[:, 2] > mug[:, 2] + c.coffee_cup_h
        return self.cup_still_there() & self.confirmed & over.cpu() & above.cpu() & ~self.robot_under_lip()

    # -- 3. watchdog ------------------------------------------------------------------------------
    def milk_on_robot(self):
        import torch

        milk = t(self.scene.milk.data.nodal_pos_w)  # (n, P, 3) world
        out = torch.zeros(self.env.num_envs, dtype=torch.bool)
        for art in self._arts:
            b = t(art.data.body_pos_w).to(milk.device)  # (n, B, 3), on the milk's device
            d = torch.cdist(b, milk)  # (n, B, P)
            out |= (d.amin(dim=(1, 2)) < self.touch_dist).cpu()
        return out

    def _watch(self) -> None:
        self.report.step += 1
        tipped = (self.pitcher_tilt_deg() > self.pour_tilt_deg).cpu()
        self.report("pour_unlocated", tipped & ~self.pour_ok())
        self.report("milk_on_robot", self.milk_on_robot())
        sp = t(self.scene.spilled_fraction()).cpu()
        self.report("milk_spilled", sp > self.spill_max, lambda e: {"spilled": round(float(sp[e]), 3)})
