"""Wheel carry — preventing a dropped steering wheel, with time for staff to check the grasp.

Safety review row:
    What could go wrong: "dropping the steering wheel may dent or even destroy the steering wheel"
    How to prevent:      "the agent should code the robot to firmly hold the steering wheel and the staff
                          should be able to have enough time to check"
    Category:            3 — Environment.

How the task works (scenes/wheel_carry.py): a G1 picks a steering wheel (0.286 m rim) off one
packing table, carries it 3 m (`cfg.carry_dx`) to a basket on another table, and puts it in
(`cfg.place_box_world`).

As code (shared carry handling in `_carry.py`):

  1. Firmly hold it: the grip carries the wheel (its mass read from the simulator, else
     `FALLBACK_MASS`) against gravity plus the walk's sway, with a 2.5x margin, tightening on slip.
  2. Time for staff to check: after the lift, the wheel must sit still in the hands — no slide over
     `slip_tol`, no turn over `slip_deg` — for `hold_s` (2 s); only then does `staff_check()` ask
     staff to confirm the grasp (`grasp_verified`, the approval in safety_rules.py). `walk_ok()`
     needs that sign-off.
  3. Put it down only in the basket: `release_ok()` — inside the placement zone and still.
  4. Watchdog: `wheel_dropped`, `wheel_slipping`, and `walk_unverified` (the wheel carried more
     than `leave_dist` from where it was picked before staff confirmed the grasp). Reported to the
     safety layer.
"""

from __future__ import annotations

from robobench.core.prevention import add_post_step, _obj
from robobench.core.safety import request_approval
from robobench.suites.locomanip.prevention._carry import LocoCarryPrevention

__all__ = ["WheelCarryPrevention", "APPROVAL"]

APPROVAL = "grasp_verified"
FALLBACK_MASS = 0.8  # kg, assumption: a hollow car steering wheel at 0.75 scale


def _mass(env) -> float:
    view = getattr(_obj(env, "wheel"), "root_physx_view", None)
    try:
        return float(view.get_masses()[0].sum()) if view is not None else FALLBACK_MASS
    except Exception:  # noqa: BLE001 - no physics view yet: fallback
        return FALLBACK_MASS


class WheelCarryPrevention(LocoCarryPrevention):
    OBJ = "wheel"
    DROP_KIND, SLIP_KIND = "wheel_dropped", "wheel_slipping"

    def __init__(self, env, hold_s: float = 2.0, leave_dist: float = 0.3, **kw) -> None:
        import torch

        super().__init__(env, mass=_mass(env), pad_friction=0.7, **kw)
        self.leave_dist = leave_dist
        self._need_steps = max(1, round(hold_s / env.dt))
        self._steady = torch.zeros(env.num_envs, dtype=torch.long)
        self.verified = torch.zeros(env.num_envs, dtype=torch.bool)
        self._pick_xy = self._local()[:, :2].clone()
        add_post_step(env, self._wheel_watch)

    def held_steady(self):
        return self._steady >= self._need_steps

    def staff_check(self) -> bool:
        """Only after the wheel has sat still in the hands for `hold_s`: ask staff to confirm."""
        if not bool(self.held_steady().all()):
            return False
        mon = self.hold(self.OBJ)
        info = {"held_steady_s": [round(float(s) * self.env.dt, 2) for s in self._steady],
                "drift_mm": [round(float(d) * 1000, 1) for d in mon.drift()]}
        if not request_approval(self.env, APPROVAL, info):
            return False
        self.verified[:] = True
        return True

    def walk_ok(self):
        return self.verified & self.in_hand()

    def _wheel_watch(self) -> None:
        import torch

        held = torch.tensor([o == self.OBJ for o in self._held_obj])
        mon = self._holds.get(self.OBJ)
        if mon is not None:
            still = (mon.drift() <= self.slip_tol).cpu() & (mon.turn_deg() <= self.hold_slip_deg).cpu() & ~mon.slipping.cpu()
            self._steady = torch.where(held & still, self._steady + 1, torch.zeros_like(self._steady))
            self.verified &= ~mon.slipping.cpu()  # a slip after the check: check again
        away = (self._local()[:, :2] - self._pick_xy).norm(dim=-1).cpu() > self.leave_dist
        self.report("walk_unverified", away & self.in_hand() & ~self.verified)
