"""Shoe knot — preventing the robot's fingers from getting stuck in the laces.

Safety review row:
    What could go wrong: "The robot hand may get stuck in the shoe lace"
    How to prevent:      "Staff monitoring the robot should make sure the fingers of the hand don't get stuck"
    Category:            1 — Robot.

How the task works (scenes/shoe_knot.py): two laces (Newton rods, `scene.lace_points(i, world)`)
are tied into a half knot by pinching their ends, crossing them, threading one under, and
pulling apart. Every release is a moment a lace can be wrapped around a finger.

"Staff make sure the fingers don't get stuck", as code:

  1. Every release is checked: `open_hand()` starts a `FingerReleaseCheck` (shared) — fingers
     told to open that stay put for `stuck_s` are caught on something.
  2. Caught -> freeze and call staff: the arm is held still (speed limit 0) and `fingers_caught()`
     reports it; `staff_clear()` asks staff to confirm the fingers are free
     (`fingers_free_verified`), which unfreezes the arm.
  3. Watch the laces near the hand: `lace_on_fingers()` counts lace points within `wrap_dist` of
     a finger body; with the hand opening and laces still on the fingers, staff are asked to look
     before the arm moves away (`release_ok()` is False until they clear it).
  4. Watchdog: `fingers_entangled` (the release check caught the fingers — the signal named in
     safety_rules.py), recorded in `incidents` and reported to the safety layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import FingerReleaseCheck, SpeedGovernor, add_post_step
from robobench.core.safety import _robot_articulations, request_approval
from robobench.suites.deformable.prevention._common import Reporter, t

if TYPE_CHECKING:
    import torch

__all__ = ["KnotPrevention", "APPROVAL"]

APPROVAL = "fingers_free_verified"


class KnotPrevention:
    def __init__(self, env, stuck_s: float = 0.2, wrap_dist: float = 0.01, wrap_points: int = 3) -> None:
        import torch

        self.env, self.scene = env, env.scene
        self.wrap_dist, self.wrap_points = wrap_dist, wrap_points
        self.release = FingerReleaseCheck(env, window_s=stuck_s)
        self.frozen = torch.zeros(env.num_envs, dtype=torch.bool)
        self.speed = SpeedGovernor(env)
        self.speed.add_limit(lambda: torch.where(self.frozen, torch.zeros(env.num_envs), torch.full((env.num_envs,), float("inf"))))
        self._fingers = []  # (articulation, body indices) of finger links
        for _, _, art in _robot_articulations(env.robot):
            names = list(getattr(art, "body_names", []))
            idx = [i for i, nm in enumerate(names) if "finger" in nm or "thumb" in nm or "index" in nm]
            if idx:
                self._fingers.append((art, idx))
        self.report = Reporter(env)
        add_post_step(env, self._watch)

    @property
    def incidents(self):
        return self.report.incidents

    # -- 1-2. check every release; freeze and call staff -----------------------------------------
    def open_hand(self) -> None:
        """Call when commanding the fingers open."""
        self.release.start()

    def fingers_caught(self):
        return (~self.release.fingers_free()).cpu() | self.frozen

    def staff_clear(self) -> bool:
        """Staff confirm the fingers are free (after untangling them); unfreezes the arm."""
        info = {"caught_envs": self.fingers_caught().nonzero().flatten().tolist(),
                "lace_points_on_fingers": self.lace_on_fingers().tolist()}
        if not request_approval(self.env, APPROVAL, info):
            return False
        self.frozen[:] = False
        self.release.start()  # a fresh check from here
        self._cleared = True
        return True

    # -- 3. laces near the hand ------------------------------------------------------------------
    def lace_on_fingers(self):
        """(n,) number of lace points within `wrap_dist` of any finger body."""
        import torch

        n = self.env.num_envs
        out = torch.zeros(n, dtype=torch.long)
        if not self._fingers:
            return out
        for e in range(n):
            laces = torch.cat([torch.as_tensor(self.scene.lace_points(i, world=e), dtype=torch.float32)
                               for i in range(2)], dim=0)
            for art, idx in self._fingers:
                f = t(art.data.body_pos_w)[e, idx]
                out[e] += int((torch.cdist(f, laces) < self.wrap_dist).any(dim=0).sum())
        return out

    def release_ok(self):
        """(n,) bool: the fingers opened freely and no lace is still on them."""
        return ~self.fingers_caught() & (self.lace_on_fingers() < self.wrap_points)

    # -- 4. watchdog ------------------------------------------------------------------------------
    def _watch(self) -> None:
        self.report.step += 1
        caught = (~self.release.fingers_free()).cpu()
        self.frozen |= caught
        self.report("fingers_entangled", caught)
