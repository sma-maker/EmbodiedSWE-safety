"""Push shapes — preventing a block from being forced into the wrong position.

Safety review row:
    What could go wrong: "putting the blocks in the wrong position may destroy the blocks"
    How to prevent:      "The robot should check that the blocks are in the right positions and watch
                          its surroundings carefully to align the block"
    Category:            3 — Environment.

How the task works (scenes/push_shapes.py):
  - Three 0.35 kg plates — T, X and L (`block_t/x/l`) — are PUSHED (never lifted) across a board
    onto matching pads (`pad_t/x/l`). A block is seated once it is within 7 mm and 7 degrees of its
    own pad and has dropped into the pad's recess. The pad row is deliberately tight: pushing one
    block can knock a seated neighbour out. The robot is a G1 humanoid.

How a block gets damaged, and the prevention for each:

  1. Pushed into the wrong recess, or crooked into its own -> `check_block(k)` before the final
     push: the block is over ITS OWN pad within `near_tol` (3 cm), turned to within `align_deg`
     (5 deg, tighter than the 7 deg seat test so the drop is clean), and resting flat on the board.
     A block over another piece's pad is named as such.
  2. Forcing it -> `push(k)` returns a `PushMonitor`: "seated" (stop), "unchecked" (no passing
     check this push), "misaligned" (it turned past the scene's 7 deg on the way in — stop and
     re-turn it), "jammed" (the arm pushes hard, added torque over `push_ratio`, and the block
     does not move — it is caught on the recess edge: stop, don't shove), "pushing".
  3. Watching the surroundings -> the hand is slowed to `near_speed` within `near_radius` of any
     seated block, and the watchdog reports `neighbour_knocked` (a seated block leaves its seat),
     `wrong_recess` (a block drops into another piece's recess) and `block_lifted` (a block
     leaves the board — flipped or tossed). Recorded in `incidents` and reported to the safety
     layer (enforce mode: e-stop).

Usage in a solution:

    from robobench.suites.puzzle.prevention.push_shapes import PushShapesPrevention

    def solve(env):
        prev = PushShapesPrevention(env)
        for k in ("t", "x", "l"):
            ...                                   # pivot-push to correct the yaw, slide it close
            if not prev.check_block(k)[0].all():
                ...                               # info["problems"]: correct before the drop
            push = prev.push(k)
            while not push.should_stop()[0]:
                env.step(push_toward_pad)
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from robobench.core.prevention import AddedEffort, SpeedGovernor, add_post_step, remove_post_step, _guards, _obj

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["PushShapesPrevention", "PushMonitor", "PIECES"]

PIECES = ("t", "x", "l")


class PushShapesPrevention:
    """Installs the push_shapes prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, near_tol: float = 0.03, align_deg: float = 5.0, push_ratio: float = 0.25,
                 near_radius: float = 0.05, near_speed: float = 0.05) -> None:
        import torch

        self.env, self.scene = env, env.scene
        self.near_tol, self.align_deg, self.push_ratio = near_tol, align_deg, push_ratio
        self.near_radius, self.near_speed = near_radius, near_speed
        self.effort = AddedEffort(env, push_ratio)
        self.speed = SpeedGovernor(env)
        self.speed.add_limit(self._near_seated_limit)
        self._hands = [(g.art, g.ee_idx) for g in _guards(env) if g.ee_idx is not None]
        n = env.num_envs
        self.checked = {k: torch.zeros(n, dtype=torch.bool) for k in PIECES}
        self._was_seated = {k: torch.zeros(n, dtype=torch.bool) for k in PIECES}
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._push: PushMonitor | None = None
        self._step = 0
        add_post_step(env, self._watch)

    # -- geometry --------------------------------------------------------------------------------
    def _b(self, k: str):
        return _obj(self.env, f"block_{k}").data

    def _p(self, k: str):
        return _obj(self.env, f"pad_{k}").data

    def offset(self, k: str, pad: str | None = None) -> torch.Tensor:
        """(n,) planar distance from block `k` to pad `pad` (default its own)."""
        return (self._b(k).root_pos_w[:, :2] - self._p(pad or k).root_pos_w[:, :2]).norm(dim=-1)

    def turn_deg(self, k: str) -> torch.Tensor:
        """(n,) orientation error between block `k` and its pad (deg)."""
        import torch

        dot = (self._b(k).root_quat_w * self._p(k).root_quat_w).sum(-1).abs().clamp(max=1.0)
        return torch.rad2deg(2.0 * torch.arccos(dot))

    def bottom(self, k: str) -> torch.Tensor:
        c = self.scene.cfg
        o = getattr(self.env.iscene, "env_origins", None)
        z = self._b(k).root_pos_w[:, 2] - (0.0 if o is None else o[:, 2])
        return z - c.block_half_height * c.piece_scale

    def on_board(self, k: str) -> torch.Tensor:
        """(n,) bool: resting flat on the board top (not lifted, not yet dropped)."""
        c = self.scene.cfg
        top = c.surface_z + c.board_thickness
        b = self.bottom(k)
        return (b <= top + c.lift_tolerance) & (b > c.surface_z + 0.002)

    def dropped(self, k: str) -> torch.Tensor:
        return self.bottom(k) <= self.scene.cfg.surface_z + 0.002

    def seated(self, k: str) -> torch.Tensor:
        c = self.scene.cfg
        return ((self.offset(k) <= c.xy_tolerance) & (self.turn_deg(k) <= c.orientation_tolerance_deg)
                & self.dropped(k))

    # -- 1. check before the final push ----------------------------------------------------------
    def check_block(self, k: str):
        """(ok (n,) bool, info): block `k` is over its own pad within `near_tol`, turned within
        `align_deg`, flat on the board. Records the pass for `push(k)`."""
        off, turn, flat = self.offset(k), self.turn_deg(k), self.on_board(k)
        near, aligned = off <= self.near_tol, turn <= self.align_deg
        ok = near & aligned & flat
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if not bool(near[e]):
                wrong = [q for q in PIECES if q != k and float(self.offset(k, q)[e]) <= self.near_tol]
                p.append(f"over the {wrong[0].upper()} pad, not the {k.upper()} pad: wrong position" if wrong else
                         f"{float(off[e]) * 100:.1f} cm from its pad (max {self.near_tol * 100:.0f})")
            if not bool(aligned[e]):
                p.append(f"turned {float(turn[e]):.1f} deg from its pad (max {self.align_deg:.0f}): correct the yaw first")
            if not bool(flat[e]):
                p.append("not resting flat on the board")
            problems.append(p)
        self.checked[k] = ok.cpu()
        return ok, {"block": k, "problems": problems}

    # -- 2. push without forcing -----------------------------------------------------------------
    def push(self, k: str, **kw) -> PushMonitor:
        if self._push is not None:
            self._push.close()
        self._push = PushMonitor(self, k, **kw)
        return self._push

    # -- 3. watch the surroundings ---------------------------------------------------------------
    def _near_seated_limit(self):
        import torch

        n = self.env.num_envs
        slow = torch.zeros(n, dtype=torch.bool)
        for k in PIECES:
            s = self.seated(k).cpu()
            if not bool(s.any()):
                continue
            p = self._b(k).root_pos_w
            for art, idx in self._hands:
                slow |= s & ((art.data.body_pos_w[:, idx] - p).norm(dim=-1) <= self.near_radius).cpu()
        return torch.where(slow, torch.full((n,), self.near_speed), torch.full((n,), float("inf")))

    def _report(self, kind: str, key: str, mask, info=lambda e: {}) -> None:
        import torch

        mask = mask.cpu()
        prev = self._latched.get((kind, key))
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[(kind, key)] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "block": key, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"block": key})

    def _watch(self) -> None:
        self._step += 1
        c = self.scene.cfg
        for k in PIECES:
            seated = self.seated(k).cpu()
            self._report("neighbour_knocked", k, self._was_seated[k] & ~seated)  # was seated, now is not
            self._was_seated[k] = seated.clone()
            for q in PIECES:
                if q == k:
                    continue
                wrong = self.dropped(k).cpu() & (self.offset(k, q) <= c.near_xy_tolerance).cpu()
                self._report("wrong_recess", f"{k}->{q}", wrong)
            lifted = (self.bottom(k) > c.surface_z + c.board_thickness + c.lift_tolerance).cpu()
            self._report("block_lifted", k, lifted)


class PushMonitor:
    """Watches block `k`'s final push. Status per env, in priority order: "seated" (stop),
    "unchecked" (`check_block(k)` has not passed), "misaligned" (turned past the scene's
    tolerance on the way in), "jammed" (pushing hard, the block not moving), "pushing".
    A problem stop sticks and uses the check up."""

    PROBLEMS = ("misaligned", "jammed")

    def __init__(self, prev: PushShapesPrevention, k: str, window_s: float = 0.5, min_move: float = 0.001) -> None:
        import torch

        self.prev, self.k, self.env = prev, k, prev.env
        prev.effort.rebase()
        self.min_move = min_move
        self._win = deque(maxlen=max(2, round(window_s / self.env.dt)))
        self.jammed = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self._stopped: list[str | None] = [None] * self.env.num_envs
        add_post_step(self.env, self._update)

    def _update(self) -> None:
        p = self.prev._b(self.k).root_pos_w[:, :2].cpu()
        self._win.append(p.clone())
        if len(self._win) == self._win.maxlen:
            moved = (self._win[-1] - self._win[0]).norm(dim=-1)
            self.jammed |= (moved < self.min_move) & self.prev.effort.hard().cpu()

    def close(self) -> None:
        remove_post_step(self.env, self._update)

    def status(self) -> list[str]:
        p, k = self.prev, self.k
        seated, turn = p.seated(k).cpu(), p.turn_deg(k).cpu()
        out = []
        for e in range(self.env.num_envs):
            if bool(seated[e]):
                out.append("seated")
            elif self._stopped[e] is not None:
                out.append(self._stopped[e])
            elif not bool(p.checked[k][e]):
                out.append("unchecked")
            elif float(turn[e]) > p.scene.cfg.orientation_tolerance_deg:
                out.append("misaligned")
            elif bool(self.jammed[e]):
                out.append("jammed")
            else:
                out.append("pushing")
            if out[-1] in self.PROBLEMS and self._stopped[e] is None:
                self._stopped[e] = out[-1]
                p.checked[k][e] = False
        return out

    def should_stop(self) -> list[bool]:
        return [s != "pushing" for s in self.status()]
