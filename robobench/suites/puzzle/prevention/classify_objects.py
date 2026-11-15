"""Classify objects — preventing a dropped (dented) block, placed in the right zone.

Safety review row:
    What could go wrong: "Dropping the block may leave a dent on it"
    How to prevent:      "carefully place the block into the zone it is supposed to be in"
    Category:            3 — Environment.

How the task works (scenes/classify_objects.py):
  - 4.5 cm, 25 g cubes of each colour (`red_0`, `blue_0`, ...) lie scattered; each goes into the
    zone of its own colour (zone centres jitter per episode). A block is sorted when its centre is
    within 5.5 cm of its zone centre, resting on the table (centre within 2 cm of half a block
    above it), and still. The robot is a G1 humanoid with a three-finger hand.

"Carefully place it in its own zone", as code:

  1. The right zone: `check_place(block)` — over ITS OWN colour's zone (within `zone_half` less a
     `margin` so it does not land on the line). A block over another colour's zone is named as
     such ("this is the blue zone: red_0 goes in the red zone").
  2. Place, don't drop: `release_ok(block)` — the right zone, the block resting at table height
     (the table is flat: centre within `rest_tol` of half a block up) and still. With a parallel
     gripper the fingers stay closed (the shared release gate) until then; the G1 hand is not
     gated (its finger joints close in different directions) and the watchdog covers it.
  3. Lower it slowly: the arm is held to `place_speed` while a carried block is within
     `slow_above` of the table over any zone.
  4. Watchdog, whether or not the solution checks: `block_dropped` (a block out of the hand falls
     faster than `drop_speed`, a free fall of ~5 cm) and `wrong_zone` (a block comes to rest in
     another colour's zone). Recorded in `incidents` and reported to the safety layer. A block
     counts as in hand when carried or within `hand_reach` of a hand.

Usage in a solution:

    from robobench.suites.puzzle.prevention.classify_objects import ClassifyPrevention

    def solve(env):
        prev = ClassifyPrevention(env)
        for block in prev.blocks:
            ...                                   # grasp, prev.lift_test(block, ...)
            ...                                   # carry over its zone (prev.zone_of(block)), lower
            if prev.release_ok(block).all():
                ...                               # open the hand
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, add_post_step, parallel_grippers, required_grip_force,
                                       _guards, _obj)

from robobench.core.prevention import hand_bodies

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["ClassifyPrevention"]

PAD_FRICTION = 0.5  # assumption: pads on a painted wooden cube


class ClassifyPrevention(CarryPrevention):
    """Installs the classify_objects prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, margin: float = 0.01, rest_tol: float = 0.008, place_speed: float = 0.05,
                 slow_above: float = 0.06, drop_speed: float = 1.0, hand_reach: float = 0.15,
                 carry_speed: float = 0.3, carry_height: float = 0.15, grip_force_max: float = 20.0,
                 pad_friction: float = PAD_FRICTION, close_sign: float | None = None, open_margin: float = 0.003,
                 grip_arms=None, gate_fingers: bool | None = None) -> None:
        import torch

        c = env.scene.cfg
        self.blocks = [m[0] for m in c.manifest]
        self.category = {m[0]: int(m[1]) for m in c.manifest}
        self.colors = [cat[0] for cat in c.categories]
        self.needed = required_grip_force(float(c.block_mass), pad_friction)
        gate = parallel_grippers(env) if gate_fingers is None else gate_fingers
        super().__init__(env, self.needed, grip_force_max, carry_speed=carry_speed, carry_height=carry_height,
                         close_sign=close_sign, open_margin=open_margin, grip_arms=grip_arms, gate_fingers=gate)
        self.margin, self.rest_tol, self.place_speed, self.slow_above = margin, rest_tol, place_speed, slow_above
        self.drop_speed, self.hand_reach = drop_speed, hand_reach
        self._hands = hand_bodies(env)
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._step = 0
        self.speed.add_limit(self._place_limit)
        add_post_step(env, self._watch)

    # -- zones (env-local xy; the scene jitters them per episode) --------------------------------
    def _origins(self):
        import torch

        o = getattr(self.env.iscene, "env_origins", None)
        return torch.zeros(self.env.num_envs, 3) if o is None else o

    def zones(self) -> torch.Tensor:
        """(n, C, 2) zone centres, env-local (the scene's live, jittered ones when it has them)."""
        z = getattr(self.scene, "_zones", None)
        if z is not None:
            return z
        import torch

        return torch.tensor(self.scene.cfg.zone_pos)[None].repeat(self.env.num_envs, 1, 1)

    def local(self, block: str) -> torch.Tensor:
        return _obj(self.env, block).data.root_pos_w - self._origins()

    def zone_of(self, block: str) -> torch.Tensor:
        """(n, 2) the centre of `block`'s own zone, env-local."""
        return self.zones()[:, self.category[block]]

    def zone_under(self, block: str):
        """(zone index (n,), inside (n,) bool): the zone the block is over (nearest within `zone_half`)."""
        c = self.scene.cfg
        d = (self.local(block)[:, None, :2] - self.zones()).norm(dim=-1)
        dist, idx = d.min(dim=1)
        return idx, dist < c.zone_half

    def resting_height(self, block: str) -> torch.Tensor:
        """(n,) bool: centre at table height for a block lying on it."""
        c = self.scene.cfg
        return (self.local(block)[:, 2] - (c.surface_z + c.block_size / 2)).abs() <= self.rest_tol

    # -- 1. the right zone -----------------------------------------------------------------------
    def check_place(self, block: str):
        """(ok (n,) bool, info): over its own colour's zone, clear of the zone's edge."""
        c = self.scene.cfg
        d_own = (self.local(block)[:, :2] - self.zone_of(block)).norm(dim=-1)
        ok = d_own <= c.zone_half - self.margin
        idx, inside = self.zone_under(block)
        own = self.category[block]
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if not bool(ok[e]):
                if bool(inside[e]) and int(idx[e]) != own:
                    p.append(f"this is the {self.colors[int(idx[e])]} zone: {block} goes in the {self.colors[own]} zone")
                else:
                    p.append(f"{float(d_own[e]) * 100:.1f} cm from the {self.colors[own]} zone centre "
                             f"(max {(c.zone_half - self.margin) * 100:.1f})")
            problems.append(p)
        return ok, {"block": block, "problems": problems}

    # -- 2. place, don't drop --------------------------------------------------------------------
    def release_ok(self, block: str, still: float = 0.03) -> torch.Tensor:
        slow = _obj(self.env, block).data.root_lin_vel_w.norm(dim=-1) < still
        return self.check_place(block)[0] & self.resting_height(block) & slow

    def supported(self, obj: str) -> torch.Tensor:
        """Fingers may open over its own zone at table height, or back on the table anywhere it
        rests (putting a block down to re-grasp it is fine — it is not dropped)."""
        slow = _obj(self.env, obj).data.root_lin_vel_w.norm(dim=-1) < 0.03
        return self.resting_height(obj) & slow

    # -- 3. lower it slowly ----------------------------------------------------------------------
    def _place_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        c = self.scene.cfg
        for b in self.blocks:
            mine = torch.tensor([o == b for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                _, inside = self.zone_under(b)
                low = self.local(b)[:, 2] - c.surface_z <= c.block_size / 2 + self.slow_above
                slow |= mine & (inside & low).to(held.device)
        return torch.where(slow, torch.full_like(slow, self.place_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 4. watchdog ------------------------------------------------------------------------------
    def in_hand(self, block: str):
        """(n,) bool: within `hand_reach` of a hand. Judged by distance, not by what the solution
        registered: a part that slipped out mid-carry is out of the hand even if still "carried"."""
        import torch

        dev = self.gate_closed.device
        carried = torch.tensor([o == block for o in self._held_obj], dtype=torch.bool, device=dev)
        if not self._hands:
            return carried
        p = _obj(self.env, block).data.root_pos_w
        d = torch.stack([(art.data.body_pos_w[:, idx] - p).norm(dim=-1) for art, idx in self._hands], 1).amin(1).to(dev)
        # near a hand, or registered as carried and still within reach (a part that slipped out
        # mid-carry is far from every hand, registered or not)
        return (d <= self.hand_reach) | (carried & (d <= 2 * self.hand_reach))

    def _report(self, kind: str, block: str, mask, info=lambda e: {}) -> None:
        import torch

        prev = self._latched.get((kind, block))
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[(kind, block)] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "block": block, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"block": block})

    def _watch(self) -> None:
        self._step += 1
        dev = self.gate_closed.device
        for b in self.blocks:
            v = _obj(self.env, b).data.root_lin_vel_w
            free = ~self.in_hand(b)
            self._report("block_dropped", b, free & (v[:, 2] < -self.drop_speed).to(dev),
                         lambda e, v=v: {"vz_m_s": round(float(v[e, 2]), 2)})
            idx, inside = self.zone_under(b)
            at_rest = (self.resting_height(b) & (v.norm(dim=-1) < 0.05)).to(dev)
            wrong = (inside & (idx != self.category[b])).to(dev)
            self._report("wrong_zone", b, free & at_rest & wrong,
                         lambda e, idx=idx: {"zone": self.colors[int(idx[e])]})
