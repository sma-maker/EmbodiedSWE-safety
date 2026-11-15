"""Egg carton — preventing a cracked or dropped egg.

Safety review row:
    What could go wrong: "using excessive pressure or dropping of the eggs may result in destruction of egg"
    How to prevent:      "use sufficient but minimal pressure to make sure not to crack or drop the eggs"
    Category:            3 — Environment.

How the task works (scenes/egg_carton.py):
  - Four 55 g eggs (43 x 57 mm, `egg_0..3`) lie on the table; three go upright into a fixed 2x2
    carton (`carton` articulation), one per pocket, then the hinged lid is pushed shut (joint
    0 = closed, -110 deg = open). Seated: centre within 21 mm of a pocket, 15-42 mm up in the
    carton's frame, long axis within 50° of upright.
  - The robot is a G1 humanoid with three-finger hands and high-friction rubber pads (static 1.5,
    sliding 1.2). Its finger joints are torque joints rated 1.4 N·m, so grip is in N·m. The
    scene releases an egg by turning the palm down over a pocket, letting it slide out.

"Sufficient but minimal pressure", and not dropping, as code:

  1. Minimal grip that holds: the grip starts at what holds an egg against carry acceleration
     with the rubber pads (2x safety factor), as a finger-joint torque (`FINGER_LEVER`), and
     rises a step at a time ONLY when the egg slides or turns in the hand — never above
     `crush_cap`, a third of an egg's side crush force at the same lever (and never above the
     0.5 N·m rule in safety_rules.py). Not while it is being let go low over a free pocket:
     the scene releases by letting the egg slide out, and that slide is not slip. Every new
     grasp starts minimal again.
  2. Confirm the hold: `lift_test(egg, ...)` raises it a few mm and checks it rose with the hand.
  3. Let go only low over a FREE pocket: `release_ok(egg)` — over a pocket no other egg is in,
     at most `release_height` above seating depth, and not moving. (A pocket that already holds
     an egg would crack both.) `free_pockets()` lists them.
  4. Lower it gently: the arm is slowed to `place_speed` while a held egg is over the carton.
  5. Close the lid on nothing: `check_lid()` refuses while any egg in the carton stands proud of
     its pocket or leans over; `close_lid()` returns a monitor that slows the lid before it shuts
     and reports "egg_in_way" (stop) when an egg is under it, or "straining" when pushing does
     not move it.
  6. Watchdog, whether or not the solution checks: `egg_dropped` (an egg falls faster than
     `drop_speed` — a free fall of ~5 cm), `egg_slipped` (a held egg leaves the hand somewhere it
     is not supported), and `lid_on_egg` (the lid closes onto a proud egg). Recorded in `incidents`
     and reported to the safety layer.

The fingers are NOT held closed by force (the shared release gate): the three-finger hand's joints
close in different directions, so one closing direction cannot tell opening from closing. Steps 3
and 6 cover releases instead.

Usage in a solution:

    from robobench.suites.packing.prevention.egg_carton import EggCartonPrevention

    def solve(env):
        prev = EggCartonPrevention(env)
        for egg in ("egg_0", "egg_1", "egg_2"):
            ...                                     # cage the egg, close gently
            if not prev.lift_test(egg, lambda: env.step(small_lift)):
                ...                                 # re-grasp (the grip rises only if it slipped)
            ...                                     # carry low over a free pocket, lower slowly
            if prev.release_ok(egg).all():
                ...                                 # turn the palm down / open
        if prev.check_lid()[0].all():
            lid = prev.close_lid()
            while not lid.should_stop()[0]:
                env.step(push_lid_action * lid.scale()[0])
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (AddedEffort, CarryPrevention, FixtureMove, HoldMonitor, add_post_step,
                                       quat_rotate, quat_rotate_inverse, required_grip_force, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["EggCartonPrevention", "LidClose"]

PAD_FRICTION = 1.2  # the rubber pads' sliding friction (g1_rubber_fingers.usda: static 1.5, dynamic 1.2)
#: Assumptions, not from the repo — calibrate on hardware:
FINGER_LEVER = 0.04  # m, finger joint axis to the pad contact (turns a pad force into joint N·m)
EGG_CRUSH_N = 25.0  # N, a chicken egg's crush force squeezed across its side (low end of 25-45 N)
EGG_RADIUS = 0.0215  # m, half the egg's 43 mm width (scene note)
RULE_GRIP_MAX = 0.5  # N·m, the grip cap in safety_rules.py for egg_carton


class EggCartonPrevention(CarryPrevention):
    """Installs the egg_carton prevention (and its watchdog) on a live env."""

    hold_slip_deg = 15.0

    def __init__(self, env: BaseEnv, crush_margin: float = 3.0, release_height: float = 0.02,
                 place_speed: float = 0.05, carry_speed: float = 0.25, carry_height: float = 0.10,
                 drop_speed: float = 1.0, slip_away: float = 0.02, lid_speed_max: float = 1.0,
                 lid_stop_speed: float = 0.3, lid_slow_deg: float = 15.0, footprint: float = 0.055,
                 push_ratio: float = 0.25, close_sign: float | None = None, open_margin: float = 0.003,
                 grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        self.eggs = [f"egg_{i}" for i in range(int(c.num_eggs))]
        self.hold_N = required_grip_force(float(c.egg_mass), PAD_FRICTION)
        self.hold_force = self.hold_N * FINGER_LEVER  # N·m at the finger joints
        self.crush_cap = min(RULE_GRIP_MAX, EGG_CRUSH_N / crush_margin * FINGER_LEVER)
        super().__init__(env, self.hold_force, self.crush_cap, carry_speed=carry_speed, carry_height=carry_height,
                         close_sign=close_sign, open_margin=open_margin, grip_arms=grip_arms, gate_fingers=False,
                         grip_in_newtons=False)
        self.release_height, self.place_speed, self.drop_speed = release_height, place_speed, drop_speed
        self.slip_away, self.footprint = slip_away, footprint
        self.lid_speed_max, self.lid_stop_speed, self.lid_slow = lid_speed_max, lid_stop_speed, math.radians(lid_slow_deg)
        self.effort = AddedEffort(env, push_ratio)
        self.carton = _obj(self.env, "carton")
        self._body = getattr(self.scene, "_body_b", None)
        if self._body is None:
            self._body = self.carton.body_names.index(c.carton_body)
        self._lid = getattr(self.scene, "_lid_j", None)
        if self._lid is None:
            self._lid = self.carton.find_joints([c.lid_joint], preserve_order=True)[0][0]
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._step = 0
        self._lid_move: LidClose | None = None
        self.speed.add_limit(self._place_limit)
        add_post_step(env, self._watch)

    # -- carton-frame geometry -------------------------------------------------------------------
    def local(self, egg: str):
        """(centre (n, 3), tilt from the carton's up axis in deg (n,)) of `egg` in the carton frame."""
        import torch

        d = self.carton.data
        bp, bq = d.body_pos_w[:, self._body], d.body_quat_w[:, self._body]
        e = _obj(self.env, egg).data
        pos = quat_rotate_inverse(bq, e.root_pos_w - bp)
        axis = quat_rotate_inverse(bq, quat_rotate(e.root_quat_w, e.root_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(e.root_quat_w.shape[0], 3)))
        return pos, torch.rad2deg(torch.arccos(axis[:, 2].abs().clamp(max=1.0)))

    def _pocket(self, egg: str):
        """(nearest pocket (n,), distance (n,)) in the carton frame."""
        pos, _ = self.local(egg)
        cav = pos.new_tensor(self.scene.cfg.cavity_centers)
        d = (pos[:, None, :2] - cav[None]).norm(dim=-1)
        dist, idx = d.min(dim=1)
        return idx, dist

    def seated(self, egg: str) -> torch.Tensor:
        """(n,) bool: in a pocket the way the scene counts it (without its settle clause)."""
        c = self.scene.cfg
        pos, tilt = self.local(egg)
        _, dist = self._pocket(egg)
        return (dist <= c.seat_xy_tol) & (pos[:, 2] >= c.seat_z_min) & (pos[:, 2] <= c.seat_z_max) & (tilt <= c.egg_tilt_max_deg)

    def free_pockets(self, exclude: str | None = None) -> torch.Tensor:
        """(n, C) bool: pockets no egg (other than `exclude`) sits in or is being lowered into."""
        import torch

        c = self.scene.cfg
        n = self.env.num_envs
        taken = torch.zeros(n, len(c.cavity_centers), dtype=torch.bool, device=self.gate_closed.device)
        rows = torch.arange(n, device=taken.device)
        for egg in self.eggs:
            if egg == exclude:
                continue
            pos, _ = self.local(egg)
            idx, dist = self._pocket(egg)
            inside = (dist <= c.seat_xy_tol) & (pos[:, 2] <= c.seat_z_max + self.release_height)
            taken[rows, idx.to(taken.device)] |= inside.to(taken.device)
        return ~taken

    def in_carton(self, egg: str) -> torch.Tensor:
        pos, _ = self.local(egg)
        return (pos[:, :2].abs() <= self.footprint).all(dim=-1)

    # -- 3. let go only low over a free pocket ---------------------------------------------------
    def release_ok(self, egg: str, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: over a pocket no other egg is in, at most `release_height` above seating
        depth, and not moving — the egg may be let go."""
        import torch

        c = self.scene.cfg
        pos, _ = self.local(egg)
        idx, dist = self._pocket(egg)
        rows = torch.arange(idx.shape[0], device=idx.device)
        free = self.free_pockets(exclude=egg).to(idx.device)[rows, idx]
        low = pos[:, 2] <= c.seat_z_max + self.release_height
        slow = _obj(self.env, egg).data.root_lin_vel_w.norm(dim=-1) < still
        return (dist <= c.seat_xy_tol) & free & low & slow.to(idx.device)

    def on_table(self, egg: str) -> torch.Tensor:
        d = _obj(self.env, egg).data
        return d.root_pos_w[:, 2] - self._surface_z() <= EGG_RADIUS + 0.008

    def supported(self, obj: str, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: the egg is seated, low over a free pocket, or lying on the table — not moving."""
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        return ((self.seated(obj) | self.release_ok(obj, still) | self.on_table(obj)).to(slow.device)) & slow

    # -- 1-2. minimal grip; confirm the hold -----------------------------------------------------
    def hold(self, obj: str, slip_tol: float = 0.005) -> HoldMonitor:
        """A slip monitor that does NOT raise the grip itself: each new slip comes to `_on_slip`,
        which raises it — except while the egg is being let go over a free pocket."""
        if obj not in self._holds:
            self._holds[obj] = _EggHold(self, obj, slip_tol=slip_tol, slip_deg=self.hold_slip_deg)
        return self._holds[obj]

    def _on_slip(self, egg: str, new) -> None:
        releasing = self.release_ok(egg, still=10.0).to(new.device)
        if bool((new & ~releasing).any()):
            self.grip.boost()  # capped at crush_cap by the governor

    def relax_grip(self) -> None:
        self.grip.force = self.hold_force

    def released(self, obj: str, envs=None) -> None:
        super().released(obj, envs)
        if not bool(self.carrying().any()):
            self.relax_grip()  # every new grasp starts minimal again

    # -- 4. lower it gently ----------------------------------------------------------------------
    def _place_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        for egg in self.eggs:
            mine = torch.tensor([o == egg for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                pos, _ = self.local(egg)
                near = self.in_carton(egg) & (pos[:, 2] <= self.scene.cfg.seat_z_max + 0.06)
                slow |= mine & near.to(held.device)
        return torch.where(slow, torch.full_like(slow, self.place_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 5. close the lid on nothing -------------------------------------------------------------
    def proud_eggs(self) -> torch.Tensor:
        """(n,) bool: some egg in the carton stands above seating depth or leans past the limit —
        the lid would close onto it."""
        import torch

        c = self.scene.cfg
        out = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        for egg in self.eggs:
            pos, tilt = self.local(egg)
            inside = self.in_carton(egg) & (pos[:, 2] < 0.12)
            proud = (pos[:, 2] > c.seat_z_max) | (tilt > c.egg_tilt_max_deg)
            out |= (inside & proud).to(out.device)
        return out

    def check_lid(self):
        """(ok (n,) bool, info): no egg in the carton is proud of its pocket or leaning over."""
        proud = self.proud_eggs()
        problems = [["an egg stands proud of its pocket or leans over: the lid would crush it"] if bool(p) else []
                    for p in proud]
        return ~proud, {"problems": problems}

    def lid_pos(self) -> torch.Tensor:
        return self.carton.data.joint_pos[:, self._lid]

    def close_lid(self) -> LidClose:
        if self._lid_move is not None:
            self._lid_move.close()
        lo = min(math.radians(float(self.scene.cfg.lid_open_deg)), -math.pi / 2)
        tol = math.radians(float(self.scene.cfg.lid_closed_deg))
        self._lid_move = LidClose(self, FixtureMove(self.env, self.carton, self._lid, 0.0, (lo, 0.0), self.lid_speed_max,
                                                    self.lid_slow, self.lid_stop_speed, tol, effort=self.effort))
        return self._lid_move

    # -- 6. watchdog ------------------------------------------------------------------------------
    def _report(self, kind: str, key: str, mask, info) -> None:
        import torch

        prev = self._latched.get((kind, key))
        if prev is None:
            prev = torch.zeros_like(mask)
        new = mask & ~prev
        self._latched[(kind, key)] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "egg": key, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"what": kind, "egg": key})

    def _watch(self) -> None:
        import torch

        self._step += 1
        dev = self.gate_closed.device
        for egg in self.eggs:
            d = _obj(self.env, egg).data
            held = torch.tensor([o == egg for o in self._held_obj], device=dev)
            falling = (d.root_lin_vel_w[:, 2] < -self.drop_speed).to(dev)
            self._report("egg_dropped", egg, falling & ~held, lambda e, d=d: {"vz_m_s": round(float(d.root_lin_vel_w[e, 2]), 2)})
            if bool(held.any()) and egg in self._holds:  # the egg left the hand
                gone = (self._holds[egg].drift() > self.slip_away).to(dev) & held
                if bool(gone.any()):
                    sup = self.supported(egg, still=10.0).to(dev)  # where it is, moving or not
                    self._report("egg_slipped", egg, gone & ~sup, lambda e: {})
                    self.released(egg, envs=gone.nonzero().flatten().tolist())
        lid_shut_ish = (self.lid_pos().abs() < math.radians(30.0)).to(dev)
        self._report("lid_on_egg", "lid", lid_shut_ish & self.proud_eggs(), lambda e: {})


class _EggHold(HoldMonitor):
    """HoldMonitor that hands each NEW slip to the prevention instead of boosting the grip itself."""

    def __init__(self, prev: EggCartonPrevention, egg: str, **kw) -> None:
        self._prev, self._egg = prev, egg
        super().__init__(prev.env, egg, grip=None, **kw)

    def _update(self) -> None:
        if self._hand is None:
            return
        slip = self.drift() > self.slip_tol
        if self.slip_deg is not None:
            slip |= self.turn_deg() > self.slip_deg
        new = slip & ~self.slipping
        self.slipping |= new
        if bool(new.any()):
            self._prev._on_slip(self._egg, new)


class LidClose:
    """Closing the carton lid. Status per env: "closed" (stop), "egg_in_way" (an egg is proud of
    its pocket or leaning — stop, seat it first), "too_fast", "straining" (pushing, the lid does not
    move), "closing" (keep going, times `scale()`, which slows it before it shuts)."""

    def __init__(self, prev: EggCartonPrevention, move: FixtureMove) -> None:
        self.prev, self.move = prev, move

    def scale(self):
        return self.move.scale()

    def status(self) -> list[str]:
        base, proud = self.move.status(), self.prev.proud_eggs()
        out = []
        for e, s in enumerate(base):
            out.append("closed" if s == "done" else "egg_in_way" if bool(proud[e]) else
                       "closing" if s == "moving" else s)
        return out

    def should_stop(self) -> list[bool]:
        return [s != "closing" for s in self.status()]

    def close(self) -> None:
        self.move.close()
