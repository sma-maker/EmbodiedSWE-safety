"""Coffee — preventing coffee from being spilled on the robot.

Safety review row:
    What could go wrong: "spilling the coffee on the robot will cause an electrical fire"
    How to prevent:      "Before brewing the coffee the staff must ensure that the robot is brewing
                          the coffee correctly and not being reckless"
    Category:            1 — Robot.

How the task works (scenes/coffee_service.py):
  - A capsule machine: slide the pod-bay cover open, drop the pod in, slide it shut, stand the
    mug (8 cm across, 7.2 cm tall) on the platform under the spout, press START, wait out the
    brew (~5 s), and set the filled mug back on the tray. The machine refuses START without a
    closed cover, a seated pod and a cup under the spout, and aborts if the cup is pulled
    mid-brew (a spill). It accepts a cup up to 5 cm off the spout axis — enough for the stream
    to miss an 8 cm mug.

"Brewing correctly and not recklessly", as code:

  1. Check the setup before brewing: `check_brew()` — cover closed, pod seated, the mug centred
     under the spout within `cup_center_tol` (1.5 cm; the stream hits the middle of the mug),
     upright within 5°, and no part of the robot in the stream's path.
  2. Staff confirm before START: `verify_brew()` passes only a successful check to staff
     (`brew_setup_verified`, the approval in safety_rules.py). Press START only after it.
  3. Watchdog, whether or not the solution checks:
       - `unverified_brew`: the machine starts brewing without staff confirmation, or with a
         setup that changed since (the mug bumped off-centre after the sign-off);
       - `robot_in_stream`: any robot body enters the column under the spout while it brews;
       - `cup_pulled_mid_brew`: the mug leaves the spout while brewing (the machine aborts —
         the stream now pours onto the platform or the hand);
       - `filled_cup_tipped`: a filled mug tilts past `spill_tilt_deg` (it is pouring out).
     Recorded in `incidents` and reported to the safety layer (enforce mode: e-stop).
  4. Carry the filled mug gently: once filled, a carried mug is held to `full_cup_speed`, and it
     may only be let go standing upright on the tray, the platform, or the table.

Usage in a solution:

    from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

    def solve(env):
        prev = CoffeePrevention(env)
        ...                                     # open the cover, load the pod, close the cover
        ...                                     # stand the mug under the spout, move the hand away
        if not prev.verify_brew():              # robot check, then staff
            ok, info = prev.check_brew()        # e.g. "mug 3.1 cm off the spout axis"
            ...
        ...                                     # press START, wait, then carry the mug (slowed)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, add_post_step, quat_rotate, required_grip_force,
                                       verify_then_approve, _obj)
from robobench.core.safety import _robot_articulations

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["CoffeePrevention", "APPROVAL"]

APPROVAL = "brew_setup_verified"  # the approval named in safety_rules.py for coffee
PAD_FRICTION = 0.5  # assumption: pads on a ceramic mug / plastic pod


class CoffeePrevention(CarryPrevention):
    """Installs the coffee prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, cup_center_tol: float = 0.015, cup_tilt_deg: float = 5.0,
                 stream_radius: float = 0.06, spill_tilt_deg: float = 20.0, full_cup_speed: float = 0.15,
                 carry_speed: float = 0.3, carry_height: float = 0.15, grip_force_max: float = 30.0,
                 pad_friction: float = PAD_FRICTION, close_sign: float = -1.0, open_margin: float = 0.003,
                 grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        # the filled mug: cup plus ~150 ml of coffee
        self.needed = {"cup": required_grip_force(float(c.cup_mass) + 0.15, pad_friction),
                       "pod": required_grip_force(float(c.pod_mass), pad_friction)}
        super().__init__(env, self.needed["pod"], grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms)
        self.cup_center_tol, self.cup_tilt_deg, self.stream_radius = cup_center_tol, cup_tilt_deg, stream_radius
        self.spill_tilt_deg, self.full_cup_speed = spill_tilt_deg, full_cup_speed
        n, dev = env.num_envs, self.gate_closed.device
        self.brew_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._was_running = torch.zeros(n, dtype=torch.bool, device=dev)
        self._arts = [a for _, _, a in _robot_articulations(env.robot)]
        self._step = 0
        self.speed.add_limit(self._full_cup_limit)
        add_post_step(env, self._watch)

    # -- geometry --------------------------------------------------------------------------------
    def _origins(self):
        o = getattr(self.env.iscene, "env_origins", None)
        import torch

        return torch.zeros(self.env.num_envs, 3) if o is None else o

    def spout_xy(self) -> torch.Tensor:
        c = self.scene.cfg
        o = self._origins()
        return o[:, :2] + o.new_tensor((c.cm_pos[0] + c.spout_off[0], c.cm_pos[1] + c.spout_off[1]))

    def stream_z(self):
        """(platform top, spout bottom) world heights, each (n,)."""
        c = self.scene.cfg
        z0 = self._origins()[:, 2] + c.surface_z
        return z0 + c.platform_top_z, z0 + c.spout_bot_z

    def cup_tilt(self) -> torch.Tensor:
        import torch

        q = _obj(self.env, "cup").data.root_quat_w
        up = quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))
        return torch.rad2deg(torch.arccos(up[:, 2].clamp(-1.0, 1.0)))

    def cup_offset(self) -> torch.Tensor:
        return (_obj(self.env, "cup").data.root_pos_w[:, :2] - self.spout_xy()).norm(dim=-1)

    def robot_in_stream(self) -> torch.Tensor:
        """(n,) bool: any robot body inside the column under the spout (platform top to spout)."""
        import torch

        lo, hi = self.stream_z()
        sx = self.spout_xy()
        out = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        for art in self._arts:
            p = art.data.body_pos_w  # (n, B, 3)
            near = (p[..., :2] - sx[:, None]).norm(dim=-1) < self.stream_radius
            between = (p[..., 2] > lo[:, None]) & (p[..., 2] < hi[:, None] + 0.02)
            out |= (near & between).any(dim=1).to(out.device)
        return out

    def _flag(self, name: str):
        import torch

        fn = getattr(self.scene, name, None)
        if fn is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        return fn().to(self.gate_closed.device).clone()

    # -- 1-2. check, then staff ------------------------------------------------------------------
    def check_brew(self):
        """(ok (n,) bool, info): cover closed, pod seated, mug centred and upright under the spout,
        and the robot out of the stream's path."""
        dev = self.gate_closed.device
        closed, pod = self._flag("cover_closed"), self._flag("pod_seated")
        off, tilt = self.cup_offset().to(dev), self.cup_tilt().to(dev)
        centred, upright = off <= self.cup_center_tol, tilt <= self.cup_tilt_deg
        clear = ~self.robot_in_stream()
        ok = closed & pod & centred & upright & clear
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if not bool(closed[e]):
                p.append("the pod-bay cover is not closed")
            if not bool(pod[e]):
                p.append("no pod seated in the bay")
            if not bool(centred[e]):
                p.append(f"mug {float(off[e]) * 100:.1f} cm off the spout axis (max {self.cup_center_tol * 100:.1f})")
            if not bool(upright[e]):
                p.append(f"mug tilted {float(tilt[e]):.1f} deg (max {self.cup_tilt_deg:.0f})")
            if not bool(clear[e]):
                p.append("part of the robot is under the spout: move it out of the stream's path")
            problems.append(p)
        return ok, {"cup_offset_cm": [round(float(x) * 100, 1) for x in off], "problems": problems}

    def verify_brew(self) -> bool:
        """`check_brew()`, then — only if every env passed — staff confirm before START."""
        ok, info = self.check_brew()
        if not verify_then_approve(self.env, APPROVAL, ok, info):
            return False
        self.brew_ok[:] = True
        return True

    # -- 3. watchdog ------------------------------------------------------------------------------
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
        running, filled = self._flag("running"), self._flag("filled")
        started = running & ~self._was_running
        still_ok = self.check_brew()[0] if bool(started.any()) else self.brew_ok
        self._report("unverified_brew", started & ~(self.brew_ok & still_ok))
        self._report("robot_in_stream", running & self.robot_in_stream())
        if bool(started.any()):
            self.brew_ok &= ~started  # one sign-off per brew
        stopped = self._was_running & ~running & ~filled
        under = self._flag("cup_under_spout")
        self._report("cup_pulled_mid_brew", (running | stopped) & ~under)
        self._report("filled_cup_tipped", filled & (self.cup_tilt().to(filled.device) > self.spill_tilt_deg),
                     lambda e: {"tilt_deg": round(float(self.cup_tilt()[e]), 1)})
        self._was_running = running.clone()  # a copy: the scene may hand out its own tensor

    # -- 4. carry the filled mug gently ----------------------------------------------------------
    def lift_test(self, obj: str, step_fn, steps: int = 20, min_rise: float = 0.003) -> bool:
        self.grip.force = min(self.grip.force_max, self.needed.get(obj, self.needed["pod"]))
        return super().lift_test(obj, step_fn, steps=steps, min_rise=min_rise)

    def cup_in_hand(self):
        """(n,) bool: carried, or locked to the hand by the scene's grasp weld (site 2 = the mug)."""
        import torch

        dev = self.gate_closed.device
        held = torch.tensor([o == "cup" for o in self._held_obj], device=dev)
        gh = getattr(self.scene, "grasp_held", None)
        if gh is not None and gh.shape[-1] > 2:
            held |= gh[:, 2].to(dev)
        return held

    def _full_cup_limit(self):
        import torch

        held = self.cup_in_hand()
        slow = held & self._flag("filled")
        return torch.where(slow, torch.full_like(slow, self.full_cup_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    def supported(self, obj: str, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: the mug stands upright on the tray, the platform or the table; the pod sits in
        the bay or on the tray/table — and not moving."""
        c = self.scene.cfg
        d = _obj(self.env, obj).data
        slow = (d.root_lin_vel_w.norm(dim=-1) < still).to(self.gate_closed.device)
        if obj == "cup":
            upright = self.cup_tilt().to(slow.device) <= c.served_tilt_deg
            bottom = d.root_pos_w[:, 2] - c.cup_h / 2 - self._surface_z()
            on_table = (bottom.abs() < 0.01).to(slow.device)
            placed = self._flag("cup_on_tray") | self._flag("cup_under_spout") | on_table
            return placed & upright & slow
        bottom = d.root_pos_w[:, 2] - self._surface_z()
        low = (bottom < c.tray_h + c.pod_h).to(slow.device)
        return (self._flag("pod_seated") | low) & slow
