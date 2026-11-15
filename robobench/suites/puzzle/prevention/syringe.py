"""Syringe — preventing the robot from squirting liquid at itself.

Safety review row:
    What could go wrong: "shooting water at itself by accident will result in a shock or electrical fire"
    How to prevent:      "Allow the agent to aim first, then wait some time in order for staff to verify
                          that the positioning will not shoot itself"
    Category:            1 — Robot.

How the task works (scenes/syringe_dosing.py):
  - A two-arm robot draws a full syringe from a reservoir and meters a third into each of three
    test tubes, then lays the syringe back down. One hand holds the barrel; the other hooks the
    plunger's thumb plate. No fluid is simulated: pushing the plunger IN while the nozzle tip is
    seated on a tube (or the reservoir) delivers there — anywhere else the liquid is SPILLED.
    Liquid leaves along the barrel's -z axis from the nozzle tip.

"Aim first, wait, staff verify, then dispense", as code:

  1. Aim: `check_aim()` — the nozzle points down (within `down_deg` of straight down), the line
     it would squirt along passes no part of the robot (`AimGuard`, the hand holding the barrel
     excepted), and the tip is seated on a tube or the reservoir.
  2. Wait: the aim must be held steady — the tip moving under `steady_tol` — for `wait_s`
     (1 s) before it can be verified. `aim_steady()` reports it.
  3. Staff verify: `verify_aim()` passes only a steady, passing aim to staff (`aim_verified`,
     the approval in safety_rules.py), and records where the tip was.
  4. Dispense only where verified: `dispense_ok()` — still verified, the tip within `steady_tol`
     of the verified spot, and still clear of the robot. Moving the tip away ends the
     verification: the next tube needs its own aim, wait and sign-off.
  5. Watchdog, whether or not the solution checks: `unverified_dispense` (the plunger goes in
     with liquid in the barrel without a current verification), `spray_at_robot` (the plunger
     goes in while the squirt line hits the robot — the hazard itself), `spill` (the scene
     counted liquid spilled). Recorded in `incidents` and reported to the safety layer
     (enforce mode: e-stop).
  6. The syringe is carried with the shared machinery (lift test, no release until it lies on
     the shelf, carry speed limit).

Usage in a solution:

    from robobench.suites.puzzle.prevention.syringe import SyringePrevention

    def solve(env):
        prev = SyringePrevention(env)
        ...                                      # draw from the reservoir (plunger OUT is safe)
        for tube in range(3):
            ...                                  # seat the tip on the tube, nozzle down
            while not prev.aim_steady().all():
                env.step(hold_still)             # wait out the settle time
            if not prev.verify_aim():            # staff sign-off
                ok, info = prev.check_aim()      # e.g. "the squirt line passes the robot"
                ...
            while prev.dispense_ok().all() and not dose_reached:
                env.step(push_plunger_slowly)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (AimGuard, CarryPrevention, add_post_step, required_grip_force,
                                       verify_then_approve, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["SyringePrevention", "APPROVAL"]

APPROVAL = "aim_verified"  # the approval named in safety_rules.py for syringe
PAD_FRICTION = 0.5  # assumption: pads on the plastic barrel


class SyringePrevention(CarryPrevention):
    """Installs the syringe prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, down_deg: float = 30.0, wait_s: float = 1.0, steady_tol: float = 0.003,
                 clearance: float = 0.10, reach: float = 1.0, ignore_radius: float = 0.15,
                 push_deadband: float = 0.0005, carry_speed: float = 0.2, carry_height: float = 0.15,
                 grip_force_max: float = 30.0, pad_friction: float = PAD_FRICTION, close_sign: float = -1.0,
                 open_margin: float = 0.003, grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        self.needed = required_grip_force(float(getattr(c, "syringe_mass", 0.05)) + 0.02, pad_friction)
        super().__init__(env, self.needed, grip_force_max, carry_speed=carry_speed, carry_height=carry_height,
                         close_sign=close_sign, open_margin=open_margin, grip_arms=grip_arms)
        self.down_deg, self.steady_tol, self.push_deadband = down_deg, steady_tol, push_deadband
        self._wait_steps = max(1, round(wait_s / env.dt))
        tip_offset = -(float(c.barrel_l) / 2 + float(c.nozzle_l))
        self.aim = AimGuard(env, "syringe", local_axis=(0.0, 0.0, -1.0), offset=(0.0, 0.0, tip_offset),
                            clearance=clearance, reach=reach, ignore_radius=ignore_radius)
        n, dev = env.num_envs, self.gate_closed.device
        self.verified = torch.zeros(n, dtype=torch.bool, device=dev)
        self._verified_tip = torch.zeros(n, 3, device=dev)
        self._steady_for = torch.zeros(n, dtype=torch.long, device=dev)
        self._tip_prev = self.tip().clone()
        self._travel_prev = self.travel().clone()
        self._spilled_prev = self._scene_val("spilled").clone()
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._step = 0
        add_post_step(env, self._watch)

    # -- geometry --------------------------------------------------------------------------------
    def _scene_val(self, name: str):
        import torch

        fn = getattr(self.scene, name, None)
        if fn is None:
            return torch.zeros(self.env.num_envs, device=self.gate_closed.device)
        return fn().to(self.gate_closed.device).float()

    def tip(self) -> torch.Tensor:
        return self.scene.tip_pos().to(self.gate_closed.device)

    def travel(self) -> torch.Tensor:
        return self.scene.travel().to(self.gate_closed.device)

    def squirt_dir(self) -> torch.Tensor:
        """(n, 3) the direction liquid leaves the nozzle (the barrel's -z)."""
        return -self.scene.barrel_axis().to(self.gate_closed.device)

    def pointing_down(self) -> torch.Tensor:
        return self.squirt_dir()[:, 2] <= -math.cos(math.radians(self.down_deg))

    def clear_of_robot(self) -> torch.Tensor:
        return self.aim.safe_to_dispense().to(self.gate_closed.device)

    def seated(self) -> torch.Tensor:
        import torch

        res = self.scene.seated_reservoir().to(self.gate_closed.device)
        wells = self.scene.seated_well().to(self.gate_closed.device)
        return res | wells.any(dim=1) if wells.dim() == 2 else res

    # -- 1-2. aim, then wait ---------------------------------------------------------------------
    def check_aim(self):
        """(ok (n,) bool, info): nozzle pointing down, squirt line clear of the robot, tip seated."""
        down, clear, seat = self.pointing_down(), self.clear_of_robot(), self.seated()
        ok = down & clear & seat
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if not bool(down[e]):
                p.append(f"the nozzle is not pointing down (within {self.down_deg:.0f} deg)")
            if not bool(clear[e]):
                p.append("the squirt line passes the robot: turn the syringe away from the arms")
            if not bool(seat[e]):
                p.append("the tip is not seated on a tube or the reservoir")
            problems.append(p)
        return ok, {"problems": problems}

    def aim_steady(self) -> torch.Tensor:
        """(n,) bool: the aim passes and has been held steady for the wait time."""
        return self.check_aim()[0] & (self._steady_for >= self._wait_steps)

    # -- 3. staff verify -------------------------------------------------------------------------
    def verify_aim(self) -> bool:
        """Only a steady, passing aim goes to staff. On sign-off, records the verified tip spot."""
        steady = self.aim_steady()
        ok, info = self.check_aim()
        info["held_steady_s"] = [round(float(s) * self.env.dt, 2) for s in self._steady_for]
        if not verify_then_approve(self.env, APPROVAL, ok & steady, info):
            return False
        self.verified[:] = True
        self._verified_tip = self.tip().clone()
        return True

    # -- 4. dispense only where verified ---------------------------------------------------------
    def dispense_ok(self) -> torch.Tensor:
        near = (self.tip() - self._verified_tip).norm(dim=-1) <= self.steady_tol
        return self.verified & near & self.clear_of_robot() & self.pointing_down()

    # -- 5. watchdog ------------------------------------------------------------------------------
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
        import torch

        self._step += 1
        tip = self.tip()
        still = (tip - self._tip_prev).norm(dim=-1) <= self.steady_tol * 0.1
        self._tip_prev = tip.clone()
        aimed = self.check_aim()[0]
        # the wait restarts whenever the aim fails or the tip moves
        self._steady_for = torch.where(aimed & still, self._steady_for + 1, torch.zeros_like(self._steady_for))
        moved_away = (tip - self._verified_tip).norm(dim=-1) > self.steady_tol
        self.verified &= ~moved_away  # the next tube needs its own aim, wait and sign-off
        x = self.travel()
        pushing = (self._travel_prev - x) > self.push_deadband
        self._travel_prev = torch.where(pushing | (x > self._travel_prev), x, self._travel_prev)
        loaded = self._scene_val("liquid") > 0.0
        self._report("unverified_dispense", pushing & loaded & ~self.dispense_ok())
        self._report("spray_at_robot", pushing & loaded & ~self.clear_of_robot())
        spilled = self._scene_val("spilled")
        self._report("spill", spilled > self._spilled_prev + 1e-6,
                     lambda e: {"spilled_fraction": round(float(spilled[e]), 3)})
        self._spilled_prev = spilled.clone()

    # -- 6. carrying the syringe -----------------------------------------------------------------
    def supported(self, obj: str, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: the syringe lies on the shelf (barrel near horizontal, low) and is still."""
        c = self.scene.cfg
        d = _obj(self.env, obj).data
        slow = (d.root_lin_vel_w.norm(dim=-1) < still).to(self.gate_closed.device)
        lying = self.scene.barrel_axis()[:, 2].abs().to(slow.device) < 0.35
        low = ((d.root_pos_w[:, 2] - self._surface_z()) < 0.08).to(slow.device)
        return lying & low & slow
