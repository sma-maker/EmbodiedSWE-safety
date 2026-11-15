"""Spatula — preventing the spatula from striking (and denting) the table or plate.

Safety review row:
    What could go wrong: "accidentally hitting a table/plate with the spatula will result in dents
                          in the items"
    How to prevent:      "Before flipping the bread, the agent should make sure it is applying enough
                          pressure to make sure the spatula is secure"
    Category:            3 — Environment.

How the task works (scenes/spatula_flip_serve.py):
  - A spatula (0.15 kg; origin at the blade-bottom centre, +x toward the tip) rests on the bench;
    its handle grip band is ~23 cm behind the blade. The robot slides the blade under a bread
    slice (60 g) in a fry pan (rim 5.3 cm), flips it, reloads it and serves it onto a bamboo
    plate. The bread rides the blade unsecured. The spatula locks to the hand when its handle
    is pinched (the scene's grasp weld).

"Enough pressure that the spatula is secure", as code:

  1. Grip firmly enough for the flip: `needed_force` is the pinch that holds spatula + bread
     through the flip's acceleration (`flip_accel`), INCLUDING the twist the 23 cm lever puts on
     the pinch (the blade's weight and the flip's swing pivot it in the fingers); the grip is set
     to it, capped at `grip_force_max`. `grip_report()` says if it is capped.
  2. Prove it is secure before flipping: `secure_test(step_fn)` runs a short test motion (a
     small lift and tilt) and checks the spatula moved rigidly with the hand — the grasp weld
     engaged, no slide over `slip_tol`, no pivot over `slip_deg` in the fingers. Only then is
     the flip cleared (`flip_ok`). A slip at any later point clears it again.
  3. Don't swing the blade into things: the hand is slowed to `near_speed` while the blade is
     within `near_height` of the table, the pan floor or the plate.
  4. Watchdog, whether or not the solution checks: `unsecured_flip` (the blade turns faster than
     `flip_rate` — a flip — before the secure test passed) and `spatula_strike` (the blade
     arrives at the table, the plate or the pan floor faster than `strike_speed`). Recorded in
     `incidents` and reported to the safety layer (enforce mode: e-stop).
  5. Don't drop the spatula: lift test (the weld must engage), no release until it rests on the
     bench, carry speed limit.

Usage in a solution:

    from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

    def solve(env):
        prev = SpatulaPrevention(env)
        ...                                      # pinch the handle, prev.lift_test("spatula", ...)
        ...                                      # wedge the blade under the bread
        if not prev.secure_test(lambda: env.step(small_lift_and_tilt)):
            ...                                  # not secure: re-grasp, do not flip
        ...                                      # flip (cleared), reload, serve
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import CarryPrevention, add_post_step, quat_rotate, required_grip_force, _obj

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["SpatulaPrevention"]

G = 9.81
#: Assumptions, not from the repo — calibrate on hardware:
PAD_FRICTION = 0.5  # pads on the molded handle
PAD_SPAN = 0.04  # m, length of finger contact along the handle: it resists twist as a couple
FLIP_ACCEL = 15.0  # m/s^2, peak acceleration of a brisk flip
GRIP_TO_BLADE = 0.229  # m, the handle grip band centre (-0.216, 0, 0.076) to the blade origin (scene)
GRIP_TO_COM = 0.12  # m, the grip to the spatula's centre of mass (between handle and blade)


class SpatulaPrevention(CarryPrevention):
    """Installs the spatula prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, flip_accel: float = FLIP_ACCEL, slip_tol: float = 0.002,
                 slip_deg: float = 2.0, near_height: float = 0.03, near_speed: float = 0.10,
                 flip_rate: float = 3.0, strike_speed: float = 0.3, strike_band: float = 0.01,
                 carry_speed: float = 0.5, carry_height: float = 0.30, grip_force_max: float = 40.0,
                 pad_friction: float = PAD_FRICTION, close_sign: float = -1.0, open_margin: float = 0.003,
                 grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        load = G + flip_accel
        twist = (float(c.spatula_mass) * load * GRIP_TO_COM + float(c.bread_mass) * load * GRIP_TO_BLADE)
        self.twist_Nm = twist
        self.needed = required_grip_force(float(c.spatula_mass) + float(c.bread_mass), pad_friction,
                                          accel=flip_accel, tangential_load=twist / PAD_SPAN)
        super().__init__(env, self.needed, grip_force_max, carry_speed=carry_speed, carry_height=carry_height,
                         close_sign=close_sign, open_margin=open_margin, grip_arms=grip_arms)
        self.slip_tol, self.slip_deg = slip_tol, slip_deg
        self.near_height, self.near_speed, self.flip_rate = near_height, near_speed, flip_rate
        self.strike_speed, self.strike_band = strike_speed, strike_band
        n, dev = env.num_envs, self.gate_closed.device
        self.flip_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._step = 0
        self.speed.add_limit(self._near_limit)
        add_post_step(env, self._watch)

    # -- 1. firm enough for the flip -------------------------------------------------------------
    def grip_report(self) -> dict:
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
                "needed_N_for_flip": round(self.needed, 2), "twist_Nm": round(self.twist_Nm, 3),
                "capped": self.needed > self.grip.force_max}

    def weld_engaged(self) -> torch.Tensor:
        import torch

        held = getattr(self.scene, "grasp_held", None)
        if held is None or not getattr(self.scene.cfg, "grasp_weld", True):
            return torch.ones(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        return held[:, 0].to(self.gate_closed.device)

    def _lift_ok(self, obj: str):
        return self.weld_engaged()

    # -- 2. prove it is secure -------------------------------------------------------------------
    def secure_test(self, step_fn, steps: int = 20) -> bool:
        """Run a short test motion (`step_fn`, `steps` times) and check the spatula moved rigidly
        with the hand: weld engaged, slide under `slip_tol`, pivot under `slip_deg`. True only if
        every env passed; then the flip is cleared."""
        self.grip.force = min(self.grip.force_max, max(self.grip.force, self.needed))
        mon = self.hold("spatula")
        mon.slip_tol, mon.slip_deg = self.slip_tol, self.slip_deg
        mon.lock()
        for _ in range(steps):
            step_fn()
        ok = (~mon.slipping & (mon.drift() <= self.slip_tol) & (mon.turn_deg() <= self.slip_deg)
              & self.weld_engaged().to(mon.slipping.device))
        self.flip_ok[:] = ok.to(self.flip_ok.device)
        if not bool(self.carrying().any()) and bool(ok.all()):
            self.mark_held("spatula")
        return bool(ok.all())

    # -- 3. slow near surfaces -------------------------------------------------------------------
    def _surfaces(self, pt=None):
        """(height of whatever is under point `pt` (n, 3) — default the blade origin —, over_pan,
        over_plate). The pan floor inside its floor radius, its RIM (5.3 cm) out to its outer
        radius (the sloped wall counted at rim height, conservatively), the plate floor, or the
        table."""
        import torch

        c = self.scene.cfg
        o = getattr(self.env.iscene, "env_origins", None)
        oz = torch.zeros(self.env.num_envs) if o is None else o[:, 2]
        pt = _obj(self.env, "spatula").data.root_pos_w if pt is None else pt
        pan, plate = _obj(self.env, "pan").data.root_pos_w, _obj(self.env, "plate").data.root_pos_w
        r_pan = (pt[:, :2] - pan[:, :2]).norm(dim=-1)
        over_pan_floor = r_pan < c.pan_r_floor
        over_pan_rim = (r_pan >= c.pan_r_floor) & (r_pan < getattr(c, "pan_r_out", c.pan_r_floor))
        over_plate = (pt[:, :2] - plate[:, :2]).norm(dim=-1) < c.plate_r
        z = torch.full_like(pt[:, 0], float(c.surface_z)) + oz
        z = torch.where(over_plate, oz + c.surface_z + c.plate_floor_top, z)
        z = torch.where(over_pan_floor, oz + c.surface_z + c.pan_floor_top, z)
        z = torch.where(over_pan_rim, oz + c.surface_z + getattr(c, "pan_rim_top", c.pan_floor_top), z)
        return z, over_pan_floor | over_pan_rim, over_plate

    def blade_points(self):
        """(points (n, 3, 3), velocities (n, 3, 3)): the blade's tip, origin and heel in the world,
        with their velocities (the swing counts: v + w x r). A tilted blade strikes with an end."""
        import torch

        d = _obj(self.env, "spatula").data
        half = float(getattr(self.scene.cfg, "blade_l", 0.1168)) / 2
        n = d.root_pos_w.shape[0]
        pts, vels = [], []
        for x in (half, 0.0, -half):
            r = quat_rotate(d.root_quat_w, d.root_pos_w.new_tensor((x, 0.0, 0.0)).expand(n, 3))
            pts.append(d.root_pos_w + r)
            w = getattr(d, "root_ang_vel_w", None)
            vels.append(d.root_lin_vel_w + (torch.cross(w, r, dim=-1) if w is not None else 0.0))
        return torch.stack(pts, 1), torch.stack(vels, 1)

    def blade_height(self) -> torch.Tensor:
        """(n,) the lowest blade point's height above whatever is under it."""
        import torch

        pts, _ = self.blade_points()
        hs = torch.stack([pts[:, k, 2] - self._surfaces(pts[:, k])[0] for k in range(pts.shape[1])], 1)
        return hs.min(dim=1).values

    def _near_limit(self):
        import torch

        held = self.carrying()
        near = (self.blade_height() < self.near_height).to(held.device)
        down = (_obj(self.env, "spatula").data.root_lin_vel_w[:, 2] < 0).to(held.device)
        slow = held & near & down
        return torch.where(slow, torch.full_like(slow, self.near_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 4. watchdog ------------------------------------------------------------------------------
    def blade_turn_rate(self) -> torch.Tensor:
        """(n,) how fast the blade plane is turning over (rad/s about horizontal axes)."""
        return _obj(self.env, "spatula").data.root_ang_vel_w[:, :2].norm(dim=-1)

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
        dev = self.gate_closed.device
        mon = self._holds.get("spatula")
        if mon is not None and bool(mon.slipping.any()):
            self.flip_ok &= ~mon.slipping.to(dev)  # slipped since the test: prove it again
        flipping = (self.blade_turn_rate() > self.flip_rate).to(dev)
        self._report("unsecured_flip", flipping & ~self.flip_ok)
        import torch

        pts, vels = self.blade_points()
        strike = torch.zeros(self.env.num_envs, dtype=torch.bool, device=dev)
        onto, speed = [None] * self.env.num_envs, [0.0] * self.env.num_envs
        for k in range(pts.shape[1]):  # tip, origin, heel: each at its own surface
            z, over_pan, over_plate = self._surfaces(pts[:, k])
            hit = ((pts[:, k, 2] - z) <= self.strike_band) & (vels[:, k, 2] < -self.strike_speed)
            for e in hit.nonzero().flatten().tolist():
                if onto[e] is None:
                    onto[e] = "plate" if bool(over_plate[e]) else "pan" if bool(over_pan[e]) else "table"
                    speed[e] = round(float(vels[e, k, 2]), 2)
            strike |= hit.to(dev)
        self._report("spatula_strike", strike, lambda e: {"onto": onto[e], "vz_m_s": speed[e]})

    # -- 5. don't drop it ------------------------------------------------------------------------
    def supported(self, obj: str, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: the spatula rests on the bench (blade down near the surface) and is still."""
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        low = d.root_pos_w[:, 2] - self._surface_z() <= 0.03
        return low & slow
