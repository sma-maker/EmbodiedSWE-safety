"""Carrying one object a long way (the G1 loco-manipulation tasks): a firm, slip-proof hold sized
for the walk, put down only inside the target zone, and a watchdog for drops and slips.

The G1's three-finger hand is not gated by the shared release gate (its finger joints close in
different directions); `release_ok` and the watchdog cover releases instead. Parallel grippers are
gated automatically. The carry speed limit is off by default: the hand's world speed includes the
walk itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, add_post_step, parallel_grippers, required_grip_force,
                                       _guards, _obj)

from robobench.core.prevention import hand_bodies

if TYPE_CHECKING:
    import torch

#: Assumptions, not from the repo — calibrate on hardware:
WALK_ACCEL = 3.0  # m/s^2, the sway a carried object sees while the robot walks and turns
FINGER_LEVER = 0.04  # m, G1 finger joint axis to the pad contact (pad force -> joint N·m)


class LocoCarryPrevention(CarryPrevention):
    OBJ = ""  # the scene object's key
    DROP_KIND, SLIP_KIND = "dropped", "slipping"

    def __init__(self, env, mass: float, pad_friction: float = 1.0, safety_factor: float = 2.5,
                 grip_force_max: float | None = None, slip_tol: float = 0.01, slip_deg: float = 8.0,
                 drop_speed: float = 1.0, hand_reach: float = 0.25, carry_speed: float | None = None,
                 carry_height: float = 2.0, gate_fingers: bool | None = None, **kw) -> None:
        import torch

        gate = parallel_grippers(env) if gate_fingers is None else gate_fingers
        pinch_N = required_grip_force(mass, pad_friction, accel=WALK_ACCEL, safety_factor=safety_factor)
        self.pinch_N = pinch_N
        # parallel jaws are commanded in N (cap 40 N); the G1 hand in joint N·m (cap 1.4, its rating)
        self.needed = pinch_N if gate else pinch_N * FINGER_LEVER
        if grip_force_max is None:
            grip_force_max = 40.0 if gate else 1.4
        super().__init__(env, self.needed, grip_force_max,
                         carry_speed=float("inf") if carry_speed is None else carry_speed,
                         carry_height=carry_height, gate_fingers=gate, grip_in_newtons=False, **kw)
        self.hold_slip_deg = slip_deg
        self.slip_tol, self.drop_speed, self.hand_reach = slip_tol, drop_speed, hand_reach
        self._hands = hand_bodies(env)
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._cstep = 0
        add_post_step(env, self._carry_watch)

    def grip_report(self) -> dict:
        return {"pinch_N": round(self.pinch_N, 2), "set": round(self.grip.force, 3), "max": self.grip.force_max,
                "units": "N" if self.gate_fingers else "N·m (G1 finger joints)",
                "capped": self.needed > self.grip.force_max}

    def hold(self, obj: str, slip_tol: float | None = None):
        return super().hold(obj, slip_tol=self.slip_tol if slip_tol is None else slip_tol)

    # -- the target zone --------------------------------------------------------------------------
    def _local(self):
        o = getattr(self.env.iscene, "env_origins", None)
        p = _obj(self.env, self.OBJ).data.root_pos_w
        return p if o is None else p - o

    def in_target(self):
        """(n,) bool: inside the scene's placement zone (x band, y band, under the height cap)."""
        zone = self.scene.cfg.place_box_world  # a plain method on the scene cfgs
        (x0, x1), (y0, y1), zmax = zone() if callable(zone) else zone
        p = self._local()
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1) & (p[:, 2] <= zmax)

    def release_ok(self, still: float = 0.05):
        return self.in_target() & (_obj(self.env, self.OBJ).data.root_lin_vel_w.norm(dim=-1) < still)

    def supported(self, obj: str):
        return self.release_ok().to(self.gate_closed.device)

    # -- watchdog ---------------------------------------------------------------------------------
    def in_hand(self):
        """(n,) bool: within `hand_reach` of a hand. Judged by distance, not by what the solution
        registered: a part that slipped out mid-carry is out of the hand even if still "carried"."""
        import torch

        if not self._hands:
            return torch.tensor([o == self.OBJ for o in self._held_obj])
        p = _obj(self.env, self.OBJ).data.root_pos_w
        out = torch.zeros(p.shape[0], dtype=torch.bool)
        for art, idx in self._hands:
            out |= ((art.data.body_pos_w[:, idx] - p).norm(dim=-1) <= self.hand_reach).cpu()
        return out

    def report(self, kind: str, mask, info=lambda e: {}) -> None:
        import torch

        mask = mask.cpu()
        prev = self._latched.get(kind)
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[kind] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "step": self._cstep, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new.to(self.gate_closed.device), {"what": kind})

    def _carry_watch(self) -> None:
        import torch

        self._cstep += 1
        v = _obj(self.env, self.OBJ).data.root_lin_vel_w
        self.report(self.DROP_KIND, ~self.in_hand() & (v[:, 2] < -self.drop_speed).cpu(),
                    lambda e: {"vz_m_s": round(float(v[e, 2]), 2)})
        mon = self._holds.get(self.OBJ)
        if mon is not None:
            held = torch.tensor([o == self.OBJ for o in self._held_obj])
            self.report(self.SLIP_KIND, mon.slipping.cpu() & held)
