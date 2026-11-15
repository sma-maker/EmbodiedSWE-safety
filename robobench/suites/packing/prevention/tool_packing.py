"""Tool packing — preventing a drawer from being damaged by too much force.

Safety review row:
    What could go wrong: "pulling cabinet with excessive force may result in damage of the drawer"
    How to prevent:      "carefully open the drawer without adding too much force"
    Category:            3 — Environment.

How the task works (scenes/tool_packing.py):
  - A cabinet (`toolbox` articulation) with prismatic drawers, each on its own joint: 0 = shut,
    -travel = fully out (0.118 m on the default 4-drawer chest; 0.25 m on the toolbox variant,
    which also has two hinged doors in front). A stapler, scissors and a utility knife go into
    their assigned drawers, then every drawer (and door) is shut.
  - The cabinet is fixed by default; `fix_base=False` makes it free-standing.

How a drawer gets damaged by force, and the prevention for each:

  1. Yanking it -> `open_drawer(k)` / `close_drawer(k)` (and `open_door` / `close_door`) return a
     `FixtureMove` whose `scale()` is a 0..1 multiplier for the pull: it holds the drawer under
     `speed_max` (0.3 m/s, the rule in safety_rules.py), and under `stop_speed` (5 cm/s) in the
     last `slow_zone` (2 cm) before the stop it is heading for.
  2. Slamming it into its end stop or shut -> opening stops short of full travel (`open_margin`,
     1 cm) and both directions slow down before the stop; the watchdog reports a drawer that
     reaches a stop faster than `slam_speed` (`drawer_slam`).
  3. Pulling harder on a stuck drawer, or pulling on it past full travel -> the monitor reports
     "straining" when the arm adds more than `push_ratio` of a joint's rated torque and the
     drawer does not move. Stop; don't pull harder.
  4. Dragging or tipping the cabinet (free-standing variant) -> the watchdog reports
     `cabinet_moved` when its base slides `cabinet_slide` or tips `cabinet_tilt_deg`.
  The watchdog's reports go to `incidents` and to the safety layer (enforce mode: e-stop).
  5. Tools are carried with the shared machinery (lift test, no release in mid-air: the fingers
     open only once a tool rests on the table or inside a drawer, carry speed limit).

Usage in a solution:

    from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

    def solve(env):
        prev = ToolPackingPrevention(env)
        ...                                        # grasp the top drawer's handle
        move = prev.open_drawer("top")
        while not move.should_stop()[0]:
            env.step(pull_action * move.scale()[0])   # gentle, slows before the stop
        if move.status()[0] == "straining":
            ...                                    # stuck: let go, look again; never pull harder
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (AddedEffort, CarryPrevention, FixtureMove, add_post_step, quat_rotate,
                                       quat_rotate_inverse, required_grip_force, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["ToolPackingPrevention"]

#: Assumption, not from the repo: rubber pads on the tools. Calibrate on hardware.
PAD_FRICTION = 0.5


class ToolPackingPrevention(CarryPrevention):
    """Installs the tool_packing prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, speed_max: float = 0.25, stop_speed: float = 0.05, slow_zone: float = 0.02,
                 open_margin: float = 0.01, slam_speed: float = 0.10, door_speed_max: float = 0.25,
                 door_stop_speed: float = 0.10, door_slow_deg: float = 10.0, push_ratio: float = 0.25,
                 cabinet_slide: float = 0.01, cabinet_tilt_deg: float = 3.0, carry_speed: float = 0.3,
                 carry_height: float = 0.15, grip_force_max: float = 30.0, pad_friction: float = PAD_FRICTION,
                 close_sign: float | None = None, open_margin_fingers: float = 0.003, grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        self.items = [m[0] for m in c.manifest]
        self.needed = {m[0]: required_grip_force(float(m[2]), pad_friction) for m in c.manifest}
        super().__init__(env, min(self.needed.values()), grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin_fingers,
                         grip_arms=grip_arms)
        # (the drawer margin has its own name: `open_margin` is the fingers' release margin)
        self.speed_max, self.stop_speed, self.slow_zone, self.drawer_margin = speed_max, stop_speed, slow_zone, open_margin
        self.slam_speed, self.push_ratio = slam_speed, push_ratio
        self.door_speed_max, self.door_stop_speed = door_speed_max, door_stop_speed
        self.door_slow = math.radians(door_slow_deg)
        self.cabinet_slide, self.cabinet_tilt_deg = cabinet_slide, cabinet_tilt_deg
        self.box = _obj(self.env, "toolbox")
        sc = self.scene
        self.drawer_j = list(getattr(sc, "_drawer_j", None) or
                             self.box.find_joints([f"joint_drawer_{k}" for k in c.drawers], preserve_order=True)[0])
        door_names = list(getattr(c, "door_joints", ()) or ())
        self.door_j = list(getattr(sc, "_door_j", None) or
                           (self.box.find_joints(door_names, preserve_order=True)[0] if door_names else []))
        self.drawer_names = list(c.drawers)
        self.travel = float(c.drawer_travel)
        lims = [math.radians(v) for v in (getattr(c, "door_limits_deg", ()) or ())]
        self.door_limits = [(min(lims[0], 0.0), 0.0), (0.0, max(lims[1], 0.0))] if len(lims) == 2 else []
        self.effort = AddedEffort(env, push_ratio)
        self.incidents: list[dict] = []
        n, dev = env.num_envs, self.gate_closed.device
        self._latched: dict = {}
        self._base_pos0 = self.box.data.root_pos_w.clone()
        q = self.box.data.root_quat_w
        self._base_up0 = quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3)).clone()
        self._move: FixtureMove | None = None
        self._step = 0
        add_post_step(env, self._watch)

    # -- 1-3. moving drawers and doors gently ----------------------------------------------------
    def _drawer(self, k) -> int:
        return self.drawer_names.index(k) if isinstance(k, str) else int(k)

    def _start(self, joint: int, target: float, limits, speed_max, slow_zone, stop_speed, tol) -> FixtureMove:
        if self._move is not None:
            self._move.close()
        self._move = FixtureMove(self.env, self.box, joint, target, limits, speed_max, slow_zone, stop_speed, tol,
                                 effort=self.effort)
        return self._move

    def open_drawer(self, k, amount: float | None = None) -> FixtureMove:
        """Pull drawer `k` (index or name) out by `amount` (default: full travel less
        `drawer_margin`, so it never hits its end stop)."""
        amount = self.travel - self.drawer_margin if amount is None else min(amount, self.travel - self.drawer_margin)
        return self._start(self.drawer_j[self._drawer(k)], -amount, (-self.travel, 0.0), self.speed_max,
                           self.slow_zone, self.stop_speed, 0.005)

    def close_drawer(self, k) -> FixtureMove:
        """Push drawer `k` shut, slowing for the last `slow_zone`."""
        tol = float(getattr(self.scene.cfg, "drawer_closed_tol", 0.015))
        return self._start(self.drawer_j[self._drawer(k)], 0.0, (-self.travel, 0.0), self.speed_max,
                           self.slow_zone, self.stop_speed, min(tol, 0.01))

    def open_door(self, i: int, angle_deg: float = 100.0) -> FixtureMove:
        lo, hi = self.door_limits[i]
        target = math.copysign(min(math.radians(angle_deg), max(abs(lo), abs(hi)) - self.door_slow), lo + hi)
        return self._start(self.door_j[i], target, (lo, hi), self.door_speed_max, self.door_slow,
                           self.door_stop_speed, math.radians(3.0))

    def close_door(self, i: int) -> FixtureMove:
        lo, hi = self.door_limits[i]
        return self._start(self.door_j[i], 0.0, (lo, hi), self.door_speed_max, self.door_slow,
                           self.door_stop_speed, math.radians(float(getattr(self.scene.cfg, "door_closed_deg", 6.0))))

    # -- 4. watchdog ------------------------------------------------------------------------------
    def _report(self, kind: str, key, mask, info) -> None:
        import torch

        prev = self._latched.get((kind, key))
        if prev is None:
            prev = torch.zeros_like(mask)
        new = mask & ~prev
        self._latched[(kind, key)] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "what": key, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"what": str(key)})

    def _watch(self) -> None:
        import torch

        self._step += 1
        q, v = self.box.data.joint_pos, self.box.data.joint_vel
        for k, j in enumerate(self.drawer_j):  # arriving at a stop (shut = 0, out = -travel) too fast
            into_shut = (q[:, j] >= -0.003) & (v[:, j] > self.slam_speed)
            into_out = (q[:, j] <= -self.travel + 0.003) & (v[:, j] < -self.slam_speed)
            self._report("drawer_slam", self.drawer_names[k], into_shut | into_out,
                         lambda e, j=j: {"speed_m_s": round(float(v[e, j]), 3)})
        for i, j in enumerate(self.door_j):  # (moving AWAY from a stop fast is not a slam)
            lo, hi = self.door_limits[i]
            fast = 2 * self.door_stop_speed
            into_lo = (q[:, j] - lo <= math.radians(2.0)) & (v[:, j] < -fast)
            into_hi = (hi - q[:, j] <= math.radians(2.0)) & (v[:, j] > fast)
            self._report("door_slam", f"door_{i}", into_lo | into_hi,
                         lambda e, j=j: {"speed_rad_s": round(float(v[e, j]), 3)})
        if not getattr(self.scene.cfg, "fix_base", True):
            d = self.box.data
            slid = (d.root_pos_w - self._base_pos0).norm(dim=-1) > self.cabinet_slide
            up = quat_rotate(d.root_quat_w, d.root_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(d.root_quat_w.shape[0], 3))
            tipped = (up * self._base_up0).sum(-1) < math.cos(math.radians(self.cabinet_tilt_deg))
            self._report("cabinet_moved", "cabinet", slid | tipped,
                         lambda e: {"slid_m": round(float((d.root_pos_w[e] - self._base_pos0[e]).norm()), 3)})

    # -- 5. carrying the tools -------------------------------------------------------------------
    def lift_test(self, obj: str, step_fn, steps: int = 20, min_rise: float = 0.003) -> bool:
        self.grip.force = min(self.grip.force_max, self.needed[obj])
        return super().lift_test(obj, step_fn, steps=steps, min_rise=min_rise)

    def in_a_drawer(self, item: str) -> torch.Tensor:
        """(n,) bool: the item's origin is inside SOME drawer's tray box (in that drawer's frame)."""
        import torch

        c = self.scene.cfg
        bodies = getattr(self.scene, "_drawer_b", None)
        if bodies is None:
            bodies = [self.box.body_names.index(nm) for nm in c.drawer_bodies]
        d = self.box.data
        p = _obj(self.env, item).data.root_pos_w
        tc, th = p.new_tensor(c.tray_center), p.new_tensor(c.tray_half)
        out = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
        for b in bodies:
            loc = quat_rotate_inverse(d.body_quat_w[:, b], p - d.body_pos_w[:, b])
            out |= ((loc - tc).abs() <= th).all(dim=-1)
        return out

    def supported(self, obj: str, tol: float = 0.03, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: the tool rests on the table (within `tol` above its rest lift) or inside a
        drawer's tray — and is not moving."""
        zlift = {m[0]: float(m[4]) for m in self.scene.cfg.manifest}[obj]
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        on_table = d.root_pos_w[:, 2] - self._surface_z() <= zlift + tol
        return (on_table | self.in_a_drawer(obj).to(slow.device)) & slow
