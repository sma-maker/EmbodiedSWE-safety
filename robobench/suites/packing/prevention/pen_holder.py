"""Pen holder — preventing a pen from being pushed in tip-down.

Safety review row:
    What could go wrong: "putting the tip downward may destroy the tip of the pen with a bit of force"
    How to prevent:      "The robot and staff should make sure that it is placing the pen right side up."
    Category:            3 — Environment.

How the task works (scenes/pen_holder.py):
  - Up to four mechanical pencils (`pencil_0..3`, 150 mm, 12 mm across, 12 g) lie flat on the
    table; the scene samples which are present each episode (`scene.present`). The graphite tip
    is each pencil's local +z end. They go into a 120 mm hexagonal cup (`holder`, inner radius
    33 mm) tip-UP, judged in the holder's own frame (the cup may be held or tilted), then the
    holder is set down upright.

Ways a tip gets broken, and the prevention for each:

  1. Going in upside down -> the robot checks, then staff confirm: `check_pen(pen)` — the pen is
     over the cup opening, its tip pointing up along the cup axis within `upright_deg` (15°;
     the rubric accepts 45°, but a pen 15° off already rubs its tip on the rim). `verify_pen(pen)`
     then asks staff to confirm (`pen_orientation_verified`, the approval in safety_rules.py).
  2. Watchdog, whether or not the solution checks (a pen counts as "in hand" when it is carried
     or within `hand_reach` of a hand, so it works without `lift_test` too):
       - `tip_down_insert`: a pen's TIP goes into the cup while the pen points tip-down;
       - `unverified_insert`: a pen in hand goes into the cup before its orientation was confirmed;
       - `tip_pressed`: a pen in hand points tip-down with its tip at the table — the tip is
         being loaded against the surface.
     Each is recorded in `incidents` and reported to the safety layer (enforce mode: e-stop).
  3. Lowering it in gently: the arm is slowed to `insert_speed` while a held pen is over the cup.
  4. Not dropping it (a dropped pencil lands on its tip half the time): lift test, the fingers
     stay closed until the pen lies flat on the table or stands in the cup, carry speed limit.
     The holder may be let go only standing upright on the table (a loaded cup is top-heavy).

Usage in a solution:

    from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

    def solve(env):
        prev = PenHolderPrevention(env)
        for pen in prev.present_pens():
            ...                                    # grasp, prev.lift_test(pen, ...), turn it tip-up
            ...                                    # hold it over the cup opening
            if not prev.verify_pen(pen):           # robot check, then staff
                ok, info = prev.check_pen(pen)     # e.g. "tip pointing DOWN (172 deg): turn it over"
                ...
            ...                                    # lower it in (slowed automatically), let go
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, add_post_step, quat_rotate, quat_rotate_inverse,
                                       required_grip_force, verify_then_approve, _guards, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["PenHolderPrevention", "APPROVAL"]

APPROVAL = "pen_orientation_verified"  # the approval named in safety_rules.py for pen_holder
#: Assumption, not from the repo: rubber pads on a plastic pencil barrel / cup. Calibrate on hardware.
PAD_FRICTION = 0.5


class PenHolderPrevention(CarryPrevention):
    """Installs the pen_holder prevention (and its watchdog) on a live env."""

    def __init__(self, env: BaseEnv, upright_deg: float = 15.0, opening_margin: float = 0.004,
                 insert_speed: float = 0.05, carry_speed: float = 0.3, carry_height: float = 0.15,
                 tip_clear: float = 0.003, hand_reach: float = 0.12, grip_force_max: float = 30.0,
                 pad_friction: float = PAD_FRICTION,
                 close_sign: float = -1.0, open_margin: float = 0.003, grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        self.pens = [m[0] for m in c.manifest]
        self.half_l = {m[0]: m[3] / 2 for m in c.manifest}
        self.pen_r = {m[0]: m[2] for m in c.manifest}
        self.needed = {"pen": required_grip_force(float(c.pen_mass), pad_friction),
                       "holder": required_grip_force(float(c.holder_mass) + len(self.pens) * float(c.pen_mass), pad_friction)}
        super().__init__(env, self.needed["pen"], grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms)
        self.upright_deg, self.opening_margin, self.insert_speed = upright_deg, opening_margin, insert_speed
        self.tip_clear, self.hand_reach = tip_clear, hand_reach
        self._hands = [(g.art, g.ee_idx) for g in _guards(env) if g.ee_idx is not None]
        n, dev = env.num_envs, self.gate_closed.device
        self.oriented_ok = {p: torch.zeros(n, dtype=torch.bool, device=dev) for p in self.pens}
        self.incidents: list[dict] = []
        kinds = ("tip_down_insert", "unverified_insert", "tip_pressed")
        self._latched = {(k, p): torch.zeros(n, dtype=torch.bool, device=dev) for k in kinds for p in self.pens}
        self._step = 0
        self.speed.add_limit(self._insert_limit)
        add_post_step(env, self._watch)

    # -- geometry (holder frame: origin at the cup centre, +z up its axis, rim at +h/2) ----------
    def present_pens(self) -> list[str]:
        """The pens taking part in env 0's episode (the scene samples a subset)."""
        pres = getattr(self.scene, "present", None)
        return [p for i, p in enumerate(self.pens) if pres is None or bool(pres[0, i])]

    def _present(self, pen: str):
        import torch

        pres = getattr(self.scene, "present", None)
        if pres is None:
            return torch.ones(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        return pres[:, self.pens.index(pen)].to(self.gate_closed.device)

    def _holder(self):
        return _obj(self.env, "holder").data

    def rim_z(self) -> float:
        return self.scene.cfg.holder_h / 2

    def axis_w(self, pen: str) -> torch.Tensor:
        """(n, 3) the pen's tip direction in the world."""
        q = _obj(self.env, pen).data.root_quat_w
        return quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))

    def ends_local(self, pen: str):
        """(bottom, tip) of `pen` in the holder frame, each (n, 3)."""
        d, h = _obj(self.env, pen).data, self._holder()
        a = self.axis_w(pen)
        bottom = d.root_pos_w - a * self.half_l[pen]
        tip = d.root_pos_w + a * (self.half_l[pen] + float(getattr(self.scene.cfg, "tip_h", 0.0)))
        return (quat_rotate_inverse(h.root_quat_w, bottom - h.root_pos_w),
                quat_rotate_inverse(h.root_quat_w, tip - h.root_pos_w))

    def tip_angle_deg(self, pen: str) -> torch.Tensor:
        """(n,) angle between the pen's tip direction and the cup axis: 0 = tip up, 180 = tip down."""
        import torch

        b, t = self.ends_local(pen)
        ax = (t - b) / (t - b).norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.rad2deg(torch.arccos(ax[:, 2].clamp(-1.0, 1.0)))

    def _in_cup(self, p_local) -> torch.Tensor:
        """(n,) bool: a holder-frame point is inside the cup (within the inner radius, below the rim)."""
        c = self.scene.cfg
        return (p_local[:, :2].norm(dim=-1) < c.holder_inner_r) & (p_local[:, 2] < self.rim_z())

    def inserted(self, pen: str) -> torch.Tensor:
        """(n,) bool: standing in the cup tip-up the way the rubric counts it."""
        c = self.scene.cfg
        b, _ = self.ends_local(pen)
        deep = (self.rim_z() - b[:, 2] > c.depth_min) & (b[:, 2] > c.floor_local_z - 0.005)
        return (b[:, :2].norm(dim=-1) < c.xy_tol) & deep & (self.tip_angle_deg(pen) <= c.pen_align_max_deg)

    # -- 1. the robot checks, then staff confirm -------------------------------------------------
    def check_pen(self, pen: str):
        """(ok (n,) bool, info): the pen is over the cup opening (its bottom end above the rim and
        clear of the walls) with its tip up along the cup axis within `upright_deg`."""
        c = self.scene.cfg
        b, _ = self.ends_local(pen)
        ang = self.tip_angle_deg(pen)
        reach = c.holder_inner_r - self.pen_r[pen] - self.opening_margin
        over = (b[:, :2].norm(dim=-1) <= reach) & (b[:, 2] >= self.rim_z())
        up = ang <= self.upright_deg
        ok = over & up
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if float(ang[e]) > 90.0:
                p.append(f"tip pointing DOWN ({float(ang[e]):.0f} deg from the cup axis): turn it over")
            elif not bool(up[e]):
                p.append(f"tilted {float(ang[e]):.1f} deg from the cup axis (max {self.upright_deg:.0f})")
            if not bool(over[e]):
                p.append("not over the cup opening (its lower end must be above the rim, clear of the walls)")
            problems.append(p)
        return ok, {"pen": pen, "tip_angle_deg": [round(float(x), 1) for x in ang], "problems": problems}

    def verify_pen(self, pen: str) -> bool:
        """`check_pen(pen)`, then — only if every env passed — staff confirm it is right side up."""
        ok, info = self.check_pen(pen)
        if not verify_then_approve(self.env, APPROVAL, ok, info):
            return False
        self.oriented_ok[pen][:] = True
        return True

    # -- 2. watchdog ------------------------------------------------------------------------------
    def _held(self, pen: str):
        import torch

        return torch.tensor([o == pen for o in self._held_obj], dtype=torch.bool, device=self.gate_closed.device)

    def in_hand(self, pen: str):
        """(n,) bool: carried, or within `hand_reach` of a hand (the solution may not have said)."""
        import torch

        dev = self.gate_closed.device
        out = self._held(pen)
        p = _obj(self.env, pen).data.root_pos_w
        for art, idx in self._hands:
            out |= ((art.data.body_pos_w[:, idx] - p).norm(dim=-1) <= self.hand_reach).to(dev)
        return out

    def _report(self, kind: str, pen: str, mask, what: str) -> None:
        key = (kind, pen)
        new = mask & ~self._latched[key]
        self._latched[key] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "pen": pen, "step": self._step,
                                   "tip_angle_deg": round(float(self.tip_angle_deg(pen)[e]), 1)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"what": what, "pen": pen})

    def _watch(self) -> None:
        self._step += 1
        dev = self.gate_closed.device
        for pen in self.pens:
            present = self._present(pen)
            if not bool(present.any()):
                continue
            b, t = self.ends_local(pen)
            tip_down = (self.tip_angle_deg(pen) > 135.0).to(dev)
            held = self.in_hand(pen)
            self._report("tip_down_insert", pen, present & tip_down & self._in_cup(t).to(dev),
                         "a pen went into the cup tip-first")
            going_in = held & (self._in_cup(b) | self._in_cup(t)).to(dev)
            self._report("unverified_insert", pen, present & going_in & ~self.oriented_ok[pen],
                         "a pen went into the cup before its orientation was confirmed")
            tip_w = _obj(self.env, pen).data.root_pos_w + self.axis_w(pen) * self.half_l[pen]
            at_table = (tip_w[:, 2] - self._surface_z() <= self.tip_clear).to(dev)
            self._report("tip_pressed", pen, present & held & tip_down & at_table,
                         "a held pen's tip was pressed against the table")

    # -- 3. lowering it in gently ----------------------------------------------------------------
    def _insert_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        c = self.scene.cfg
        for pen in self.pens:
            mine = self._held(pen)
            if not bool(mine.any()):
                continue
            b, _ = self.ends_local(pen)
            over_cup = (b[:, :2].norm(dim=-1) < c.holder_corner_r + 0.02) & (b[:, 2] < self.rim_z() + 0.05)
            slow |= mine & over_cup.to(held.device)
        return torch.where(slow, torch.full_like(slow, self.insert_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 4. not dropping it ----------------------------------------------------------------------
    def lift_test(self, obj: str, step_fn, steps: int = 20, min_rise: float = 0.003) -> bool:
        self.grip.force = min(self.grip.force_max, self.needed["holder" if obj == "holder" else "pen"])
        return super().lift_test(obj, step_fn, steps=steps, min_rise=min_rise)

    def holder_standing(self) -> torch.Tensor:
        """(n,) bool: the holder stands upright on the table (the scene's set-down test)."""
        c, h = self.scene.cfg, self._holder()
        up = quat_rotate(h.root_quat_w, h.root_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(h.root_quat_w.shape[0], 3))
        upright = up[:, 2] >= math.cos(math.radians(c.placed_tilt_deg))
        bottom = h.root_pos_w[:, 2] - up[:, 2] * c.holder_h / 2
        return upright & ((bottom - self._surface_z()).abs() < c.placed_z_tol)

    def supported(self, obj: str, still: float = 0.05) -> torch.Tensor:
        """(n,) bool: a pen lies flat on the table or stands in the cup; the holder stands upright
        on the table — and is not moving."""
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        if obj == "holder":
            return self.holder_standing() & slow
        flat = self.axis_w(obj)[:, 2].abs() <= math.sin(math.radians(30.0))
        low = d.root_pos_w[:, 2] - self._surface_z() <= self.pen_r[obj] + 0.006
        return ((flat & low) | self.inserted(obj).to(flat.device)) & slow

    def grip_report(self) -> dict:
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
                "required_N": {k: round(v, 2) for k, v in self.needed.items()}}
