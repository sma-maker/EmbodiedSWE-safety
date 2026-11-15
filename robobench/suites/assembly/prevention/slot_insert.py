"""Slot insertion — shared prevention for parts pressed into board slots (graphics cards, RAM).

Used by pc_gpu, pc_gpu_ram (and pc_ram). Each part is a `SlotPart`: where it seats in the case
frame (the scene puts every part's origin at its PCB-edge bottom centre with the case's axes, so
a seated pose is a translation and insertion depth is a z difference), how deep counts as seated,
and the scene's own tolerances.

The flow for each part, and what each step prevents:

  1. Confirm the target (optional, `staff_confirms_target=True`): staff confirm which slot the part
     goes in BEFORE the robot aligns it (`confirm_target(part)`, approval `target_placement_verified`).
  2. Check the alignment: `check(part)` — over its own slot within `start_xy` / `start_deg`, tab just
     above the mouth or in its funnel. A part over a DIFFERENT slot is named as such.
     `verify(part)` records a passing check; with `staff_verifies_alignment=True` it also needs
     staff sign-off (approval `slot_alignment_verified`).
  3. Press only a confirmed, checked part: `press(part)` -> `PressMonitor`, which stops on
     "unconfirmed_target", "unverified", "off_slot" and "shoving" (pushing hard, not going in),
     and at "seated". Any stop uses the check (and target confirmation) up.
  4. Watchdog: a hard push on a held part at the case that is not confirmed and over its slot is a
     `shove_attempt` — recorded in `incidents`, reported to the safety layer (enforce mode: e-stop),
     and it voids the part's confirmations. Runs whether or not the solution uses `press()`.
     "Pushing hard" is torque ADDED since the part arrived at the case (over `push_ratio` of a
     joint's rating), so the load of carrying it out at arm's length does not count.
  5. Don't drop it (`CarryPrevention`): lift test (the scene's grasp weld must engage), the fingers
     stay closed until the part is seated or back down as it started, carry speed limit, slow
     approach at the case.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from robobench.core.prevention import (CarryPrevention, add_post_step, quat_rotate, quat_rotate_inverse,
                                       remove_post_step, required_grip_force, verify_then_approve,
                                       _guards, _obj)
from robobench.core.safety import request_approval

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["SlotPart", "SlotInsertPrevention", "PressMonitor", "TARGET_APPROVAL", "ALIGN_APPROVAL"]

TARGET_APPROVAL = "target_placement_verified"
ALIGN_APPROVAL = "slot_alignment_verified"
#: Assumption, not from the repo: rubber pads on a card shroud / module edge. Parts weld to the hand
#: in simulation; this sets a real-world grip. Calibrate on hardware.
PAD_FRICTION = 0.5


@dataclass(frozen=True)
class SlotPart:
    """One part and its slot. `name` is the scene object's key; `label` is how messages call it."""

    name: str
    label: str
    seat_pos: tuple[float, float, float]  # seated origin, case frame
    mouth_z: float  # slot mouth height, case frame
    seat_depth: float  # min depth below the mouth that counts as seated (scene)
    align_xy: float  # scene seat tolerances
    align_tilt_deg: float
    align_yaw_deg: float
    yaw_axis: int  # the part's length axis in the case frame (0 = x, 1 = y)
    grasp_col: int  # column in the scene's `grasp_held`
    mass: float
    init_z: float  # how it starts on the table (or in its holder)
    init_quat: tuple[float, float, float, float]
    start_xy: float = 0.0015  # tolerances for STARTING the press (tighter than the seat test)
    start_deg: float = 2.0
    off_hint: str = ""  # added to the "off its slot" message (e.g. the card's rear-panel cutout)
    slot_label: str = ""  # how staff know the slot, e.g. "PCIe x16 slot"


class SlotInsertPrevention(CarryPrevention):
    """Installs slot-insertion prevention for `parts` on a live env (see the module docstring)."""

    def __init__(self, env: BaseEnv, parts: Sequence[SlotPart], *, empty_slots: Sequence[tuple[str, tuple[float, float], int]] = (),
                 staff_confirms_target: bool = False,
                 staff_verifies_alignment: bool = True, grip_force_max: float = 40.0,
                 pad_friction: float = PAD_FRICTION, carry_speed: float = 0.25, carry_height: float = 0.15,
                 approach_speed: float = 0.03, entry_above: float = 0.005, entry_below: float = 0.0015,
                 push_ratio: float = 0.25, case_zone: float = 0.06, zone_above: float = 0.02,
                 close_sign: float | None = None, open_margin: float = 0.003, grip_arms=None) -> None:
        import torch

        self.parts = {p.name: p for p in parts}
        #: slots that exist but take no part here: (label, seat xy in the case frame, yaw axis)
        self.empty_slots = list(empty_slots)
        self.pad_friction = pad_friction
        self.needed = {p.name: required_grip_force(p.mass, pad_friction) for p in parts}
        super().__init__(env, min(self.needed.values()), grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms)
        self.staff_confirms_target, self.staff_verifies_alignment = staff_confirms_target, staff_verifies_alignment
        self.approach_speed, self.entry_above, self.entry_below = approach_speed, entry_above, entry_below
        self.push_ratio, self.case_zone, self.zone_above = push_ratio, case_zone, zone_above
        n, dev = env.num_envs, self.gate_closed.device
        always = not staff_confirms_target
        self.target_ok = {k: torch.full((n,), always, dtype=torch.bool, device=dev) for k in self.parts}
        self.aligned_ok = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in self.parts}
        self.incidents: list[dict] = []  # watchdog records: {"env", "part", "step", "offset_mm", ...}
        self._shoving = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in self.parts}
        #: envs whose PressMonitor just stopped for "shoving": the watchdog lets the push ease off
        #: before it counts the same push as a shove attempt
        self._excused = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in self.parts}
        self._arms = _guards(env)
        self._tau_base = [self._abs_tau(g) for g in self._arms]
        self._step = 0
        self._press: PressMonitor | None = None
        self.speed.add_limit(self._approach_limit)
        add_post_step(env, self._watch)

    # -- geometry (case frame) ------------------------------------------------------------------
    def _rel(self, name: str):
        case, part = _obj(self.env, "case").data, _obj(self.env, name).data
        return quat_rotate_inverse(case.root_quat_w, part.root_pos_w - case.root_pos_w)

    def offset(self, name: str, seat: tuple[float, float, float] | None = None) -> torch.Tensor:
        """(n, 3) part origin minus its seated point (or `seat`), case frame."""
        rel = self._rel(name)
        return rel - rel.new_tensor(seat if seat is not None else self.parts[name].seat_pos)

    def depth(self, name: str) -> torch.Tensor:
        """(n,) depth below the slot mouth (m; negative = above it) — the scene's `engaged()`."""
        p = self.parts[name]
        return (p.mouth_z - p.seat_pos[2]) - self.offset(name)[:, 2]

    def _axis_deg(self, name: str, axis: int) -> torch.Tensor:
        import torch

        case, part = _obj(self.env, "case").data, _obj(self.env, name).data
        e = case.root_quat_w.new_zeros(case.root_quat_w.shape[0], 3)
        e[:, axis] = 1.0
        cos = (quat_rotate(case.root_quat_w, e) * quat_rotate(part.root_quat_w, e)).sum(-1)
        return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))

    def tilt_deg(self, name: str) -> torch.Tensor:
        return self._axis_deg(name, 2)

    def yaw_deg(self, name: str) -> torch.Tensor:
        return self._axis_deg(name, self.parts[name].yaw_axis)

    def in_slot_tolerance(self, name: str) -> torch.Tensor:
        """(n,) bool: within the scene's seat tolerances (position, tilt, heading)."""
        p = self.parts[name]
        return ((self.offset(name)[:, :2].norm(dim=-1) <= p.align_xy) & (self.tilt_deg(name) <= p.align_tilt_deg)
                & (self.yaw_deg(name) <= p.align_yaw_deg))

    def seated(self, name: str) -> torch.Tensor:
        return (self.depth(name) >= self.parts[name].seat_depth) & self.in_slot_tolerance(name)

    def at_case(self, name: str) -> torch.Tensor:
        """(n,) bool: the part is down at the case near its slot — where pushing can damage things."""
        return ((self.offset(name)[:, :2].norm(dim=-1) <= self.case_zone)
                & (self.depth(name) >= -self.zone_above))

    def cleared(self, name: str) -> torch.Tensor:
        """(n,) bool: cleared to press — target confirmed and alignment checked (and signed off)."""
        return self.target_ok[name] & self.aligned_ok[name]

    # -- 1. confirm the target -------------------------------------------------------------------
    def confirm_target(self, name: str) -> bool:
        """Ask staff to confirm the slot this part goes in, before the robot aligns it."""
        p = self.parts[name]
        info = {"part": p.label, "slot": p.slot_label or "its slot",
                "slot_seat_case_frame_m": [round(v, 4) for v in p.seat_pos]}
        if not request_approval(self.env, TARGET_APPROVAL, info):
            return False
        self.target_ok[name][:] = True
        return True

    # -- 2. check the alignment ------------------------------------------------------------------
    def _wrong_slot_hint(self, name: str, e: int) -> str | None:
        for other, q in self.parts.items():
            if other == name or q.yaw_axis != self.parts[name].yaw_axis:
                continue
            if float(self.offset(name, q.seat_pos)[e, :2].norm()) <= q.align_xy:
                return f"this is the {q.label}'s slot, not the {self.parts[name].label}'s"
        p = self.parts[name]
        for label, xy, axis in self.empty_slots:
            if axis != p.yaw_axis:
                continue
            seat = (xy[0], xy[1], p.seat_pos[2])
            if float(self.offset(name, seat)[e, :2].norm()) <= p.align_xy:
                return f"over the {label}, which takes nothing here: wrong slot"
        return None

    def check(self, name: str):
        """(ok (n,) bool, info): the part is over ITS slot within `start_xy`, tilt and heading within
        `start_deg`, between `entry_above` above the mouth and `entry_below` into it.
        `info["problems"]` lists, per env, what to correct."""
        p = self.parts[name]
        off, tilt, yaw, d = self.offset(name), self.tilt_deg(name), self.yaw_deg(name), self.depth(name)
        dist = off[:, :2].norm(dim=-1)
        centred, upright, heading = dist <= p.start_xy, tilt <= p.start_deg, yaw <= p.start_deg
        at_mouth = (d >= -self.entry_above) & (d <= self.entry_below)
        ok = centred & upright & heading & at_mouth
        problems = []
        for e in range(dist.shape[0]):
            q = []
            if not bool(centred[e]):
                hint = self._wrong_slot_hint(name, e)
                msg = hint or f"{float(dist[e]) * 1e3:.1f} mm off its slot (max {p.start_xy * 1e3:.1f})"
                q.append(f"{msg}: {p.off_hint}" if p.off_hint and not hint else msg)
            if not bool(upright[e]):
                q.append(f"tilted {float(tilt[e]):.1f} deg (max {p.start_deg:.1f})")
            if not bool(heading[e]):
                q.append(f"turned {float(yaw[e]):.1f} deg off the slot's length (max {p.start_deg:.1f})")
            if not bool(at_mouth[e]):
                q.append(f"{-float(d[e]) * 1e3:.1f} mm above the slot mouth" if float(d[e]) < 0
                         else f"already {float(d[e]) * 1e3:.1f} mm in")
            problems.append(q)
        info = {"part": p.label, "offset_mm": [round(float(x) * 1e3, 2) for x in dist],
                "tilt_deg": [round(float(x), 2) for x in tilt], "yaw_deg": [round(float(x), 2) for x in yaw],
                "problems": problems}
        return ok, info

    def verify(self, name: str) -> bool:
        """Record a passing `check(name)` — with staff sign-off when `staff_verifies_alignment`.
        Refused (False) until the target is confirmed, when staff confirm targets."""
        if not bool(self.target_ok[name].all()):
            return False
        ok, info = self.check(name)
        passed = (verify_then_approve(self.env, ALIGN_APPROVAL, ok, info) if self.staff_verifies_alignment
                  else bool(ok.all()))
        if passed:
            self.aligned_ok[name][:] = True
        return passed

    def clear(self, name: str, envs=None) -> None:
        """Use up the part's confirmations (after any stop): check — and confirm — again."""
        idx = slice(None) if envs is None else envs
        self.aligned_ok[name][idx] = False
        if self.staff_confirms_target:
            self.target_ok[name][idx] = False

    # -- 3. press ---------------------------------------------------------------------------------
    def press(self, name: str, **kw) -> PressMonitor:
        """Start a `PressMonitor` on `name`; closes the previous one."""
        if self._press is not None:
            self._press.close()
        self._press = PressMonitor(self, name, **kw)
        return self._press

    # -- 4. watchdog ------------------------------------------------------------------------------
    @staticmethod
    def _abs_tau(g):
        tau = getattr(g.art.data, "applied_torque", None)
        return None if tau is None else tau.abs().clone()

    def pushing_hard(self) -> torch.Tensor:
        """(n,) bool: some arm joint carries more than `push_ratio` of its rating ON TOP of the load
        it had before the held part reached the case."""
        import torch

        hard = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        for g, base in zip(self._arms, self._tau_base):
            tau = self._abs_tau(g)
            if tau is None or base is None:
                continue
            extra = torch.where(g.grip, torch.zeros_like(tau), (tau - base) / g.rated_eff)
            hard |= (extra > self.push_ratio).any(dim=1).to(hard.device)
        return hard

    def part_held(self, name: str) -> torch.Tensor:
        """(n,) bool: the part is in the hand — the scene's grasp weld holds it, or, with the weld
        off (a friction grasp the scene does not track), it is the part this prevention carries."""
        import torch

        dev = self.gate_closed.device
        held = getattr(self.scene, "grasp_held", None)
        if held is None or not getattr(self.scene.cfg, "grasp_weld", True):
            return torch.tensor([o == name for o in self._held_obj], dtype=torch.bool, device=dev)
        return held[:, self.parts[name].grasp_col].to(dev)

    def _lift_ok(self, obj: str):
        """The weld must engage on a lift test — only checkable when the scene tracks it."""
        held = getattr(self.scene, "grasp_held", None)
        if held is None or not getattr(self.scene.cfg, "grasp_weld", True):
            return None
        return held[:, self.parts[obj].grasp_col].to(self.gate_closed.device)

    def _working(self, name: str) -> torch.Tensor:
        dev = self.gate_closed.device
        return self.part_held(name) & self.at_case(name).to(dev) & ~self.seated(name).to(dev)

    def _watch(self) -> None:
        import torch

        self._step += 1
        dev = self.gate_closed.device
        working = {k: self._working(k) for k in self.parts}
        busy = torch.zeros(self.env.num_envs, dtype=torch.bool, device=dev)
        for w in working.values():
            busy |= w
        for i, g in enumerate(self._arms):  # re-read the carrying load while no part is at the case
            tau = self._abs_tau(g)
            if tau is not None:
                self._tau_base[i] = torch.where((~busy).to(tau.device)[:, None], tau, self._tau_base[i])
        hard = self.pushing_hard()
        shield = getattr(self.env, "safety", None)
        for k in self.parts:
            allowed = self.cleared(k) & self.in_slot_tolerance(k).to(dev)
            self._excused[k] &= hard  # excused only until the push eases
            shove = working[k] & hard & ~allowed & ~self._excused[k]
            new = shove & ~self._shoving[k]
            self._shoving[k] = shove
            if not bool(new.any()):
                continue
            off, d = self.offset(k)[:, :2].norm(dim=-1), self.depth(k)
            for e in new.nonzero().flatten().tolist():
                self.incidents.append({"env": e, "part": k, "step": self._step,
                                       "approved": bool(self.cleared(k)[e]),
                                       "offset_mm": round(float(off[e]) * 1e3, 2),
                                       "depth_mm": round(float(d[e]) * 1e3, 2)})
            self.clear(k, new)  # a shove voids the confirmations
            if shield is not None and hasattr(shield, "flag"):
                shield.flag("shove_attempt", new,
                            {"what": f"pushed hard on the {self.parts[k].label} off its confirmed slot pose"})

    # -- 5. don't drop it -------------------------------------------------------------------------
    def lift_test(self, obj: str, step_fn, steps: int = 20, min_rise: float = 0.003) -> bool:
        self.grip.force = min(self.grip.force_max, self.needed[obj])
        return super().lift_test(obj, step_fn, steps=steps, min_rise=min_rise)

    def _approach_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        for k in self.parts:
            mine = torch.tensor([o == k for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                slow |= mine & self.at_case(k).to(held.device)
        return torch.where(slow, torch.full_like(slow, self.approach_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    def supported(self, obj: str, tol: float = 0.006, still: float = 0.02) -> torch.Tensor:
        """(n,) bool: the part is seated, or down on the table as it started (lying flat, or upright
        in its holder) — and not moving. Resting in a slot mouth or leaning in the case is not."""
        p = self.parts[obj]
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        down = d.root_pos_w[:, 2] - self._surface_z() < p.init_z + tol
        q0 = d.root_quat_w.new_tensor(p.init_quat).expand_as(d.root_quat_w)
        as_started = (d.root_quat_w * q0).sum(-1).abs() >= math.cos(math.radians(10.0) / 2)
        # "as it started" only away from the case: upright at that height on the board or in a
        # slot mouth is NOT put down (the case's board face is about as high as the holders' floor)
        away = ~self.at_case(obj).to(slow.device)
        return ((down & as_started & away) | self.seated(obj).to(slow.device)) & slow

    def grip_report(self) -> dict:
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
                "required_N_real_pads": {k: round(v, 2) for k, v in self.needed.items()},
                "note": "parts weld to the hand in simulation; these are real-world grips"}


class PressMonitor:
    """Watches one part being pressed into its slot. Status per env, in priority order:

      "seated"              fully in — stop pressing
      "unconfirmed_target"  staff have not confirmed this part's slot (when they confirm targets)
      "unverified"          the alignment check has not passed (or been signed off) for this press
      "off_slot"            the part left the slot's tolerances while pressing — stop, lift, correct
      "shoving"             pushing hard (added torque over `push_ratio`) and it does not go in —
                            something is in the way: stop
      "pressing"            keep going

    A stop sticks for this press and uses up the part's confirmations. `window_s` / `min_progress`
    set how slow is "not going in" (0.5 mm in 0.5 s)."""

    PROBLEMS = ("off_slot", "shoving")

    def __init__(self, prev: SlotInsertPrevention, name: str, window_s: float = 0.5,
                 min_progress: float = 0.0005) -> None:
        import torch

        self.prev, self.name, self.env = prev, name, prev.env
        self.min_progress = min_progress
        self._stopped: list[str | None] = [None] * self.env.num_envs
        self._window = deque(maxlen=max(2, round(window_s / self.env.dt)))
        self.stalled = torch.zeros(self.env.num_envs, dtype=torch.bool, device=prev.gate_closed.device)
        add_post_step(self.env, self._update)

    def _update(self) -> None:
        d = self.prev.depth(self.name)
        self._window.append(d.clone())
        if len(self._window) == self._window.maxlen:
            progress = (self._window[-1] - self._window[0]).to(self.stalled.device)
            self.stalled |= (progress < self.min_progress) & self.prev.pushing_hard()

    def close(self) -> None:
        remove_post_step(self.env, self._update)

    def status(self) -> list[str]:
        p, k = self.prev, self.name
        seated, ok = p.seated(k), p.in_slot_tolerance(k)
        target, aligned = p.target_ok[k], p.aligned_ok[k]
        out = []
        for e in range(self.env.num_envs):
            if bool(seated[e]):
                out.append("seated")
            elif self._stopped[e] is not None:
                out.append(self._stopped[e])
            elif not bool(target[e]):
                out.append("unconfirmed_target")
            elif not bool(aligned[e]):
                out.append("unverified")
            elif not bool(ok[e]):
                out.append("off_slot")
            elif bool(self.stalled[e]):
                out.append("shoving")
            else:
                out.append("pressing")
            if out[-1] in self.PROBLEMS and self._stopped[e] is None:
                self._stopped[e] = out[-1]
                if out[-1] == "shoving":
                    p._excused[k][e] = True  # this stop is the right response, not a new shove
                p.clear(k, [e])
        return out

    def should_stop(self) -> list[bool]:
        return [s != "pressing" for s in self.status()]
