"""Bulb — preventing a shattered bulb.

Safety review row:
    What could go wrong: the arm may add excessive pressure, causing the bulb to shatter.
    How to prevent:      use minimal pressure so the bulb doesn't break.
    Category:            3 — Environment.

How the bulb is handled in this task (scenes/bulb_assembly.py):
  - The bulb (≈50 g) is NOT welded to the hand. The fingers pinch its glass belly (48 mm wide,
    ~55 mm above the cap) and hold it by friction (glass μ = 0.3).
  - Screwing it in ALSO goes through that friction: the turning torque reaches the bulb only
    through the pads on the glass. So the grip must be firm enough to turn the bulb, yet the glass
    is exactly what breaks when squeezed. "Minimal pressure" is a balance, not just a low number.
  - The thread is slick (μ = 0.01); the bulb seats at 27 mm above the socket origin, starts to glow
    at 32 mm (first contact on the thread), and the bore mouth is at 38.5 mm.

Ways the glass gets broken, and the prevention for each:

  1. Squeezing too hard → minimal, ADAPTIVE grip. Start at the least force that holds the bulb's
     weight (~2 N); raise it only when the bulb is seen slipping as the hand turns it; never above
     `glass_force_max`. Every new grasp starts minimal again.
  2. Hitting the glass on the socket → slow approach: once the held bulb is near a socket mouth
     the arm is held to `approach_speed`.
  3. Pressing a crooked bulb into the thread (cross-threading loads the glass) → `aligned()` checks
     the bulb is on the socket axis and upright before pressing; `ScrewMonitor` reports "binding"
     when the arm pushes hard and the bulb does not advance — back off, don't press harder.
  4. Over-tightening (twisting the glass off its cap) → `ScrewMonitor` says stop the moment the
     bulb is seated, or when it will not turn even at the maximum allowed grip.
  5. Dropping it → the shared carry machinery: lift test before carrying, release gate (the fingers
     cannot open unless the bulb is lying on the table or threaded into a socket), carry low/slow.

Usage in a solution:

    from robobench.suites.assembly.prevention.bulb import BulbPrevention

    def solve(env):
        prev = BulbPrevention(env)                       # starts at ~2 N, glass cap 15 N
        ...                                              # pinch the glass belly
        if not prev.lift_test("bulb_0", lambda: env.step(small_lift)):
            ...                                          # re-grasp; do not carry
        ...                                              # carry low; slows itself near the socket
        ok, info = prev.aligned(0)
        if not ok.all():
            ...                                          # correct the pose; don't press
        screw = prev.screw(0)
        while not screw.should_stop()[0]:
            env.step(turn_and_press_action)              # grip rises only if the bulb slips
            ...                                          # wrist out of travel: open (allowed once
            prev.mark_held("bulb_0")                     # threaded), rotate back, close, mark_held
        if screw.status()[0] != "seated":
            ...                                          # "binding" / "slipping_at_cap": back off

Re-gripping is normal: from first thread contact (32 mm) to seated (27 mm) is ~1.4 turns of an E27
thread (~3.6 mm per turn), more than a wrist turns. While the fingers are open the slip check is
off (the wrist turning back is not slip), and the grip the bulb needed is kept for the re-grip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

from robobench.core.prevention import (CarryPrevention, InsertionMonitor, TurnMonitor, quat_rotate,
                                       quat_rotate_inverse, required_grip_force, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

#: From the scene and its configs: bulb mass (scene cfg, informational — the USD carries the real
#: value), the 48 mm glass belly (assembly/configs/envs.py), the pinch height (rl/tasks/bulb.py).
BULB_MASS = 0.05  # kg
GLASS_RADIUS = 0.024  # m
GRASP_Z = 0.055  # m above the cap bottom (the bulb origin)
#: Assumptions, not from the repo — calibrate on real bulbs before trusting them:
GLASS_FORCE_MAX = 15.0  # N per finger; matches the safety rule's grip cap for this task
SEAT_TORQUE = 0.1  # N·m to turn a bulb home against its contact spring


def required_twist_grip(torque: float, friction: float, radius: float = GLASS_RADIUS,
                        safety_factor: float = 1.5, fingers: int = 2) -> float:
    """Grip force (per finger) for pad friction to turn a round part: each pad carries friction
    `μ F` at `radius`, so `fingers × μ × F × radius ≥ safety_factor × torque`."""
    if friction <= 0 or radius <= 0 or fingers <= 0:
        raise ValueError("friction, radius and fingers must be positive")
    return safety_factor * torque / (fingers * friction * radius)


class BulbPrevention(CarryPrevention):
    """Installs the bulb prevention on a live env and exposes the per-step checks."""

    def __init__(self, env: BaseEnv, glass_force_max: float = GLASS_FORCE_MAX,
                 seat_torque: float = SEAT_TORQUE, glass_radius: float = GLASS_RADIUS,
                 carry_speed: float = 0.25, carry_height: float = 0.10, approach_speed: float = 0.03,
                 approach_height: float = 0.03, approach_radius: float = 0.03,
                 align_xy: float = 0.003, align_deg: float = 5.0, close_sign: float | None = None,
                 open_margin: float = 0.003) -> None:
        c = env.scene.cfg
        self.mu = float(getattr(c, "bulb_glass_friction", 0.3))
        self.hold_force = required_grip_force(BULB_MASS, self.mu)  # weight only: the starting grip
        self.turn_force = required_twist_grip(seat_torque, self.mu, glass_radius)
        self.approach_speed, self.approach_height = approach_speed, approach_height
        self.approach_radius, self.align_xy, self.align_deg = approach_radius, align_xy, align_deg
        super().__init__(env, self.hold_force, glass_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin)
        self.speed.add_limit(self._approach_limit)

    # -- 1. minimal, adaptive grip --------------------------------------------------------------
    def grip_report(self) -> dict:
        """Starting grip, the grip turning the bulb home should need, and the glass cap (per finger)."""
        return {
            "set_N": round(self.grip.force, 2),
            "start_N_hold_weight": round(self.hold_force, 2),
            "expected_N_to_seat": round(self.turn_force, 2),
            "glass_cap_N": self.grip.force_max,
            "can_seat_under_cap": self.turn_force <= self.grip.force_max,
        }

    def relax_grip(self) -> None:
        """Back to the minimal (weight-holding) grip — done automatically on every release."""
        self.grip.force = self.hold_force

    def released(self, obj: str, envs=None) -> None:
        """Back to the minimal grip once nothing is held — unless the bulb was let go part-way
        into the thread (a re-grip): then the grip it needed to turn is kept for the next close."""
        super().released(obj, envs)
        if bool(self.carrying().any()):
            return
        i = int(obj.split("_")[1])
        mid_screw = self.threaded(i) & ~self.seated(i)
        if not bool(mid_screw.any()):
            self.relax_grip()  # put down or seated: the next grasp starts minimal again

    # -- socket geometry ------------------------------------------------------------------------
    def _sockets(self) -> list:
        socks = getattr(self.scene, "sockets", None)
        if socks:
            return list(socks)
        arts = getattr(self.env.iscene, "articulations", {}) or {}
        n = int(getattr(self.scene.cfg, "num_pairs", 1))
        return [arts[f"socket_{k}"] for k in range(n)]

    def _nearest(self, i: int):
        """Bulb `i` against its NEAREST socket, as the scene measures it: (offset (n, 3) of the bulb
        origin in that socket's frame, the socket's world up-axis (n, 3))."""
        import torch

        bulb = _obj(self.env, f"bulb_{i}").data
        offs, ups = [], []
        for s in self._sockets():
            q = s.data.root_quat_w
            offs.append(quat_rotate_inverse(q, bulb.root_pos_w - s.data.root_pos_w))
            ups.append(quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3)))
        offs, ups = torch.stack(offs, 1), torch.stack(ups, 1)  # (n, S, 3)
        near = offs[..., :2].norm(dim=-1).argmin(dim=1)
        rows = torch.arange(offs.shape[0], device=offs.device)
        return offs[rows, near], ups[rows, near]

    def tilt_deg(self, i: int = 0) -> torch.Tensor:
        """(n,) angle between bulb `i`'s screw axis (local +z) and its nearest socket's axis."""
        import torch

        bulb = _obj(self.env, f"bulb_{i}").data
        _, up = self._nearest(i)
        axis = quat_rotate(bulb.root_quat_w, up.new_tensor((0.0, 0.0, 1.0)).expand_as(up))
        return torch.rad2deg(torch.arccos((axis * up).sum(-1).clamp(-1.0, 1.0)))

    def height(self, i: int = 0) -> torch.Tensor:
        """(n,) bulb `i`'s origin (its lowest point) above its nearest socket's origin (m)."""
        return self._nearest(i)[0][:, 2]

    def off_axis(self, i: int = 0) -> torch.Tensor:
        return self._nearest(i)[0][:, :2].norm(dim=-1)

    def threaded(self, i: int = 0) -> torch.Tensor:
        """(n,) bool: bulb `i` is on the thread (down to where the glow starts), on axis and upright
        within the scene's tolerances — the thread now holds it, so it may be let go."""
        c = self.scene.cfg
        return ((self.height(i) <= c.light_start_z + 0.001) & (self.off_axis(i) <= c.align_xy)
                & (self.tilt_deg(i) <= c.align_axis_deg))

    def seated(self, i: int = 0) -> torch.Tensor:
        """(n,) bool: bulb `i` is screwed home (the scene's own seated test)."""
        c = self.scene.cfg
        return ((self.height(i) <= c.seat_z) & (self.off_axis(i) <= c.align_xy)
                & (self.tilt_deg(i) <= c.align_axis_deg))

    # -- 2. slow approach to the socket ---------------------------------------------------------
    def near_socket(self, i: int = 0) -> torch.Tensor:
        """(n,) bool: bulb `i` is close above a socket mouth — where the glass can strike it."""
        c = self.scene.cfg
        h = self.height(i)
        return (self.off_axis(i) <= self.approach_radius) & (h <= c.socket_opening_z + self.approach_height)

    def _approach_limit(self):
        import torch

        n = int(getattr(self.scene.cfg, "num_pairs", 1))
        near = torch.zeros_like(self.gate_closed)
        for i in range(n):
            near |= self.near_socket(i).to(near.device)
        slow = near & self.carrying()
        return torch.where(slow, torch.full_like(slow, self.approach_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 3. aligned before pressing -------------------------------------------------------------
    def aligned(self, i: int = 0):
        """(ok (n,) bool, info): bulb `i` is on its socket's axis within `align_xy` and upright
        within `align_deg` — tighter than the scene's seated tolerance, because a crooked start is
        what cross-threads and loads the glass. Check before pressing it into the thread."""
        off, tilt = self.off_axis(i), self.tilt_deg(i)
        ok = (off <= self.align_xy) & (tilt <= self.align_deg)
        info = {"bulb": f"bulb_{i}", "off_axis_mm": [round(float(x) * 1e3, 2) for x in off],
                "tilt_deg": [round(float(x), 2) for x in tilt]}
        return ok, info

    # -- 4. screw without over-tightening -------------------------------------------------------
    def screw(self, i: int = 0, **kw) -> ScrewMonitor:
        """Start a `ScrewMonitor` on bulb `i` (call once the bulb sits aligned on the socket).
        Closes the previous one, so an old monitor never keeps raising the grip on the glass."""
        old = getattr(self, "_screw", None)
        if old is not None:
            old.close()
        self._screw = ScrewMonitor(self, i, **kw)
        return self._screw

    # -- 5. never let go in the air -------------------------------------------------------------
    def supported(self, obj: str, still: float = 0.02, lying_deg: float = 60.0) -> torch.Tensor:
        """(n,) bool: the bulb is LYING on the table (axis at least `lying_deg` off vertical), or
        threaded into a socket — and not moving. A bulb standing on its cap is not supported: it is
        top-heavy, and let go it tips over onto its glass."""
        i = int(obj.split("_")[1])
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        axis = quat_rotate(d.root_quat_w, d.root_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(d.root_pos_w.shape[0], 3))
        flat = axis[:, 2].abs() <= math.cos(math.radians(lying_deg))
        # lying, the bulb's origin settles ~5 mm up (it is dropped from bulb_init_z at reset)
        low = d.root_pos_w[:, 2] - self._surface_z() < self.scene.cfg.bulb_init_z + 0.006
        return ((low & flat) | self.threaded(i).to(low.device)) & slow


class ScrewMonitor:
    """Watches bulb `i` being screwed in. Status per env:

      "seated"           home — STOP turning now (more is how the glass twists off its cap)
      "binding"          the arm pushes hard (over `bind_effort_ratio` of a joint's rating) but the
                         bulb does not advance: crooked or cross-threaded — back off and re-align
      "slipping_at_cap"  the hand turns but the bulb does not, even at the glass grip cap — stop;
                         squeezing harder is exactly what must not happen
      "turning"          keep going

    Rotational slip raises the grip one `boost` step at a time — the "minimal pressure" part: force
    grows only when the bulb proves it needs it. Slip is judged only while the fingers hold the bulb
    and it is "turning": not while re-gripping (open fingers turning back), and not after a stop.
    `close()` when done (`prev.screw()` closes the previous monitor itself)."""

    def __init__(self, prev: BulbPrevention, i: int = 0, min_rate: float = 0.2,
                 slip_ratio: float = 0.5, slip_time: float = 0.05, boost: float = 1.25,
                 bind_effort_ratio: float = 0.5, bind_window_s: float = 0.5,
                 bind_min_progress: float = 0.0002) -> None:
        self.prev, self.i, self.env = prev, i, prev.env
        _, up = prev._nearest(i)
        remaining = (prev.height(i) - prev.scene.cfg.seat_z).clamp_min(0.0)
        self.bind = InsertionMonitor(self.env, f"bulb_{i}", axis=-up, seat_depth=remaining,
                                     effort_ratio=bind_effort_ratio, window_s=bind_window_s,
                                     min_progress=bind_min_progress)
        self.turn = TurnMonitor(self.env, f"bulb_{i}", lambda: prev._nearest(i)[1], prev.grip,
                                active_fn=self._slip_check_on, min_rate=min_rate,
                                slip_ratio=slip_ratio, slip_time=slip_time, boost=boost)

    @property
    def slipping_at_cap(self):
        return self.turn.slipping_at_cap

    def turn_rates(self):
        """(hand, bulb) angular speeds about the socket axis (rad/s), each (n,)."""
        return self.turn.rates()

    def _slip_check_on(self) -> list[bool]:
        held = [o == f"bulb_{self.i}" for o in self.prev._held_obj]
        return [h and s == "turning" for h, s in zip(held, self.status())]

    def status(self) -> list[str]:
        seated = self.prev.seated(self.i)
        out = []
        for e in range(self.env.num_envs):
            if bool(seated[e]):
                out.append("seated")
            elif bool(self.turn.slipping_at_cap[e]):
                out.append("slipping_at_cap")
            elif bool(self.bind.stalled[e]):
                out.append("binding")
            else:
                out.append("turning")
        return out

    def should_stop(self) -> list[bool]:
        """Per env: stop turning (seated, binding, or slipping at the grip cap)."""
        return [s != "turning" for s in self.status()]

    def reset(self) -> None:
        """After backing off and re-aligning."""
        self.turn.slipping_at_cap.zero_()
        self.turn._count.zero_()
        self.bind.reset()

    def close(self) -> None:
        """Stop watching (unhooks the slip check and the binding monitor)."""
        self.turn.close()
        self.bind.close()
