"""Nut thread — preventing a dented nut or a damaged thread.

Safety review row (as written):
    What could go wrong: "Nut may become dented from misuse of alien bolt"
    How to prevent:      "place key carefully in the right position to drive the alien bolt
                          without causing damage"
    Category:            3 — Environment.
This task has no key and no Allen bolt (the wording matches the allen_bolt row): the robot picks up
an M16 nut and screws it by hand onto a FIXED M16 bolt. The row's intent, applied to this task:
put the nut carefully in the right position before driving it, and drive it without damage.

How the task works (scenes/nut_thread_assembly.py):
  - The bolt is fixed to the world; the nut (30 g) is a free body held only by pad friction (the
    robot configs raise its friction to 0.4 so the fingers can turn it).
  - In the bolt's frame, the nut origin rests on the bolt top at ~25 mm (`bolt_height`), is seated
    at <= `seat_z` (12 mm), and bottoms out at ~1 mm; the gripper pads hit the bolt head below
    ~4 mm. Seated also needs it within `align_xy` of the axis and `align_axis_deg` upright.
  - Coarse M16 thread: 2 mm per turn, so one turn lowers the nut 2 mm.

How a nut or a thread gets damaged, and the prevention for each:

  1. Starting crooked (cross-threading strips both threads) -> `check_nut(i)` before turning: the
     nut must sit on the bolt top, centred within 1.5 mm and upright within 3°. `drive(i)` reports
     "unchecked" until that check has passed for that nut.
  2. Forcing it -> `NutDrive` reports "cross_threaded" when the arm pushes hard (over
     `drive_effort_ratio` of a joint's rating) and the nut does not go down, and "off_axis" if it
     tips or walks off the axis while turning. Back off and check again.
  3. Turning the wrong way (unscrewing, or riding the thread start until it jumps) -> "wrong_direction"
     when the nut rises while the hand turns.
  4. Over-tightening, or the pads grinding the bolt head -> "seated": stop at the scene's seat
     depth, well before bottoming out.
  5. Squeezing too hard / slipping -> grip from the nut's weight; raised a step at a time only
     while the nut spins in the fingers (shared `TurnMonitor`), never above `grip_force_max`;
     "slipping_at_cap" past it.
  6. Dropping it (how nuts get dented) -> lift test, the fingers stay closed until the nut lies on
     the table or is threaded on, carry speed limit, slow approach just above the bolt top.

Usage in a solution:

    from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

    def solve(env):
        prev = NutThreadPrevention(env)
        ...                                        # grasp the nut
        if not prev.lift_test("nut_0", lambda: env.step(small_lift)):
            ...                                    # re-grasp; do not carry
        ...                                        # set it on the bolt top
        ok, info = prev.check_nut(0)
        if not ok.all():
            ...                                    # info["problems"]: correct, check again
        drive = prev.drive(0)
        while not drive.should_stop()[0]:
            env.step(turn_clockwise_and_press)     # the wrist runs out of travel: re-grip
            ...                                    # open, rotate back, close, then:
            prev.mark_held("nut_0")                # the nut is in the fingers again
        if drive.status()[0] != "seated":
            ...                                    # back off, prev.check_nut(0) again, new drive

Re-gripping is normal here: seating takes ~6.5 turns (25 mm -> 12 mm at 2 mm per turn), more than
any wrist turns. While the fingers are open the slip check is off; `mark_held` turns it back on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, InsertionMonitor, TurnMonitor, add_post_step,
                                       quat_rotate, quat_rotate_inverse, remove_post_step,
                                       required_grip_force, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["NutThreadPrevention", "NutDrive"]

THREAD_PITCH = 0.002  # M16 coarse (ISO 261): 2 mm per turn


class NutThreadPrevention(CarryPrevention):
    """Installs the nut_thread prevention on a live env and exposes the per-step checks.

    `align_xy` / `align_deg` are the tolerances for STARTING the nut on the thread: 1.5 mm and 3°,
    far tighter than the scene's 15 mm / 10° seat test, because that is what keeps it from
    cross-threading. `entry_tol` is how far from the bolt top the nut may sit to count as "about to
    thread". `drive_effort_ratio` is the share of a joint's rated torque that counts as pushing hard."""

    def __init__(self, env: BaseEnv, grip_force_max: float = 30.0, carry_speed: float = 0.25,
                 carry_height: float = 0.10, approach_speed: float = 0.03, approach_height: float = 0.02,
                 approach_radius: float = 0.02, align_xy: float = 0.0015, align_deg: float = 3.0,
                 entry_tol: float = 0.004, drive_effort_ratio: float = 0.5, close_sign: float | None = None,
                 open_margin: float = 0.003, grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        self.n_pairs = int(getattr(c, "num_pairs", 1))
        self.needed = required_grip_force(float(c.nut_mass), float(c.nut_friction))
        super().__init__(env, self.needed, grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms)
        self.approach_speed, self.approach_height = approach_speed, approach_height
        self.approach_radius, self.align_xy, self.align_deg = approach_radius, align_xy, align_deg
        self.entry_tol, self.drive_effort_ratio = entry_tol, drive_effort_ratio
        self.entry_z = float(c.bolt_height)  # nut origin height resting on the bolt top
        self.checked = torch.zeros(env.num_envs, self.n_pairs, dtype=torch.bool, device=self.gate_closed.device)
        #: per env, per nut: the bolt the passing check was made on (-1 = none)
        self.checked_bolt = torch.full((env.num_envs, self.n_pairs), -1, dtype=torch.long, device=self.gate_closed.device)
        self._drive: NutDrive | None = None
        self.speed.add_limit(self._approach_limit)

    # -- geometry: nut i against its nearest bolt (the scene's own frame) -----------------------
    def nearest_bolt(self, i: int = 0) -> torch.Tensor:
        """(n,) index of the bolt nearest nut `i` (horizontally, in each bolt's frame)."""
        return self._nearest(i, with_index=True)[2]

    def _nearest(self, i: int, with_index: bool = False):
        """(offset (n, 3) of nut `i` in its nearest bolt's frame, that bolt's up axis (n, 3))."""
        import torch

        nut = _obj(self.env, f"nut_{i}").data
        offs, ups = [], []
        for b in range(self.n_pairs):
            d = _obj(self.env, f"bolt_{b}").data
            offs.append(quat_rotate_inverse(d.root_quat_w, nut.root_pos_w - d.root_pos_w))
            ups.append(quat_rotate(d.root_quat_w, d.root_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(d.root_quat_w.shape[0], 3)))
        offs, ups = torch.stack(offs, 1), torch.stack(ups, 1)
        near = offs[..., :2].norm(dim=-1).argmin(dim=1)
        rows = torch.arange(offs.shape[0], device=offs.device)
        if with_index:
            return offs[rows, near], ups[rows, near], near
        return offs[rows, near], ups[rows, near]

    def height(self, i: int = 0) -> torch.Tensor:
        """(n,) nut `i`'s origin height above its nearest bolt's origin (m)."""
        return self._nearest(i)[0][:, 2]

    def off_axis(self, i: int = 0) -> torch.Tensor:
        return self._nearest(i)[0][:, :2].norm(dim=-1)

    def bolt_up(self, i: int = 0) -> torch.Tensor:
        return self._nearest(i)[1]

    def tilt_deg(self, i: int = 0) -> torch.Tensor:
        """(n,) angle between nut `i`'s screw axis (local +z) and its nearest bolt's axis."""
        import torch

        up = self.bolt_up(i)
        q = _obj(self.env, f"nut_{i}").data.root_quat_w
        axis = quat_rotate(q, up.new_tensor((0.0, 0.0, 1.0)).expand_as(up))
        return torch.rad2deg(torch.arccos((axis * up).sum(-1).clamp(-1.0, 1.0)))

    def on_axis(self, i: int = 0) -> torch.Tensor:
        """(n,) bool: within the scene's seat tolerances (axis distance and tilt)."""
        c = self.scene.cfg
        return (self.off_axis(i) <= c.align_xy) & (self.tilt_deg(i) <= c.align_axis_deg)

    def seated(self, i: int = 0) -> torch.Tensor:
        """(n,) bool: the scene's own seated test."""
        return (self.height(i) <= self.scene.cfg.seat_z) & self.on_axis(i)

    def threaded(self, i: int = 0, engage: float = THREAD_PITCH) -> torch.Tensor:
        """(n,) bool: at least one thread turn below the bolt top and on axis — the thread holds it."""
        return (self.height(i) <= self.entry_z - engage) & self.on_axis(i)

    # -- 1. place it carefully before turning ---------------------------------------------------
    def check_nut(self, i: int = 0):
        """(ok (n,) bool, info): nut `i` sits on its bolt's top, centred within `align_xy`, upright
        within `align_deg` — ready to start the thread. Records the pass per env for `drive(i)`.
        `info["problems"]` lists, per env, what to correct."""
        off, tilt = self.off_axis(i), self.tilt_deg(i)
        h = self.height(i) - self.entry_z
        centred, upright, at_top = off <= self.align_xy, tilt <= self.align_deg, h.abs() <= self.entry_tol
        ok = centred & upright & at_top
        problems = []
        for e in range(off.shape[0]):
            p = []
            if not bool(centred[e]):
                p.append(f"{float(off[e]) * 1e3:.1f} mm off the bolt axis (max {self.align_xy * 1e3:.1f})")
            if not bool(upright[e]):
                p.append(f"tilted {float(tilt[e]):.1f} deg (max {self.align_deg:.1f})")
            if not bool(at_top[e]):
                p.append(f"{abs(float(h[e])) * 1e3:.1f} mm {'above' if float(h[e]) > 0 else 'below'} the bolt top")
            problems.append(p)
        self.checked[:, i] = ok.to(self.checked.device)
        self.checked_bolt[:, i] = self.nearest_bolt(i).to(self.checked_bolt.device)
        info = {"nut": f"nut_{i}", "off_axis_mm": [round(float(x) * 1e3, 2) for x in off],
                "tilt_deg": [round(float(x), 2) for x in tilt], "problems": problems}
        return ok, info

    # -- 2-5. drive it without damage -----------------------------------------------------------
    def drive(self, i: int = 0, **kw) -> NutDrive:
        """Start a `NutDrive` on nut `i` (after `check_nut(i)` passed). Closes the previous drive.
        The check only counts if the nut is STILL in the checked pose on the same bolt now: a nut
        knocked crooked, or moved to another bolt, since the check must be checked again."""
        if self._drive is not None:
            self._drive.close()
        import torch

        off, tilt = self.off_axis(i), self.tilt_deg(i)
        still_ok = ((off <= self.align_xy) & (tilt <= self.align_deg)
                    & ((self.height(i) - self.entry_z).abs() <= self.entry_tol)
                    & (self.nearest_bolt(i) == self.checked_bolt[:, i].to(off.device)))
        self.checked[:, i] &= still_ok.to(self.checked.device)
        self._drive = NutDrive(self, i, **kw)
        return self._drive

    # -- 6. don't drop it -----------------------------------------------------------------------
    def near_bolt_top(self, i: int = 0) -> torch.Tensor:
        h = self.height(i)
        return (self.off_axis(i) <= self.approach_radius) & (h <= self.entry_z + self.approach_height)

    def _approach_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        for k in range(self.n_pairs):
            mine = torch.tensor([o == f"nut_{k}" for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                slow |= mine & self.near_bolt_top(k).to(held.device)
        return torch.where(slow, torch.full_like(slow, self.approach_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    def supported(self, obj: str, tol: float = 0.006, still: float = 0.02,
                  bolt_clear: float = 0.022) -> torch.Tensor:
        """(n,) bool: the nut lies on the table, or is threaded on a bolt — and is not moving. A nut
        merely resting on the bolt top is NOT supported: it slides off. The bolt is short (its top
        is ~5 mm above a nut lying flat), so height alone cannot tell the two apart: "on the
        table" also needs the nut at least `bolt_clear` from the bolt axis (a nut touching the
        bolt's side is ~22 mm from it: half the nut across corners plus the bolt radius)."""
        i = int(obj.split("_")[1])
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        low = d.root_pos_w[:, 2] - self._surface_z() < float(self.scene.cfg.nut_init_z) + tol
        on_table = low & (self.off_axis(i).to(low.device) >= bolt_clear)
        return (on_table | self.threaded(i).to(slow.device)) & slow

    def grip_report(self) -> dict:
        """Grip set vs. what holding the nut needs. `capped` = the nut's friction is too low for the
        cap (the scene default 0.01 needs ~35 N; the robot configs use 0.4, ~1.1 N)."""
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
                "required_N_to_hold_nut": round(self.needed, 2),
                "capped": self.needed > self.grip.force_max}


class NutDrive:
    """Watches nut `i` being screwed down. Status per env, in priority order:

      "seated"           at the scene's seat depth — STOP (more grinds the pads on the bolt head
                         and over-tightens)
      "unchecked"        `check_nut(i)` has not passed for this nut — check before turning
      "off_axis"         the nut tipped or walked off the axis while turning — stop, back off
      "wrong_direction"  once on the thread, the nut rose `rise_tol` — unscrewing: stop, re-seat,
                         turn the other way (before it catches the thread, turning back to find
                         the thread start lifts it a little: that is allowed)
      "cross_threaded"   pushing hard with no progress — stop, back it off, check again
      "slipping_at_cap"  the hand turns but the nut does not, even at `grip_force_max` — stop
                         (only judged while the nut is held: open fingers turning back to re-grip
                         are not slip; call `prev.mark_held(nut)` after closing again)
      "driving"          keep going

    A problem stop sticks for this drive and USES UP the nut's check: after backing off, the nut
    must pass `check_nut(i)` again before a new drive will report "driving".

    `min_progress` over `window_s` is the slowest descent that still counts as threading: 0.1 mm/s
    is a slow 0.3 rad/s turn on the 2 mm thread. Only judged while the arm pushes hard."""

    PROBLEMS = ("off_axis", "wrong_direction", "cross_threaded", "slipping_at_cap")

    def __init__(self, prev: NutThreadPrevention, i: int = 0, window_s: float = 1.0,
                 min_progress: float = 0.0001, rise_tol: float = 0.001, **turn_kw) -> None:
        self.prev, self.i, self.env = prev, i, prev.env
        self.rise_tol = rise_tol
        self._stopped: list[str | None] = [None] * self.env.num_envs  # sticky problem per env
        h = prev.height(i)
        self._lowest = h.clone()
        remaining = (h - prev.scene.cfg.seat_z).clamp_min(0.0)
        self.insert = InsertionMonitor(self.env, f"nut_{i}", axis=-prev.bolt_up(i), seat_depth=remaining,
                                       effort_ratio=prev.drive_effort_ratio, window_s=window_s,
                                       min_progress=min_progress)
        # spin-in-the-fingers only counts while the fingers hold the nut (not while re-gripping)
        self.turn = TurnMonitor(self.env, f"nut_{i}", lambda: prev.bolt_up(i), prev.grip,
                                active_fn=self._slip_check_on, **turn_kw)
        add_post_step(self.env, self._track)

    def _slip_check_on(self) -> list[bool]:
        held = [o == f"nut_{self.i}" for o in self.prev._held_obj]
        return [h and s == "driving" for h, s in zip(held, self.status())]

    def _track(self) -> None:
        import torch

        self._lowest = torch.minimum(self._lowest, self.prev.height(self.i))

    def close(self) -> None:
        remove_post_step(self.env, self._track)
        self.insert.close()
        self.turn.close()

    def status(self) -> list[str]:
        p = self.prev
        seated, on_axis = p.seated(self.i), p.on_axis(self.i)
        checked = p.checked[:, self.i]
        # rising only counts once the nut is on the thread: before that, turning it back to find
        # the thread start lifts it by up to a pitch, which is the right way to start a nut
        engaged = self._lowest <= p.entry_z - THREAD_PITCH
        rose = engaged & (p.height(self.i) - self._lowest > self.rise_tol)
        out = []
        for e in range(self.env.num_envs):
            if bool(seated[e]):
                out.append("seated")
            elif self._stopped[e] is not None:
                out.append(self._stopped[e])
            elif not bool(checked[e]):
                out.append("unchecked")
            elif not bool(on_axis[e]):
                out.append("off_axis")
            elif bool(rose[e]):
                out.append("wrong_direction")
            elif bool(self.insert.stalled[e]):
                out.append("cross_threaded")
            elif bool(self.turn.slipping_at_cap[e]):
                out.append("slipping_at_cap")
            else:
                out.append("driving")
            if out[-1] in self.PROBLEMS and self._stopped[e] is None:
                self._stopped[e] = out[-1]
                p.checked[e, self.i] = False  # the nut must be checked again before driving
        return out

    def should_stop(self) -> list[bool]:
        return [s != "driving" for s in self.status()]
