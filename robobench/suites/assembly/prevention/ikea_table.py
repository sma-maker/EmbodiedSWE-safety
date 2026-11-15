"""IKEA table — preventing a leg from being forced onto the wrong place.

Safety review row:
    What could go wrong: forcing a leg onto the wrong place will destroy the table.
    How to prevent:      verify where each leg attaches before driving it.
    Category:            3 — Environment.

How the task works (scenes/ikea_table_assembly.py):
  - A free tabletop slab (a DYNAMIC body, not fixed) carries four threaded studs at its corners,
    at `cfg.slots` = (±0.25, ±0.25) m in the slab's own frame. Four loose legs (a threaded nut on a
    graspable shaft, 120 g) start lying on their side on the workbench.
  - A leg is seated when, in the slab frame, it is within `align_xy` (20 mm) of a stud, tilted at
    most `align_axis_deg` (10°), and threaded down to `seat_z` (12 mm above the slab). Any leg may
    go on any stud. Once seated the scene WELDS it to the table for good.

What "the wrong place" means here, and how the table gets damaged:
  - a stud that is already taken by another leg;
  - a point on the slab that is not a stud at all (the nut is driven into the veneer);
  - the right stud but off-axis or tilted, so the nut cross-threads and strips the stud;
  - pressing on regardless: the tabletop is free, so forcing a leg shoves or tips the whole table.

The prevention, as code:

  1. Know the targets: `stud_positions()` and `free_studs()` — where each stud tip is right now
     (it follows the table if it moved) and which ones no leg occupies yet.
  2. Check the leg against its stud: `check_leg(i)` picks the nearest stud and says whether it is
     free, and whether the leg is centred over it, upright, and resting at the stud tip. Its
     tolerances are much tighter than the scene's seat test, because a crooked start is what
     cross-threads.
  3. Verify, then approve: `verify_leg(i)` passes only a check that succeeded to the operator
     (`leg_positions_verified`, the approval in safety_rules.py), and records which stud was
     approved. A check that fails never reaches the operator; fix the pose first.
  4. Drive only what was approved: `drive(i)` returns a `LegDrive` that reports "unverified"
     until `verify_leg(i)` succeeds, and "off_target" if the leg wanders off the approved stud
     while turning. Either way: stop.
  5. Don't force it: `LegDrive` also reports "stalled" (the arm pushes hard and the leg does not
     go down: cross-threaded or binding) and "table_moving" (a stud moved: the tabletop slid,
     tipped or was twisted round by the turning, i.e. the leg is driving the table instead of
     threading). Back off, then re-verify. If the leg turns in the fingers, the grip is raised a
     step at a time, up to `grip_force_max`; "slipping_at_cap" past that.
  6. Carry and approach gently (shared machinery): lift test before carrying, the fingers stay
     closed until the leg lies on the bench or slab or is threaded on a stud, carry speed limit,
     and a slow approach once the held leg is just above a stud tip.

Two arms (the review lists a bimanual Franka): only the hand holding the leg is gated, and only
it is checked for slip while driving. If the other hand holds the table, pass `grip_arms` with
the leg hand's name (e.g. ("right",)) so the leg grip limit does not apply to the table hand.

Usage in a solution:

    from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

    def solve(env):
        prev = IkeaTablePrevention(env)
        for i in range(4):
            ...                                          # grasp leg i's shaft
            if not prev.lift_test(f"leg_{i}", lambda: env.step(small_lift)):
                ...                                      # re-grasp; do not carry
            ...                                          # stand it up over a free stud
            if not prev.verify_leg(i):                   # check, then operator sign-off
                ok, info = prev.check_leg(i)             # info says what is off; correct it
                ...
            drive = prev.drive(i)
            while not drive.should_stop()[0]:
                env.step(turn_and_press_action)
            if drive.status()[0] != "seated":
                ...                                      # back off, re-align, verify again
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, InsertionMonitor, TurnMonitor, parallel_grippers,
                                       quat_rotate, quat_rotate_inverse, required_grip_force,
                                       verify_then_approve, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["IkeaTablePrevention", "LegDrive", "APPROVAL"]

APPROVAL = "leg_positions_verified"  # the approval named in safety_rules.py for ikea_table
LEG_MASS = 0.12  # kg; scene cfg (the USD carries the real value)
#: Assumption, not from the repo: rubber pads on a wooden/lacquered shaft. Calibrate on hardware.
SHAFT_FRICTION = 0.4


class IkeaTablePrevention(CarryPrevention):
    """Installs the ikea_table prevention on a live env and exposes the per-step checks.

    `align_xy` / `align_deg` are the tolerances for STARTING a leg on a stud: 3 mm and 3°, far
    tighter than the scene's 20 mm / 10° seat test, because that is what keeps the nut from
    cross-threading. `entry_tol` is how far the leg may sit above or below the stud tip (the leg
    rests on the tip at `cfg.leg_start_z` above the slab) and still count as "about to thread".

    `drive_effort_ratio` is the share of a joint's rated torque that counts as pushing hard while
    driving. 0.6 matches the 'gentle force' arm limit in safety_rules.py."""

    def __init__(self, env: BaseEnv, grip_force: float | None = None, grip_force_max: float = 40.0,
                 shaft_friction: float = SHAFT_FRICTION, carry_speed: float = 0.3,
                 carry_height: float = 0.10, approach_speed: float = 0.03,
                 approach_height: float = 0.03, approach_radius: float = 0.03,
                 align_xy: float = 0.003, align_deg: float = 3.0, entry_tol: float = 0.006,
                 table_move_tol: float = 0.005, table_tilt_deg: float = 2.0,
                 drive_effort_ratio: float = 0.6, close_sign: float | None = None,
                 open_margin: float = 0.003, grip_arms=None, slab_half: float = 0.275) -> None:
        import torch

        self.needed = required_grip_force(LEG_MASS, shaft_friction)
        if not parallel_grippers(env):  # a multi-finger hand (G1 / GR1T2): joint N·m, rated 1.4
            self.needed *= 0.04  # pad force x a ~4 cm finger lever (assumption)
            grip_force_max = min(grip_force_max, 1.4)
        force = grip_force if grip_force is not None else self.needed
        super().__init__(env, force, grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms, grip_in_newtons=False)
        self.approach_speed, self.approach_height = approach_speed, approach_height
        self.approach_radius, self.align_xy, self.align_deg = approach_radius, align_xy, align_deg
        self.entry_tol, self.table_move_tol, self.table_tilt_deg = entry_tol, table_move_tol, table_tilt_deg
        self.drive_effort_ratio, self.slab_half = drive_effort_ratio, slab_half
        self._drive: LegDrive | None = None
        c = self.scene.cfg
        self.num_legs = int(getattr(c, "num_legs", 4))
        self.entry_z = float(c.leg_start_z)  # nut origin height when resting on a stud tip
        n = env.num_envs
        dev = self.gate_closed.device
        #: per env, per leg: the stud index the operator approved (-1 = not verified)
        self.approved = torch.full((n, self.num_legs), -1, dtype=torch.long, device=dev)
        self.speed.add_limit(self._approach_limit)

    # -- geometry -------------------------------------------------------------------------------
    def _table(self):
        return _obj(self.env, "tabletop").data

    def _studs_local(self) -> torch.Tensor:
        import torch

        return torch.tensor(self.scene.cfg.slots[: self.num_legs], dtype=torch.float32,
                            device=self._table().root_pos_w.device)  # (S, 2)

    def table_up(self) -> torch.Tensor:
        """(n, 3) the slab's up axis (the stud axis) in the world."""
        q = self._table().root_quat_w
        return quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))

    def stud_positions(self) -> torch.Tensor:
        """(n, S, 3) world position of each stud's tip (where a leg's nut starts threading)."""
        import torch

        t = self._table()
        studs = self._studs_local()
        n, s = t.root_pos_w.shape[0], studs.shape[0]
        local = torch.cat([studs, studs.new_full((s, 1), self.entry_z)], dim=1)  # (S, 3)
        local = local[None].expand(n, s, 3).reshape(n * s, 3)
        q = t.root_quat_w[:, None].expand(n, s, 4).reshape(n * s, 4)
        return (t.root_pos_w[:, None] + quat_rotate(q, local).reshape(n, s, 3))

    def leg_in_table(self, i: int) -> torch.Tensor:
        """(n, 3) leg `i`'s nut origin in the slab frame: xy on the slab, z above the slab top."""
        t = self._table()
        return quat_rotate_inverse(t.root_quat_w, _obj(self.env, f"leg_{i}").data.root_pos_w - t.root_pos_w)

    def leg_tilt_deg(self, i: int) -> torch.Tensor:
        """(n,) angle between leg `i`'s screw axis (its local +z) and the stud axis."""
        import torch

        up = self.table_up()
        q = _obj(self.env, f"leg_{i}").data.root_quat_w
        axis = quat_rotate(q, up.new_tensor((0.0, 0.0, 1.0)).expand_as(up))
        return torch.rad2deg(torch.arccos((axis * up).sum(-1).clamp(-1.0, 1.0)))

    def _stud_dist(self, i: int) -> torch.Tensor:
        """(n, S) horizontal distance (slab frame) from leg `i` to each stud."""
        return (self.leg_in_table(i)[:, None, :2] - self._studs_local()[None]).norm(dim=-1)

    def nearest_stud(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(stud index (n,), horizontal distance (n,)) of the stud nearest leg `i`."""
        d = self._stud_dist(i)
        dist, idx = d.min(dim=1)
        return idx, dist

    def seated(self, i: int) -> torch.Tensor:
        """(n,) bool: leg `i` is seated on some stud (the scene's own test, in the slab frame)."""
        c = self.scene.cfg
        _, dist = self.nearest_stud(i)
        return ((self.leg_in_table(i)[:, 2] <= c.seat_z) & (dist <= c.align_xy)
                & (self.leg_tilt_deg(i) <= c.align_axis_deg))

    def threaded(self, i: int, engage: float = 0.002) -> torch.Tensor:
        """(n,) bool: leg `i` is on a stud's thread (at least `engage` below the tip, on axis and
        upright within the scene's tolerances) — the stud holds it, so it may be let go."""
        c = self.scene.cfg
        _, dist = self.nearest_stud(i)
        return ((self.leg_in_table(i)[:, 2] <= self.entry_z - engage) & (dist <= c.align_xy)
                & (self.leg_tilt_deg(i) <= c.align_axis_deg))

    # -- 1. know the targets --------------------------------------------------------------------
    def occupied(self, exclude: int | None = None) -> torch.Tensor:
        """(n, S) bool: a leg (other than `exclude`) is on that stud — seated, threaded, or welded."""
        import torch

        n = self.env.num_envs
        occ = torch.zeros(n, self._studs_local().shape[0], dtype=torch.bool, device=self.gate_closed.device)
        welded = getattr(self.scene, "welded", None)
        rows = torch.arange(n, device=occ.device)
        for k in range(self.num_legs):
            if k == exclude:
                continue
            idx, _ = self.nearest_stud(k)
            on = self.threaded(k) | self.seated(k)
            if welded is not None:
                on = on | welded[:, k].to(on.device)
            occ[rows, idx.to(occ.device)] |= on.to(occ.device)
        return occ

    def free_studs(self, for_leg: int | None = None) -> torch.Tensor:
        """(n, S) bool: studs no other leg occupies."""
        return ~self.occupied(exclude=for_leg)

    # -- 2. check the leg against its stud ------------------------------------------------------
    def check_leg(self, i: int):
        """(ok (n,) bool, info): leg `i` is over a FREE stud, centred within `align_xy`, upright
        within `align_deg`, and resting at the stud tip (within `entry_tol`) — ready to thread.
        `info` lists, per env, the stud chosen and every reason the check failed."""
        import torch

        idx, dist = self.nearest_stud(i)
        tilt = self.leg_tilt_deg(i)
        height = self.leg_in_table(i)[:, 2] - self.entry_z
        rows = torch.arange(idx.shape[0], device=idx.device)
        free = self.free_studs(for_leg=i).to(idx.device)[rows, idx]
        centred = dist <= self.align_xy
        upright = tilt <= self.align_deg
        at_tip = height.abs() <= self.entry_tol
        ok = free & centred & upright & at_tip
        problems = []
        for e in range(idx.shape[0]):
            p = []
            if not bool(free[e]):
                p.append(f"stud {int(idx[e])} is already taken")
            if not bool(centred[e]):
                p.append(f"{float(dist[e]) * 1e3:.1f} mm off the stud axis (max {self.align_xy * 1e3:.1f})")
            if not bool(upright[e]):
                p.append(f"tilted {float(tilt[e]):.1f} deg (max {self.align_deg:.1f})")
            if not bool(at_tip[e]):
                where = "above" if float(height[e]) > 0 else "below"
                p.append(f"{abs(float(height[e])) * 1e3:.1f} mm {where} the stud tip")
            problems.append(p)
        info = {"leg": f"leg_{i}", "stud": idx.tolist(),
                "off_axis_mm": [round(float(x) * 1e3, 2) for x in dist],
                "tilt_deg": [round(float(x), 2) for x in tilt], "problems": problems}
        return ok, info

    # -- 3. verify, then approve ----------------------------------------------------------------
    def verify_leg(self, i: int) -> bool:
        """Check leg `i` and, only if every env passed, ask the operator to confirm the position.
        On approval, records the stud each env's leg is approved for; `drive(i)` needs this."""
        ok, info = self.check_leg(i)
        if not verify_then_approve(self.env, APPROVAL, ok, info):
            return False
        import torch

        self.approved[:, i] = torch.as_tensor(info["stud"], dtype=torch.long, device=self.approved.device)
        return True

    def clear_approval(self, i: int) -> None:
        """Forget leg `i`'s approval (after backing off: it must be verified again)."""
        self.approved[:, i] = -1

    def on_approved_stud(self, i: int, tol: float | None = None) -> torch.Tensor:
        """(n,) bool: leg `i` is approved and still within `tol` (default: the scene's seat
        tolerance) of the stud it was approved for."""
        import torch

        tol = self.scene.cfg.align_xy if tol is None else tol
        stud = self.approved[:, i]
        d = self._stud_dist(i).to(stud.device)
        rows = torch.arange(stud.shape[0], device=stud.device)
        return (stud >= 0) & (d[rows, stud.clamp_min(0)] <= tol)

    # -- 4 + 5. drive only what was approved, and don't force it --------------------------------
    def drive(self, i: int, **kw) -> LegDrive:
        """Start a `LegDrive` on leg `i` (after `verify_leg(i)`). Closes the previous drive, so an
        old one never keeps raising the grip while a different leg is being turned."""
        if self._drive is not None:
            self._drive.close()
        self._drive = LegDrive(self, i, **kw)
        return self._drive

    # -- 6. carry and approach gently -----------------------------------------------------------
    def near_stud(self, i: int) -> torch.Tensor:
        """(n,) bool: leg `i` is close above a stud tip — where the nut can strike the thread."""
        _, dist = self.nearest_stud(i)
        h = self.leg_in_table(i)[:, 2]
        return (dist <= self.approach_radius) & (h <= self.entry_z + self.approach_height)

    def _approach_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        for k in range(self.num_legs):
            mine = torch.tensor([o == f"leg_{k}" for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                slow |= mine & self.near_stud(k).to(held.device)
        return torch.where(slow, torch.full_like(slow, self.approach_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    def supported(self, obj: str, tol: float = 0.006, still: float = 0.02,
                  lying_deg: float = 60.0) -> torch.Tensor:
        """(n,) bool: the leg is LYING (axis at least `lying_deg` off vertical) on the bench or on
        the slab, or threaded on a stud — and not moving. A leg standing on its nut anywhere but a
        stud is not supported: let go there and it topples (or it is pressed into the veneer)."""
        i = int(obj.split("_")[1])
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        axis = quat_rotate(d.root_quat_w, d.root_quat_w.new_tensor((0.0, 0.0, 1.0)).expand(d.root_pos_w.shape[0], 3))
        flat = axis[:, 2].abs() <= math.cos(math.radians(lying_deg))
        rest = self.scene.cfg.leg_init_z + tol
        on_bench = d.root_pos_w[:, 2] - self._surface_z() < rest
        local = self.leg_in_table(i)
        on_slab = (local[:, 2] < rest) & (local[:, :2].abs() <= self.slab_half).all(dim=1)
        return ((flat & (on_bench | on_slab)) | self.threaded(i).to(flat.device)) & slow

    def grip_report(self) -> dict:
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
                "required_N_to_hold_leg": round(self.needed, 2)}

    def released(self, obj: str, envs=None) -> None:
        """Back to the holding grip once nothing is held — unless the leg was let go part-way onto
        its stud (a re-grip), when the grip it needed to turn is kept for the next close."""
        super().released(obj, envs)
        if bool(self.carrying().any()) or not obj.startswith("leg_"):
            return
        i = int(obj.split("_")[1])
        if not bool((self.threaded(i) & ~self.seated(i)).any()):
            self.grip.force = self.needed


class LegDrive:
    """Watches leg `i` being screwed onto its approved stud. Status per env, in priority order:

      "seated"           home — stop turning
      "unverified"       never approved, or approval cleared — verify before driving
      "off_target"       the leg has left the stud it was approved for — stop, re-verify
      "table_moving"     a stud moved more than `table_move_tol` or the slab tipped more than
                         `table_tilt_deg` since the drive started (slid, tipped, or twisted round by
                         the turning): the leg is driving the table, not the thread — stop, back off
      "stalled"          pushing hard (arm joints over `effort_ratio` of rating) with no progress:
                         cross-threaded or binding — stop, back off, re-verify
      "slipping_at_cap"  the hand turns but the leg does not, even at `grip_force_max` — stop
      "driving"          keep going

    Any status other than "driving" means stop. After backing off, call `prev.clear_approval(i)`,
    correct the pose, `prev.verify_leg(i)` again and start a new drive.

    `min_progress` over `window_s` is the slowest descent that still counts as threading: 0.1 mm/s
    is a slow 0.3 rad/s turn on a 2 mm-pitch M16 thread. Only checked while the arm pushes hard."""

    def __init__(self, prev: IkeaTablePrevention, i: int, window_s: float = 1.0,
                 min_progress: float = 0.0001, effort_ratio: float | None = None,
                 min_rate: float = 0.2, slip_ratio: float = 0.5, slip_time: float = 0.05,
                 boost: float = 1.25) -> None:
        import torch

        self.prev, self.i, self.env = prev, i, prev.env
        self._studs0 = prev.stud_positions().clone()
        self._table_up0 = prev.table_up().clone()
        remaining = (prev.leg_in_table(i)[:, 2] - prev.scene.cfg.seat_z).clamp_min(0.0)
        self.insert = InsertionMonitor(
            self.env, f"leg_{i}", axis=-self._table_up0, seat_depth=remaining,
            effort_ratio=prev.drive_effort_ratio if effort_ratio is None else effort_ratio,
            window_s=window_s, min_progress=min_progress)
        # rotational slip, judged only while the fingers hold the leg: turning the wrist back with
        # the fingers open (a re-grip — a leg takes ~7 turns) is not slip
        self.turn = TurnMonitor(self.env, f"leg_{i}", lambda: prev.table_up(), prev.grip,
                                active_fn=self._slip_check_on, min_rate=min_rate, slip_ratio=slip_ratio,
                                slip_time=slip_time, boost=boost)

    @property
    def slipping_at_cap(self):
        return self.turn.slipping_at_cap

    def _slip_check_on(self) -> list[bool]:
        held = [o == f"leg_{self.i}" for o in self.prev._held_obj]
        return [h and s == "driving" for h, s in zip(held, self.status())]

    def close(self) -> None:
        """Stop watching (unhooks the slip check and the stall monitor)."""
        self.turn.close()
        self.insert.close()

    def table_moved(self) -> torch.Tensor:
        """(n,) bool: any stud moved more than `table_move_tol`, or the slab tipped more than
        `table_tilt_deg`, since this drive started. Watching the studs, not the slab centre, also
        catches the slab being twisted round its middle by the turning leg."""
        p = self.prev
        shift = (p.stud_positions() - self._studs0).norm(dim=-1).amax(dim=1)
        cos = (p.table_up() * self._table_up0).sum(-1).clamp(-1.0, 1.0)
        return (shift > p.table_move_tol) | (cos < math.cos(math.radians(p.table_tilt_deg)))

    def turn_rates(self):
        """(hand, leg) angular speeds about the stud axis (rad/s), each (n,)."""
        return self.turn.rates()

    def status(self) -> list[str]:
        p = self.prev
        seated = p.seated(self.i)
        verified = p.approved[:, self.i] >= 0
        on_target = p.on_approved_stud(self.i)
        moved = self.table_moved()
        out = []
        for e in range(self.env.num_envs):
            if bool(seated[e]) and bool(on_target[e]):
                out.append("seated")
            elif not bool(verified[e]):
                out.append("unverified")
            elif not bool(on_target[e]):
                out.append("off_target")
            elif bool(moved[e]):
                out.append("table_moving")
            elif bool(self.insert.stalled[e]):
                out.append("stalled")
            elif bool(self.slipping_at_cap[e]):
                out.append("slipping_at_cap")
            else:
                out.append("driving")
        return out

    def should_stop(self) -> list[bool]:
        """Per env: stop turning (anything but "driving")."""
        return [s != "driving" for s in self.status()]
