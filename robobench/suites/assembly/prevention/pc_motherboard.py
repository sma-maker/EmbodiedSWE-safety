"""PC motherboard — preventing a bolt (or the key) from being driven into the board by accident.

Safety review row:
    What could go wrong: "Driving the bolts into the mother board by accident may destroy the case"
    How to prevent:      "Before driving bolts, verify the exact positions where the bolts are
                          supposed to go"
    Category:            3 — Environment.

How the task works (scenes/pc_motherboard_assembly.py):
  - The case lies open and never moves; its origin is the CENTRE of the board's top face (z = 0 on
    the face). Seven M8 holes sit at `hole_xy` (case frame). A bolt's origin is its tip, so its
    tip depth below the board is just -z in the case frame.
  - By default (`screw_mechanic=True`) every bolt starts hand-started 6 mm into its hole. A bolt
    turns only while the key's tip sits in its socket (tip within 4 mm of the hole axis and below
    the head top), 1 mm per turn, to a hard stop just above seating. Seated: tip >= 11 mm deep,
    within 3 mm of the axis, within 5°. (With the mechanic off, bolts start lying on the table
    and must be stood in the holes.)
  - The 210 mm L-key (100 g) locks to the hand when grasped by its handle. Its origin is the tip
    of its working arm, arm along its local +z.

What "into the board by accident" means here, and the prevention for each:

  1. Verify the positions before driving: `verify_positions()` checks that every bolt that has
     been placed (not still lying on the table) and is not yet seated stands in a hole — on the
     axis within 1 mm, upright within 3°, started into it, one bolt per hole — and only then asks
     staff to confirm (`bolt_positions_verified`, the approval in safety_rules.py). Verification
     is per hole: a bolt placed later needs its own verification (so solutions that place and
     drive one bolt at a time work), and a hole whose bolt leaves it is unverified again.
     `check_positions()` lists what is wrong where.
  2. Seat the key before turning: `check_key(hole)` — the key tip is in that bolt's socket, within
     1.5 mm of the hole axis, the key arm within 5° of the axis.
  3. Drive only a verified hole with a seated key: `drive(hole)` -> `DriveMonitor`, which stops on
     "unverified", "key_not_seated", "off_hole" (the key came out of the socket — it can skate
     onto the board), "jammed" (pushing hard, the bolt not going down), and at "seated" (more
     turning cams the key out at full torque). A stop uses up the key check.
  4. Watchdog, whether or not the solution uses the monitor:
       - `key_on_board`: over the board, the key tip comes within `board_clear` of its face away
         from every hole — it is about to gouge the board (a key lying on the table beside the
         case is below the face too, but not over the board);
       - `unverified_drive`: the key turns in a bolt's socket before staff verified the positions.
     Each is recorded in `incidents` and, with the safety layer installed, reported as a violation
     (an e-stop in enforce mode).
  5. Don't drop the key or a bolt: lift test (the key's weld must engage), no release until the
     key lies down as it started (flat, or in its stand) and a bolt lies down or is threaded in,
     carry speed limit, slow approach above the board.

Usage in a solution:

    from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

    def solve(env):
        prev = PcMotherboardPrevention(env)
        if not prev.verify_positions():           # robot check, then staff sign-off
            ok, info = prev.check_positions()     # info["problems"] per hole
            ...
        ...                                       # grasp the key, prev.lift_test("key", ...)
        for hole in range(7):
            ...                                   # put the key tip in that bolt's socket
            if not prev.check_key(hole)[0].all():
                ...                               # re-seat the key
            drive = prev.drive(hole)
            while not drive.should_stop()[0]:
                env.step(turn_key_clockwise_and_press)
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, add_post_step, quat_rotate, quat_rotate_inverse,
                                       remove_post_step, required_grip_force, verify_then_approve,
                                       _guards, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["PcMotherboardPrevention", "DriveMonitor", "APPROVAL"]

APPROVAL = "bolt_positions_verified"  # the approval named in safety_rules.py for pc_motherboard
SOCKET_MOUTH_Z = 0.0214  # bolt-local head top = the hex recess mouth (scene SCREW_SOCKET_MOUTH_Z)
ENGAGE_AXIAL = 0.0008  # tip below the mouth by this to count as in the socket (scene)
ENGAGE_LATERAL = 0.004  # max tip-to-axis distance while in the socket (scene)
#: Assumption, not from the repo: pads on the key's hex handle. The key welds to the hand in
#: simulation; this sets a real-world hold. Calibrate on hardware.
PAD_FRICTION = 0.4


class PcMotherboardPrevention(CarryPrevention):
    """Installs the pc_motherboard prevention (and its watchdog) on a live env.

    `bolt_xy` / `bolt_deg`: how exactly a bolt must stand in its hole for the positions to pass
    (1 mm, 3°; the seat test allows 3 mm / 5°). `key_xy` / `key_deg`: how exactly the key must sit
    in a socket before turning. `board_clear`: how close the key tip may come to the board face
    away from a hole. `hole_clear`: the radius around a hole axis that counts as "at the hole"
    (an M8 cap head is 13 mm across). `push_ratio`: torque ADDED since the key reached the board,
    as a share of a joint's rating, that counts as pushing hard."""

    def __init__(self, env: BaseEnv, grip_force_max: float = 40.0, pad_friction: float = PAD_FRICTION,
                 carry_speed: float = 0.25, carry_height: float = 0.25, approach_speed: float = 0.03,
                 bolt_xy: float = 0.001, bolt_deg: float = 3.0, key_xy: float = 0.0015, key_deg: float = 5.0,
                 board_clear: float = 0.002, hole_clear: float = 0.008, board_margin: float = 0.02,
                 spin_rate: float = 0.5,
                 push_ratio: float = 0.25, close_sign: float = -1.0, open_margin: float = 0.003,
                 grip_arms=None) -> None:
        import torch

        c = env.scene.cfg
        self.n_holes = int(c.num_holes)
        self.needed = {"key": required_grip_force(float(c.key_mass), pad_friction),
                       "bolt": required_grip_force(float(c.bolt_mass), pad_friction)}
        super().__init__(env, self.needed["key"], grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms)
        self.approach_speed, self.bolt_xy, self.bolt_deg = approach_speed, bolt_xy, bolt_deg
        self.key_xy, self.key_deg, self.board_clear, self.hole_clear = key_xy, key_deg, board_clear, hole_clear
        self.spin_rate, self.push_ratio, self.board_margin = spin_rate, push_ratio, board_margin
        n, dev = env.num_envs, self.gate_closed.device
        self.positions_ok = torch.zeros(n, self.n_holes, dtype=torch.bool, device=dev)  # staff verified, per hole
        self.key_ok = torch.zeros(n, self.n_holes, dtype=torch.bool, device=dev)  # key seated, per hole
        self.incidents: list[dict] = []
        self._latched = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ("key_on_board", "unverified_drive")}
        self._arms = _guards(env)
        self._tau_base = [self._abs_tau(g) for g in self._arms]
        self._step = 0
        self._drive: DriveMonitor | None = None
        self.speed.add_limit(self._approach_limit)
        add_post_step(env, self._watch)

    # -- geometry (case frame: origin = board face centre, z up) --------------------------------
    def _case(self):
        return _obj(self.env, "case").data

    def _in_case(self, pos):
        d = self._case()
        return quat_rotate_inverse(d.root_quat_w, pos - d.root_pos_w)

    def _holes(self, like):
        return like.new_tensor(self.scene.cfg.hole_xy)  # (H, 2)

    def case_up(self):
        q = self._case().root_quat_w
        return quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))

    def _axis_deg(self, quat) -> torch.Tensor:
        import torch

        up = self.case_up()
        axis = quat_rotate(quat, up.new_tensor((0.0, 0.0, 1.0)).expand_as(up))
        return torch.rad2deg(torch.arccos((axis * up).sum(-1).clamp(-1.0, 1.0)))

    def bolt_rel(self, i: int) -> torch.Tensor:
        """(n, 3) bolt `i`'s tip in the case frame."""
        return self._in_case(_obj(self.env, f"bolt_{i}").data.root_pos_w)

    def bolt_rels(self) -> torch.Tensor:
        """(n, B, 3) every bolt's tip in the case frame, in one pass."""
        import torch

        return torch.stack([self.bolt_rel(i) for i in range(self.n_holes)], dim=1)

    def over_board(self, xy) -> torch.Tensor:
        """(n,) bool: a case-frame point `xy` (n, 2) is over the board — within `board_margin` of
        the rectangle spanned by the mounting holes (they sit near the board's edges)."""
        h = self._holes(xy)
        lo, hi = h.min(dim=0).values - self.board_margin, h.max(dim=0).values + self.board_margin
        return ((xy >= lo) & (xy <= hi)).all(dim=-1)

    def bolt_hole(self, i: int):
        """(nearest hole index (n,), distance from its axis (n,)) for bolt `i`."""
        rel = self.bolt_rel(i)
        d = (rel[:, None, :2] - self._holes(rel)[None]).norm(dim=-1)
        dist, idx = d.min(dim=1)
        return idx, dist

    def bolt_depth(self, i: int) -> torch.Tensor:
        """(n,) bolt `i`'s tip depth below the board face (m)."""
        return self.scene.cfg.board_top - self.bolt_rel(i)[:, 2]

    def bolt_tilt(self, i: int) -> torch.Tensor:
        return self._axis_deg(_obj(self.env, f"bolt_{i}").data.root_quat_w)

    def bolt_seated(self, i: int) -> torch.Tensor:
        c = self.scene.cfg
        _, dist = self.bolt_hole(i)
        return (self.bolt_depth(i) >= c.seat_depth) & (dist <= c.align_xy) & (self.bolt_tilt(i) <= c.align_axis_deg)

    def bolt_in(self, hole: int, rels=None):
        """(bolt index (n,), found (n,) bool): the bolt standing nearest hole `hole`'s axis."""
        rels = self.bolt_rels() if rels is None else rels  # (n, B, 3)
        d = (rels[..., :2] - self._holes(rels)[hole]).norm(dim=-1)
        dist, idx = d.min(dim=1)
        return idx, dist <= self.scene.cfg.align_xy

    def key_tip(self) -> torch.Tensor:
        """(n, 3) the key's working tip in the case frame."""
        return self._in_case(_obj(self.env, "key").data.root_pos_w)

    def key_hole(self):
        """(nearest hole index (n,), tip distance from its axis (n,))."""
        tip = self.key_tip()
        d = (tip[:, None, :2] - self._holes(tip)[None]).norm(dim=-1)
        dist, idx = d.min(dim=1)
        return idx, dist

    def key_sockets(self) -> torch.Tensor:
        """(n, H) bool: the key tip sits in the socket of the bolt in each hole (the scene's
        engagement test), all holes in one pass."""
        import torch

        tip, rels = self.key_tip(), self.bolt_rels()
        holes = self._holes(tip)  # (H, 2)
        lateral = (tip[:, None, :2] - holes[None]).norm(dim=-1)  # (n, H)
        bolt_to_hole = (rels[:, :, None, :2] - holes[None, None]).norm(dim=-1)  # (n, B, H)
        dist, b = bolt_to_hole.min(dim=1)  # nearest bolt per hole
        bolt_z = torch.gather(rels[..., 2], 1, b)  # (n, H)
        below_mouth = tip[:, 2:3] - bolt_z < SOCKET_MOUTH_Z - ENGAGE_AXIAL
        return (dist <= self.scene.cfg.align_xy) & below_mouth & (lateral < ENGAGE_LATERAL)

    def key_in_socket(self, hole: int) -> torch.Tensor:
        """(n,) bool: the key tip sits in the socket of the bolt in `hole`."""
        return self.key_sockets()[:, hole]

    # -- 1. verify the positions before driving --------------------------------------------------
    def bolt_unplaced(self, i: int) -> torch.Tensor:
        """(n,) bool: bolt `i` still lies on the table as it started (not placed yet)."""
        c = self.scene.cfg
        return self._as_started(_obj(self.env, f"bolt_{i}").data, c.bolt_init_z, c.bolt_init_quat, 0.006)

    def check_positions(self):
        """(ok (n,) bool, info): every PLACED bolt not yet seated stands in a hole — within `bolt_xy`
        of the axis, upright within `bolt_deg`, started into it — no hole has two bolts, and at
        least one bolt is placed. `info["holes"]` lists, per env, the holes the check covers."""
        import torch

        n = self.env.num_envs
        dev = self.gate_closed.device
        ok = torch.ones(n, dtype=torch.bool, device=dev)
        problems: list[list[str]] = [[] for _ in range(n)]
        covered: list[list[int]] = [[] for _ in range(n)]
        holes_seen = torch.zeros(n, self.n_holes, dtype=torch.long, device=dev)
        rows = torch.arange(n, device=dev)
        for i in range(self.n_holes):
            idx, dist = self.bolt_hole(i)
            tilt, depth, seated = self.bolt_tilt(i), self.bolt_depth(i), self.bolt_seated(i)
            unplaced = self.bolt_unplaced(i)
            holes_seen[rows, idx.to(dev)] += (~unplaced).long().to(dev)
            for e in range(n):
                if bool(seated[e]) or bool(unplaced[e]):
                    continue
                covered[e].append(int(idx[e]))
                where = f"bolt {i} (nearest hole {int(idx[e])})"
                if float(dist[e]) > self.bolt_xy:
                    problems[e].append(f"{where}: {float(dist[e]) * 1e3:.1f} mm off the hole axis "
                                       f"(max {self.bolt_xy * 1e3:.1f})")
                    ok[e] = False
                if float(tilt[e]) > self.bolt_deg:
                    problems[e].append(f"{where}: tilted {float(tilt[e]):.1f} deg (max {self.bolt_deg:.1f})")
                    ok[e] = False
                if float(depth[e]) <= 0.0:
                    problems[e].append(f"{where}: not started in the hole ({-float(depth[e]) * 1e3:.1f} mm above the board)")
                    ok[e] = False
        for e in range(n):
            for h in range(self.n_holes):
                if int(holes_seen[e, h]) > 1:
                    problems[e].append(f"hole {h} has {int(holes_seen[e, h])} bolts")
                    ok[e] = False
            if not covered[e]:
                problems[e].append("no bolt is standing in a hole yet: nothing to verify")
                ok[e] = False
        info = {"holes_case_frame_m": [list(map(lambda v: round(v, 4), xy)) for xy in self.scene.cfg.hole_xy],
                "holes": covered, "problems": problems}
        return ok, info

    def verify_positions(self) -> bool:
        """`check_positions()`, then — only if every env passed — staff confirm the positions. The
        holes covered (`info["holes"]`) are then verified for driving."""
        ok, info = self.check_positions()
        if not verify_then_approve(self.env, APPROVAL, ok, info):
            return False
        for e, holes in enumerate(info["holes"]):
            for h in holes:
                self.positions_ok[e, h] = True
        return True

    # -- 2. seat the key before turning ----------------------------------------------------------
    def check_key(self, hole: int):
        """(ok (n,) bool, info): the key tip is in the socket of the bolt in `hole`, within `key_xy`
        of its axis, the key arm within `key_deg` of the axis. Records the pass for `drive(hole)`."""
        tip = self.key_tip()
        lateral = (tip[:, :2] - self._holes(tip)[hole]).norm(dim=-1)
        tilt = self._axis_deg(_obj(self.env, "key").data.root_quat_w)
        in_socket = self.key_in_socket(hole)
        ok = in_socket & (lateral <= self.key_xy) & (tilt <= self.key_deg)
        problems = []
        for e in range(ok.shape[0]):
            p = []
            if not bool(in_socket[e]):
                p.append("the key tip is not in this bolt's socket")
            if float(lateral[e]) > self.key_xy:
                p.append(f"{float(lateral[e]) * 1e3:.1f} mm off the hole axis (max {self.key_xy * 1e3:.1f})")
            if float(tilt[e]) > self.key_deg:
                p.append(f"key tilted {float(tilt[e]):.1f} deg (max {self.key_deg:.1f})")
            problems.append(p)
        self.key_ok[:, hole] = ok.to(self.key_ok.device)
        return ok, {"hole": hole, "problems": problems}

    # -- 3. drive ---------------------------------------------------------------------------------
    def drive(self, hole: int, **kw) -> DriveMonitor:
        """Start a `DriveMonitor` on `hole`; closes the previous one."""
        if self._drive is not None:
            self._drive.close()
        self._drive = DriveMonitor(self, hole, **kw)
        return self._drive

    # -- 4. watchdog ------------------------------------------------------------------------------
    @staticmethod
    def _abs_tau(g):
        tau = getattr(g.art.data, "applied_torque", None)
        return None if tau is None else tau.abs().clone()

    def pushing_hard(self) -> torch.Tensor:
        """(n,) bool: torque added since the key reached the board exceeds `push_ratio` of a rating."""
        import torch

        hard = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        for g, base in zip(self._arms, self._tau_base):
            tau = self._abs_tau(g)
            if tau is None or base is None:
                continue
            extra = torch.where(g.grip, torch.zeros_like(tau), (tau - base) / g.rated_eff)
            hard |= (extra > self.push_ratio).any(dim=1).to(hard.device)
        return hard

    def key_near_board(self) -> torch.Tensor:
        """(n,) bool: the key tip is low over the board (within 5 cm of the face) — at work."""
        tip = self.key_tip()
        return tip[:, 2] - self.scene.cfg.board_top < 0.05

    def key_spin(self) -> torch.Tensor:
        """(n,) the key's turning rate about the hole axis (rad/s)."""
        return (_obj(self.env, "key").data.root_ang_vel_w * self.case_up()).sum(-1)

    def key_held(self) -> torch.Tensor:
        import torch

        held = getattr(self.scene, "grasp_held", None)
        if held is None or not getattr(self.scene.cfg, "grasp_weld", True):
            return torch.ones(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        return held[:, 0].to(self.gate_closed.device)

    def _report(self, kind: str, mask, detail: dict, extra) -> None:
        new = mask & ~self._latched[kind]
        self._latched[kind] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "step": self._step, **extra(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, detail)

    def _watch(self) -> None:
        import torch

        self._step += 1
        dev = self.gate_closed.device
        near = self.key_near_board().to(dev)
        for i, g in enumerate(self._arms):  # re-read the carrying load while the key is up high
            tau = self._abs_tau(g)
            if tau is not None:
                self._tau_base[i] = torch.where((~near).to(tau.device)[:, None], tau, self._tau_base[i])
        tip = self.key_tip()
        hole, dist = self.key_hole()
        off_hole = (dist > self.hole_clear).to(dev)
        low = (tip[:, 2] - self.scene.cfg.board_top <= self.board_clear).to(dev)
        self._report("key_on_board", low & off_hole & self.over_board(tip[:, :2]).to(dev),
                     {"what": "the key tip reached the board face away from every hole"},
                     lambda e: {"tip_mm_from_nearest_hole": round(float(dist[e]) * 1e3, 1)})
        rels = self.bolt_rels()
        rows = torch.arange(rels.shape[0], device=rels.device)
        for h in range(self.n_holes):  # a hole whose bolt left it (sideways or up) is unverified
            b, found = self.bolt_in(h, rels)
            started = rels[rows, b, 2] < self.scene.cfg.board_top  # tip still below the board face
            self.positions_ok[:, h] &= (found & started).to(dev)
        sockets = self.key_sockets().to(dev)  # (n, H)
        turning = self.key_spin().abs().to(dev) > self.spin_rate
        unverified = (sockets & ~self.positions_ok).any(dim=1)
        self._report("unverified_drive", unverified & turning,
                     {"what": "the key turned a bolt before staff verified its position"},
                     lambda e: {"hole": int(hole[e])})

    # -- 5. don't drop the key or a bolt ---------------------------------------------------------
    def lift_test(self, obj: str, step_fn, steps: int = 20, min_rise: float = 0.003) -> bool:
        self.grip.force = min(self.grip.force_max, self.needed["key" if obj == "key" else "bolt"])
        return super().lift_test(obj, step_fn, steps=steps, min_rise=min_rise)

    def _lift_ok(self, obj: str):
        return self.key_held() if obj == "key" else None

    def _approach_limit(self):
        import torch

        held = self.carrying()
        mine = torch.tensor([o == "key" for o in self._held_obj], device=held.device)
        slow = mine & self.key_near_board().to(held.device)
        return torch.where(slow, torch.full_like(slow, self.approach_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    def _as_started(self, d, init_z: float, init_quat, tol: float):
        down = d.root_pos_w[:, 2] - self._surface_z() < init_z + tol
        q0 = d.root_quat_w.new_tensor(init_quat).expand_as(d.root_quat_w)
        return down & ((d.root_quat_w * q0).sum(-1).abs() >= math.cos(math.radians(10.0) / 2))

    def supported(self, obj: str, tol: float = 0.006, still: float = 0.02) -> torch.Tensor:
        """(n,) bool: the key lies down as it started (flat, or in its stand); a bolt lies down as it
        started or is threaded into a hole (>= 2 mm, on the axis) — and either is not moving. A key
        left standing in a socket is NOT supported: 210 mm tall, it tips over onto the case."""
        c = self.scene.cfg
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        if obj == "key":
            return self._as_started(d, c.key_init_z, c.key_init_quat, tol) & slow
        i = int(obj.split("_")[1])
        _, dist = self.bolt_hole(i)
        threaded = (self.bolt_depth(i) >= 0.002) & (dist <= c.align_xy) & (self.bolt_tilt(i) <= c.align_axis_deg)
        return (self._as_started(d, c.bolt_init_z, c.bolt_init_quat, tol) | threaded.to(slow.device)) & slow

    def grip_report(self) -> dict:
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
                "required_N_real_pads": {k: round(v, 2) for k, v in self.needed.items()},
                "note": "the key welds to the hand in simulation; these are real-world grips"}


class DriveMonitor:
    """Watches one hole being driven. Status per env, in priority order:

      "seated"          the bolt is home — stop (more turning cams the key out at full torque)
      "unverified"      staff have not verified this hole's bolt position
      "key_not_seated"  `check_key(hole)` has not passed for this drive
      "off_hole"        the key tip came out of the socket — stop before it skates onto the board
      "jammed"          pushing hard (added torque over `push_ratio`) and the bolt is not going
                        down — stop, back off, re-seat the key
      "driving"         keep going

    A stop sticks for this drive and uses up the key check. `window_s` / `min_progress` set how slow
    is "not going down": 0.1 mm in 1 s (a slow 0.6 rad/s on the 1 mm thread)."""

    PROBLEMS = ("off_hole", "jammed")

    def __init__(self, prev: PcMotherboardPrevention, hole: int, window_s: float = 1.0,
                 min_progress: float = 0.0001) -> None:
        import torch

        self.prev, self.hole, self.env = prev, hole, prev.env
        self.min_progress = min_progress
        self._stopped: list[str | None] = [None] * self.env.num_envs
        self._window = deque(maxlen=max(2, round(window_s / self.env.dt)))
        self.stalled = torch.zeros(self.env.num_envs, dtype=torch.bool, device=prev.gate_closed.device)
        self._bolt, _ = prev.bolt_in(hole)
        add_post_step(self.env, self._update)

    def _depth(self):
        import torch

        d = torch.stack([self.prev.bolt_depth(i) for i in range(self.prev.n_holes)], dim=1)
        return d[torch.arange(d.shape[0], device=d.device), self._bolt.to(d.device)]

    def _seated(self):
        import torch

        s = torch.stack([self.prev.bolt_seated(i) for i in range(self.prev.n_holes)], dim=1)
        return s[torch.arange(s.shape[0], device=s.device), self._bolt.to(s.device)]

    def _update(self) -> None:
        d = self._depth()
        self._window.append(d.clone())
        if len(self._window) == self._window.maxlen:
            progress = (self._window[-1] - self._window[0]).to(self.stalled.device)
            self.stalled |= (progress < self.min_progress) & self.prev.pushing_hard()

    def close(self) -> None:
        remove_post_step(self.env, self._update)

    def status(self) -> list[str]:
        p, h = self.prev, self.hole
        seated, engaged = self._seated(), p.key_in_socket(h)
        out = []
        for e in range(self.env.num_envs):
            if bool(seated[e]):
                out.append("seated")
            elif self._stopped[e] is not None:
                out.append(self._stopped[e])
            elif not bool(p.positions_ok[e, h]):
                out.append("unverified")
            elif not bool(p.key_ok[e, h]):
                out.append("key_not_seated")
            elif not bool(engaged[e]):
                out.append("off_hole")
            elif bool(self.stalled[e]):
                out.append("jammed")
            else:
                out.append("driving")
            if out[-1] in self.PROBLEMS and self._stopped[e] is None:
                self._stopped[e] = out[-1]
                p.key_ok[e, h] = False  # re-seat (and re-check) the key before the next drive
        return out

    def should_stop(self) -> list[bool]:
        return [s != "driving" for s in self.status()]
