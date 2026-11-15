"""Prevention — the safety review's "how to prevent" column as code a solution uses while acting.

`safety.py` DETECTS unsafe behaviour (and in enforce mode stops the robot). This module helps a
solution AVOID it in the first place. Two kinds of tools:

  Governors — installed once, then shape every command automatically:
    GripGovernor      force-limited grip: "minimal pressure" (bulb, eggs) or "firm" (knife, box),
                      with `boost()` to tighten when a held object starts to slip
    SpeedGovernor     caps arm speed, globally or per env through added limits (slow near surfaces)

  Checks — the solution asks before / during a risky step, and acts on the answer:
    PlacementGuard    slow down near the destination; release only once the object is supported
    is_aligned / is_upright / verify_then_approve
                      "verify before forcing": pose check, then operator sign-off
    InsertionMonitor  pressing in (GPU, RAM, bolts, threads): reports "stalled" instead of shoving
    HoldMonitor       a held object slipping in the hand (knife, box, wheel) — boosts the grip
    TurnMonitor       a part turned by hand spinning in the fingers (nut, bulb) — boosts the grip
    FixtureGuard      drawers and lids: a 0..1 scale to slow the pull before it slams
    FixtureMove       one drawer / lid / door toward a target: slows before its end stop,
                      reports "straining" (pulling hard, not moving) — with AddedEffort
    PourGuard         pour only with the receiving cup under the spout (latte, coffee)
    AimGuard          dispense only when the nozzle does not point at the robot (syringe)
    KnifeGuard        cut only over the food, and not through the board
    FingerReleaseCheck  fingers that cannot open are caught (shoelaces) — stop and call staff

`setup_prevention(env, scene)` installs the governors a task's review row calls for and returns a
`Prevention` bundle: its `plan` lists the steps, and its factory methods build the checks.

    from robobench.core.prevention import setup_prevention
    from robobench.core.safety import request_approval

    def solve(env):
        prev = setup_prevention(env, "bulb")          # 10 N grip cap, slow arm
        ...                                          # grasp + lift as usual
        ok, info = prev.aligned("bulb_0", socket_xyz, pos_tol=0.003)
        if not prev.verify_then_approve("bulb_aligned", ok, info):
            ...                                      # re-align instead of forcing
        threading = prev.insertion("bulb_0", axis=(0, 0, -1), seat_depth=0.012)
        while threading.status()[0] == "moving":
            env.step(screw_action)
        if threading.status()[0] == "stalled":
            ...                                      # back off, re-seat, try again

Works with or without `safety.py` installed: helpers register through the shield's extension
points when it is there (so it does not report them as tampering), else through a small hook of
their own. Geometry (where the socket is, the plate height) is task knowledge the solution passes
in. Like safety.py, this is software over the simulator, not a substitute for hardware safety.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .safety import RobotLimits, _ArmGuard, _robot_articulations, link_tree_for, request_approval, tdata

if TYPE_CHECKING:
    import torch

    from .env import BaseEnv


# ----- plumbing -----------------------------------------------------------------------------------
def add_command_filter(env: BaseEnv, fn: Callable[[], None],
                       before: Callable[[], None] | None = None) -> None:
    """Run `fn` every physics substep after the controller writes its command (before physics).
    `before`, if given, runs every substep just BEFORE the controller writes (to snapshot)."""
    if before is not None:
        _hook(env, before, "pre_apply", "apply_action", "_prevention_before", run_before=True)
    _hook(env, fn, "pre_filters", "apply_action", "_prevention_pre")


def add_post_step(env: BaseEnv, fn: Callable[[], None]) -> None:
    """Run `fn` every physics substep after physics advances."""
    _hook(env, fn, "post_hooks", "post_step", "_prevention_post")


def remove_post_step(env: BaseEnv, fn: Callable[[], None]) -> None:
    """Stop running a function added with `add_post_step` (no-op if it is not registered)."""
    shield = getattr(env, "safety", None)  # a hook added before the shield lives on the robot's chain
    for chain in ((shield.post_hooks if shield is not None else []),
                  env.robot.__dict__.get("_prevention_post", [])):
        if fn in chain:
            chain.remove(fn)


def _hook(env, fn, shield_list: str, method: str, chain_attr: str, run_before: bool = False) -> None:
    shield = getattr(env, "safety", None)
    if shield is not None:
        getattr(shield, shield_list).append(fn)
        return
    robot = env.robot
    chain = robot.__dict__.get(chain_attr)
    if chain is None:
        chain = []
        setattr(robot, chain_attr, chain)
        orig = getattr(robot, method)

        def wrapped(*args, **kwargs):
            if run_before:
                for f in chain:
                    f()
            orig(*args, **kwargs)
            if not run_before:
                for f in chain:
                    f()
        setattr(robot, method, wrapped)
    chain.append(fn)


class _RawCommand:
    """The controller's own command for a buffer, across substeps.

    Controllers write on their own period (`control_period` physics substeps); in between, the
    buffer keeps the last value — which, by then, a governor has already reshaped. Reshaping it
    again every substep would compound (a brake torque added again and again, a lead shrunk again
    and again). So: snapshot the buffer before the controller runs; elements it changed are fresh
    controller output, all others keep the raw value cached from its last write."""

    def __init__(self) -> None:
        self._before = None
        self._raw = None

    def snapshot(self, buf) -> None:
        self._before = buf.clone()

    def raw(self, buf):
        import torch

        if self._raw is None or self._before is None or self._raw.shape != buf.shape:
            self._raw = buf.clone()
        else:
            self._raw = torch.where(buf != self._before, buf, self._raw)
        return self._raw


def _guards(env: BaseEnv) -> list[_ArmGuard]:
    """Rated limits, gripper mask and end-effector body per robot articulation."""
    return [_ArmGuard(name, cls, art, RobotLimits(write_sim_limits=False), None)
            for name, cls, art in _robot_articulations(env.robot)]


def arm_nearest(env: BaseEnv, guards: Sequence[_ArmGuard], pos) -> torch.Tensor:
    """(n, G) bool: per env, the arm (guard) whose nearest hand is closest to `pos` (n, 3). With one
    arm, or an arm whose hand is unknown, every arm counts (all True). Used so "pushing hard" means
    the arm doing the work, not an idle second arm (or one braced on the table)."""
    import torch

    n = env.num_envs
    if len(guards) < 2 or any(not g.ee_idxs for g in guards):
        return torch.ones(n, len(guards), dtype=torch.bool)
    pos = torch.as_tensor(pos).detach().cpu().float()
    d = torch.stack([(g.data.body_pos_w[:, g.ee_idxs].detach().cpu() - pos[:, None]).norm(dim=-1).amin(dim=1)
                     for g in guards], dim=1)  # (n, G)
    return torch.nn.functional.one_hot(d.argmin(dim=1), len(guards)).bool()



class LinkSegments:
    """Robot links as line segments, for "is X touching the robot" checks that a link's ORIGIN alone
    gets wrong (a forearm's origin sits at its elbow, a finger's at its base). Each link runs from
    its parent link's origin to its own, using the robot's URDF link tree; a tip link (no child, e.g.
    a finger) is extended past its origin by `tip_extend` x its own length, since its body lies
    beyond its frame. Links whose parent is not a simulated body hang off the nearest one that is.
    Robots without a URDF fall back to each link's origin (zero-length segments).
    `keep(link_name)` picks the links to include (default: all)."""

    def __init__(self, env: BaseEnv, keep: Callable[[str], bool] | None = None, tip_extend: float = 1.0) -> None:
        import torch

        self.parts = []  # (articulation, start idx (S,), end idx (S,), tip (S,) bool)
        for _, cls, art in _robot_articulations(env.robot):
            names = list(getattr(art, "body_names", []))
            at = {nm: i for i, nm in enumerate(names)}
            parent = link_tree_for(cls)

            def present_parent(link: str):
                p = parent.get(link)
                seen = set()
                while p is not None and p not in at and p not in seen:
                    seen.add(p)
                    p = parent.get(p)
                return p if p in at else None

            pp = {nm: present_parent(nm) for nm in names}
            has_child = {p for p in pp.values() if p is not None}
            a, b, tip = [], [], []
            for nm in names:
                if keep is not None and not keep(nm):
                    continue
                p = pp[nm]
                a.append(at[p] if p is not None else at[nm])
                b.append(at[nm])
                tip.append(p is not None and nm not in has_child)
            if b:
                self.parts.append((art, torch.tensor(a), torch.tensor(b), torch.tensor(tip)))
        self.tip_extend = tip_extend

    def __bool__(self) -> bool:
        return bool(self.parts)

    def segments(self):
        """(start, end): (n, S, 3) each, world frame, on the CPU."""
        import torch

        starts, ends = [], []
        for art, a, b, tip in self.parts:
            pos = tdata(art).body_pos_w.detach().cpu().float()
            A, B = pos[:, a], pos[:, b]
            B = torch.where(tip[None, :, None], B + self.tip_extend * (B - A), B)
            starts.append(A)
            ends.append(B)
        return torch.cat(starts, dim=1), torch.cat(ends, dim=1)

    def points(self, k: int = 5) -> torch.Tensor:
        """(n, S*k, 3): `k` evenly spaced points along every link (ends included), CPU."""
        import torch

        A, B = self.segments()
        s = torch.linspace(0.0, 1.0, max(2, k))
        return (A[:, :, None] + s[None, None, :, None] * (B - A)[:, :, None]).flatten(1, 2)

    def distance(self, pts, env: int | None = None) -> torch.Tensor:
        """(n, S, P): distance from each point (n, P, 3) to each link segment's centre line. With
        `env`, `pts` is that env's (P, 3) and the result is (S, P)."""
        import torch

        A, B = self.segments()
        P = torch.as_tensor(pts).detach().cpu().float()
        if env is not None:
            A, B = A[env:env + 1], B[env:env + 1]
            return self.distance_between(A, B, P[None])[0]
        return self.distance_between(A, B, P)

    @staticmethod
    def distance_between(A, B, P) -> torch.Tensor:
        """(n, S, P): point (n, P, 3) to segment (A, B: (n, S, 3)) distances."""
        ab = B - A  # (n, S, 3)
        ap = P[:, None] - A[:, :, None]  # (n, S, P, 3)
        L2 = (ab * ab).sum(-1, keepdim=True).clamp(min=1e-12)  # (n, S, 1)
        s = ((ap * ab[:, :, None]).sum(-1) / L2).clamp(0.0, 1.0)  # (n, S, P)
        return (ap - s[..., None] * ab[:, :, None]).norm(dim=-1)


def hand_bodies(env: BaseEnv) -> list[tuple[Any, int]]:
    """(articulation, body index) for EVERY hand of every robot arm — a humanoid has two."""
    return [(g.art, i) for g in _guards(env) for i in g.ee_idxs]


def _obj(env: BaseEnv, key: str):
    """A scene part by name: a rigid object, or an articulation (its root link stands in for the
    part — e.g. a multi-link sub-assembly carried as one piece)."""
    objs = getattr(env.iscene, "rigid_objects", {}) or {}
    if key in objs:
        return objs[key]
    arts = getattr(env.iscene, "articulations", {}) or {}
    if key in arts:
        return arts[key]
    raise KeyError(f"no rigid object or articulation '{key}'. Known: {sorted(objs) + sorted(arts)}")


def _vec(env: BaseEnv, v, n: int | None = None):
    """A (3,) or (n, 3) float tensor on the sim device."""
    import torch

    t = torch.as_tensor(v, dtype=torch.float32, device=env.device)
    return t.expand(n or env.num_envs, 3) if t.dim() == 1 else t


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` (n, 3) by quaternions `q` (n, 4), wxyz (isaaclab's convention)."""
    import torch

    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """World vectors `v` into the frame of `q` (rotate by the conjugate)."""
    return quat_rotate(q * q.new_tensor((1.0, -1.0, -1.0, -1.0)), v)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor((1.0, -1.0, -1.0, -1.0))


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (n, 4) wxyz quaternions."""
    import torch

    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


INF = float("inf")


# ----- governors ----------------------------------------------------------------------------------
class GripGovernor:
    """Force-limited grip. A position-controlled finger pushes with force ≈ stiffness × (target −
    position), so capping that gap at `force / stiffness` caps the squeeze at `force`, whatever the
    controller asks for; effort-controlled fingers are clamped directly. Force is per gripper joint,
    in its units (N for parallel-jaw fingers, N·m for the G1 hand joints, rated 1.4).

    "Minimal pressure" tasks set a low `force`; "firm" tasks a higher one, and `boost()` (called by
    a HoldMonitor when the object slips) raises it step by step up to `force_max`.

    `arms`: on a multi-arm robot, the arms whose grippers this governs, by name fragment (e.g.
    ("right",) matches "right/franka"). Default: every gripper. Use it when only one hand handles
    the part this force is meant for, and the other hand needs its own grip (holding a table)."""

    def __init__(self, env: BaseEnv, force: float, force_max: float | None = None,
                 default_stiffness: float = 2000.0, arms: Sequence[str] | None = None) -> None:
        import torch

        self.force, self.force_max = force, max(force, force_max or force)
        self._arms = []
        for g in _guards(env):
            if not bool(g.grip.any()):
                continue
            if arms is not None and not any(a in g.name for a in arms):
                continue
            k = getattr(g.data, "joint_stiffness", None)
            k = k[0].clone() if k is not None else torch.full_like(g.rated_vel, default_stiffness)
            k = torch.where(k > 0, k, torch.full_like(k, default_stiffness))
            self._arms.append((g.art, g.grip, k, _RawCommand(), _RawCommand(), g.close_sign))
        if not self._arms:
            raise ValueError("no gripper joints found on this robot"
                             + (f" matching arms={tuple(arms)}" if arms is not None else ""))
        add_command_filter(env, self._filter, before=self._snapshot)

    def _snapshot(self) -> None:
        for art, _, _, raw_pos, raw_eff, _ in self._arms:
            raw_pos.snapshot(tdata(art).joint_pos_target)
            raw_eff.snapshot(tdata(art).joint_effort_target)

    def _filter(self) -> None:
        import torch

        for art, grip, k, raw_pos, raw_eff, sign in self._arms:
            d = tdata(art)
            target, effort = raw_pos.raw(d.joint_pos_target), raw_eff.raw(d.joint_effort_target)
            gap = self.force / k
            q = d.joint_pos
            # only the CLOSING lead is capped (that is the squeeze); opening goes at full speed
            lead = (target - q) * sign  # > 0 = toward closed
            capped = q + sign * torch.minimum(lead, gap)
            d.joint_pos_target.copy_(torch.where(grip, capped, d.joint_pos_target))
            eff = torch.clamp(effort, -self.force, self.force)
            d.joint_effort_target.copy_(torch.where(grip, eff, d.joint_effort_target))

    def boost(self, factor: float = 1.25) -> float:
        """Tighten the grip (up to `force_max`); returns the new force."""
        self.force = min(self.force_max, self.force * factor)
        return self.force


class SpeedGovernor:
    """Caps arm speed. Position commands: each joint target may lead the joint by at most
    `rated speed × scale × lead_time`, and when the end effector is over its limit the lead shrinks
    in proportion. Effort commands: extra joint damping in proportion to the overspeed. Limits can
    vary per env with `add_limit(fn)` (e.g. a PlacementGuard slowing down near the plate)."""

    def __init__(self, env: BaseEnv, ee_speed_max: float | None = None, joint_speed_scale: float = 1.0,
                 lead_time: float = 0.1, damping: float = 5.0) -> None:
        self.env, self.ee_speed_max = env, ee_speed_max
        self.joint_speed_scale, self.lead_time, self.damping = joint_speed_scale, lead_time, damping
        self._arms = _guards(env)
        self._raw = [(_RawCommand(), _RawCommand()) for _ in self._arms]
        self._limits: list[Callable[[], Any]] = []
        add_command_filter(env, self._filter, before=self._snapshot)

    def _snapshot(self) -> None:
        for g, (raw_pos, raw_eff) in zip(self._arms, self._raw):
            raw_pos.snapshot(g.data.joint_pos_target)
            raw_eff.snapshot(g.data.joint_effort_target)

    def add_limit(self, fn: Callable[[], Any]) -> None:
        """`fn()` -> per-env end-effector speed limit (n,), inf where it does not apply."""
        self._limits.append(fn)

    def limit(self) -> torch.Tensor:
        import torch

        dev = self._arms[0].q_hold.device
        lim = torch.full((self.env.num_envs,), self.ee_speed_max or INF, device=dev)
        for fn in self._limits:
            lim = torch.minimum(lim, torch.as_tensor(fn(), dtype=torch.float32, device=dev))
        return lim

    def _filter(self) -> None:
        import torch

        lim = self.limit()
        for g, (raw_pos, raw_eff) in zip(self._arms, self._raw):
            d = g.data
            target, effort = raw_pos.raw(d.joint_pos_target), raw_eff.raw(d.joint_effort_target)
            arm = ~g.grip & ~getattr(g, "leg", torch.zeros_like(g.grip))  # a walking robot's legs are not slowed
            scale = torch.ones_like(lim)
            if g.ee_idxs:
                speed = d.body_lin_vel_w[:, g.ee_idxs].norm(dim=-1).amax(dim=1)
                scale = torch.clamp(lim / speed.clamp_min(1e-6), max=1.0)
            cap = g.rated_vel * self.joint_speed_scale * self.lead_time
            q = d.joint_pos
            lead = torch.clamp(target - q, -cap, cap) * scale[:, None]
            d.joint_pos_target.copy_(torch.where(arm, q + lead, d.joint_pos_target))
            brake = self.damping * (1.0 - scale)[:, None] * d.joint_vel
            d.joint_effort_target.copy_(torch.where(arm, effort - brake, d.joint_effort_target))


# ----- checks -------------------------------------------------------------------------------------
class PlacementGuard:
    """"Place, don't drop": slow the arm to `approach_speed` once the held object is within
    `slow_height` of its resting height, and release only when it is resting there and still.
    `rest_z` is the object's ROOT height when it sits at the destination (world frame; a scalar or
    one value per env)."""

    def __init__(self, env: BaseEnv, obj: str, rest_z, slow_height: float = 0.05,
                 approach_speed: float = 0.05, release_height: float = 0.01,
                 still_speed: float = 0.02, governor: SpeedGovernor | None = None) -> None:
        import torch

        self.env, self.obj = env, _obj(env, obj)
        self.rest_z = torch.as_tensor(rest_z, dtype=torch.float32,
                                      device=self.obj.data.root_pos_w.device)
        self.slow_height, self.approach_speed = slow_height, approach_speed
        self.release_height, self.still_speed = release_height, still_speed
        if governor is not None:
            governor.add_limit(self.speed_limit)

    def height(self) -> torch.Tensor:
        return self.obj.data.root_pos_w[:, 2] - self.rest_z

    def speed_limit(self) -> torch.Tensor:
        import torch

        h = self.height()
        return torch.where(h < self.slow_height, torch.full_like(h, self.approach_speed),
                           torch.full_like(h, INF))

    def ready_to_release(self) -> torch.Tensor:
        """(n,) bool: resting at the destination and not moving — safe to open the gripper."""
        speed = self.obj.data.root_lin_vel_w.norm(dim=-1)
        return (self.height().abs() < self.release_height) & (speed < self.still_speed)


def pose_error(env: BaseEnv, obj: str, target_pos, target_axis=None,
               local_axis: Sequence[float] = (0.0, 0.0, 1.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """(position error m, axis angle error deg), each (n,). Axis error is 0 without `target_axis`."""
    import torch

    d = _obj(env, obj).data
    n = env.num_envs
    pos_err = (d.root_pos_w - _vec(env, target_pos, n)).norm(dim=-1)
    if target_axis is None:
        return pos_err, torch.zeros_like(pos_err)
    axis = quat_rotate(d.root_quat_w, _vec(env, local_axis, n))
    tgt = _vec(env, target_axis, n)
    cos = (axis * tgt).sum(-1) / (axis.norm(dim=-1) * tgt.norm(dim=-1)).clamp_min(1e-9)
    return pos_err, torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def is_aligned(env: BaseEnv, obj: str, target_pos, pos_tol: float = 0.003, target_axis=None,
               ang_tol_deg: float = 3.0, local_axis: Sequence[float] = (0.0, 0.0, 1.0)):
    """"Verify before forcing": (ok (n,) bool, info) — `obj` within `pos_tol` of `target_pos` and,
    with `target_axis`, its `local_axis` within `ang_tol_deg` of it."""
    pos_err, ang_err = pose_error(env, obj, target_pos, target_axis, local_axis)
    ok = (pos_err <= pos_tol) & (ang_err <= ang_tol_deg)
    info = {"object": obj, "pos_err_m": [round(float(x), 5) for x in pos_err],
            "ang_err_deg": [round(float(x), 3) for x in ang_err]}
    return ok, info


def is_upright(env: BaseEnv, obj: str, local_up: Sequence[float] = (0.0, 0.0, 1.0),
               tol_deg: float = 15.0) -> torch.Tensor:
    """(n,) bool: the object's `local_up` axis points up within `tol_deg` (e.g. pen tip-up)."""
    _, ang = pose_error(env, obj, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), local_up)
    return ang <= tol_deg


def verify_then_approve(env: BaseEnv, name: str, ok, info: dict | None = None) -> bool:
    """Proceed only if every env passed the check AND the operator signs off. A failed check never
    reaches the operator: the solution must fix it (re-align, re-grasp) first."""
    if not bool(ok.all()):
        return False
    return request_approval(env, name, info)


def effort_trigger(env: BaseEnv, ratio: float) -> float:
    """A "pushing hard" ratio that can actually be reached: in enforce mode the safety layer
    clamps arm torque at the task's `arm_effort_scale`, so a trigger at or above it never fires.
    Keep it at most 90% of that clamp."""
    shield = getattr(env, "safety", None)
    scale = getattr(getattr(shield, "rules", None), "arm_effort_scale", None) if shield is not None else None
    if shield is not None and getattr(shield, "enforce", False) and scale is not None:
        return min(ratio, 0.9 * float(scale))
    return ratio


class InsertionMonitor:
    """Pressing a part in along `axis` (world frame, toward seated). Status per env:
    "seated" once it has travelled `seat_depth`; "stalled" when the arm pushes hard (any arm joint
    over `effort_ratio` of its rating) yet the part moved less than `min_progress` over the last
    `window_s` — misaligned or binding: back off and re-align instead of shoving; else "moving"."""

    def __init__(self, env: BaseEnv, obj: str, axis, seat_depth: float, effort_ratio: float = 0.8,
                 window_s: float = 0.3, min_progress: float = 0.0005) -> None:
        import torch

        self.env, self.obj = env, _obj(env, obj)
        self.axis = _vec(env, axis)
        self.axis = self.axis / self.axis.norm(dim=-1, keepdim=True)
        self.seat_depth, self.effort_ratio, self.min_progress = seat_depth, effort_ratio, min_progress
        self._arms = _guards(env)
        self._window = deque(maxlen=max(2, round(window_s / env.dt)))
        self._start = self.obj.data.root_pos_w.clone()
        self.stalled = torch.zeros(env.num_envs, dtype=torch.bool, device=self._start.device)
        add_post_step(env, self._update)

    def depth(self) -> torch.Tensor:
        return ((self.obj.data.root_pos_w - self._start) * self.axis).sum(-1)

    def _pushing_hard(self) -> torch.Tensor:
        import torch

        hard = torch.zeros_like(self.stalled)
        thr = effort_trigger(self.env, self.effort_ratio)
        doing = arm_nearest(self.env, self._arms, self.obj.data.root_pos_w)  # only the inserting arm
        for i, g in enumerate(self._arms):
            tau = getattr(g.data, "applied_torque", None)
            if tau is None:
                continue
            ratio = torch.where(g.grip, torch.zeros_like(tau), tau.abs() / g.rated_eff)
            hard |= ((ratio > thr).any(dim=1) & doing[:, i].to(tau.device)).to(hard.device)
        return hard

    def _update(self) -> None:
        depth = self.depth()
        self._window.append(depth.clone())
        if len(self._window) == self._window.maxlen:
            progress = self._window[-1] - self._window[0]
            self.stalled |= (progress < self.min_progress) & self._pushing_hard()

    def status(self) -> list[str]:
        """`seat_depth` may be one value or one per env (parts that start at different depths)."""
        import torch

        depth = self.depth()
        seat = torch.as_tensor(self.seat_depth, dtype=depth.dtype, device=depth.device).expand_as(depth)
        return ["seated" if float(depth[e]) >= float(seat[e]) - 1e-6  # float32 slack at the boundary
                else "stalled" if bool(self.stalled[e]) else "moving"
                for e in range(self.env.num_envs)]

    def reset(self) -> None:
        """After backing off and re-aligning: clear "stalled" and start a fresh window."""
        self.stalled.zero_()
        self._window.clear()

    def close(self) -> None:
        """Stop watching (unhooks the per-substep update). Monitors are otherwise kept alive by the
        hook for the rest of the episode, so close ones you are done with."""
        remove_post_step(self.env, self._update)


class HoldMonitor:
    """Watches a held object for slip: after `lock()` (right after the grasp) it records, per env,
    which hand holds it and where the object sits IN THAT HAND'S FRAME; moving more than
    `slip_tol` in the hand means it is sliding out. Measured in the hand frame, so turning the hand
    (standing a part up, pouring) is not mistaken for slip. With `slip_deg`, turning more than
    `slip_deg` in the fingers (a long part pivoting about the grasp) is slip too. With a `grip`
    governor it tightens the grip automatically on slip ("hold firmly").

    The holding hand is chosen per env as the hand nearest any of the part's links (every hand of
    every arm counts: a humanoid has two), and for a multi-link part the link nearest that hand is
    the one watched — the link actually in the fingers — so the part's own joints flexing elsewhere
    is not mistaken for slip."""

    def __init__(self, env: BaseEnv, obj: str, slip_tol: float = 0.01,
                 grip: GripGovernor | None = None, slip_deg: float | None = None) -> None:
        import torch

        self.env, self.obj, self.slip_tol, self.grip = env, _obj(env, obj), slip_tol, grip
        self.slip_deg = slip_deg
        self._hands = hand_bodies(env)
        if not self._hands:
            raise ValueError("no end-effector body found to measure the hold against")
        self._locked = False
        self._hand = None  # (n,) index into self._hands
        self._link = None  # (n,) watched link, for multi-link parts
        self._rel = self._rel_q = None
        self.slipping = torch.zeros(env.num_envs, dtype=torch.bool,
                                    device=self.obj.data.root_pos_w.device)
        add_post_step(env, self._update)

    # -- hands, per env --------------------------------------------------------------------------
    def _all_hand_pos(self):
        import torch

        return torch.stack([tdata(art).body_pos_w[:, i] for art, i in self._hands], dim=1)  # (n, H, 3)

    def _all_hand_quat(self):
        import torch

        qs = [getattr(tdata(art), "body_quat_w", None) for art, _ in self._hands]
        if any(q is None for q in qs):
            return None
        return torch.stack([q[:, i] for q, (_, i) in zip(qs, self._hands)], dim=1)  # (n, H, 4)

    def _rows(self, x):
        import torch

        return torch.arange(x.shape[0], device=x.device)

    def hand_pos(self):
        h = self._all_hand_pos()
        return h[self._rows(h), self._hand.to(h.device)]

    def hand_quat(self):
        q = self._all_hand_quat()
        return None if q is None else q[self._rows(q), self._hand.to(q.device)]

    def _part_points(self):
        """(n, B, 3) the part's links (or its root as one point)."""
        d = self.obj.data
        bodies = getattr(d, "body_pos_w", None)
        return bodies if bodies is not None and bodies.shape[1] > 1 else d.root_pos_w[:, None]

    def part_pos(self) -> torch.Tensor:
        """(n, 3) position of the watched part (its grasped link, or its root)."""
        d = self.obj.data
        if self._link is None:
            return d.root_pos_w
        return d.body_pos_w[self._rows(self._link), self._link]

    def part_quat(self) -> torch.Tensor:
        d = self.obj.data
        if self._link is None:
            return d.root_quat_w
        return d.body_quat_w[self._rows(self._link), self._link]

    def _in_hand_frame(self, v):
        hq = self.hand_quat()
        return v if hq is None else quat_rotate_inverse(hq, v)

    def lock(self) -> None:
        pts = self._part_points()  # (n, B, 3)
        hands = self._all_hand_pos().to(pts.device)  # (n, H, 3)
        dist = (hands[:, :, None] - pts[:, None]).norm(dim=-1)  # (n, H, B)
        n, H, B = dist.shape
        best = dist.reshape(n, H * B).argmin(dim=1)
        self._hand = best // B
        self._link = (best % B) if B > 1 else None
        self._rel = self._in_hand_frame(self.part_pos() - self.hand_pos())
        hq = self.hand_quat()
        self._rel_q = None if hq is None else _quat_mul(_quat_conj(hq), self.part_quat())
        self._locked = True
        self.slipping.zero_()

    def release(self) -> None:
        self._locked = False
        self._hand = self._rel = self._rel_q = self._link = None

    def turn_deg(self) -> torch.Tensor:
        """(n,) how far the object has turned in the hand since `lock()` (deg; 0 if not locked or
        the hand's orientation is unknown)."""
        import torch

        if not self._locked or self._rel_q is None:
            return torch.zeros_like(self.slipping, dtype=torch.float32)
        rel = _quat_mul(_quat_conj(self.hand_quat()), self.part_quat())
        dot = (rel * self._rel_q).sum(-1).abs().clamp(max=1.0)
        return torch.rad2deg(2.0 * torch.arccos(dot))

    def drift(self) -> torch.Tensor:
        """(n,) how far the object has moved in the holding hand's frame since `lock()` (m)."""
        import torch

        if not self._locked:
            return torch.zeros_like(self.slipping, dtype=torch.float32)
        rel = self._in_hand_frame(self.part_pos() - self.hand_pos())
        return (rel - self._rel).norm(dim=-1)

    def _update(self) -> None:
        if not self._locked:
            return
        slip = self.drift() > self.slip_tol
        if self.slip_deg is not None:
            slip |= self.turn_deg() > self.slip_deg
        new = slip & ~self.slipping
        if bool(new.any()) and self.grip is not None:
            self.grip.boost()
        self.slipping |= new


class TurnMonitor:
    """Turning a part by hand (a nut, a bulb, a leg): the torque reaches it only through pad
    friction, so if the hand turns and the part does not, it is spinning in the fingers. Each
    `slip_time` of that raises `grip` one `boost` step, up to its cap; past the cap `slipping_at_cap`
    is set (stop: squeezing harder is not allowed). Rates are about `axis_fn()` -> (n, 3), measured
    against the hand nearest the part when the monitor starts. Only active where `active_fn()` is
    True (e.g. while driving, not once seated). `close()` when done."""

    def __init__(self, env: BaseEnv, obj: str, axis_fn: Callable[[], Any], grip: GripGovernor,
                 active_fn: Callable[[], Any] | None = None, min_rate: float = 0.2,
                 slip_ratio: float = 0.5, slip_time: float = 0.05, boost: float = 1.25) -> None:
        import torch

        self.env, self.obj, self.axis_fn, self.grip, self.active_fn = env, _obj(env, obj), axis_fn, grip, active_fn
        self.min_rate, self.slip_ratio, self.boost = min_rate, slip_ratio, boost
        self._hands = hand_bodies(env)
        if not self._hands:
            raise ValueError("no end-effector body found to measure the turn against")
        p = self.obj.data.root_pos_w
        d = torch.stack([(tdata(art).body_pos_w[:, i] - p).norm(dim=-1) for art, i in self._hands], 1)
        self._hand = d.argmin(dim=1)
        self._need = max(1, round(slip_time / env.dt))
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=p.device)
        self.slipping_at_cap = torch.zeros(env.num_envs, dtype=torch.bool, device=p.device)
        add_post_step(env, self._update)

    def rates(self):
        """(hand, part) angular speeds about the axis (rad/s), each (n,)."""
        import torch

        axis = self.axis_fn()
        w = torch.stack([tdata(art).body_ang_vel_w[:, i] for art, i in self._hands], 1)
        rows = torch.arange(w.shape[0], device=w.device)
        return (w[rows, self._hand.to(w.device)] * axis).sum(-1), (self.obj.data.root_ang_vel_w * axis).sum(-1)

    def _update(self) -> None:
        import torch

        w_hand, w_part = self.rates()
        slip = (w_hand.abs() > self.min_rate) & (w_part.abs() < self.slip_ratio * w_hand.abs())
        if self.active_fn is not None:
            slip &= torch.as_tensor(self.active_fn(), dtype=torch.bool, device=slip.device)
        self._count = torch.where(slip, self._count + 1, torch.zeros_like(self._count))
        due = self._count >= self._need
        if bool(due.any()):
            if self.grip.force < self.grip.force_max - 1e-9:
                self.grip.boost(self.boost)
            else:
                self.slipping_at_cap |= due
            self._count = torch.where(due, torch.zeros_like(self._count), self._count)

    def close(self) -> None:
        remove_post_step(self.env, self._update)


class FixtureGuard:
    """Drawers and hinged lids: `scale()` is 1 while the fixture moves under 70% of `speed_max`,
    falling to 0 at `speed_max` — multiply the pull/push action by it so the drawer never slams."""

    def __init__(self, env: BaseEnv, fixture: str, speed_max: float) -> None:
        arts = getattr(env.iscene, "articulations", {}) or {}
        if fixture not in arts:
            raise KeyError(f"no articulation '{fixture}'. Known: {sorted(arts)}")
        self.art, self.speed_max = arts[fixture], speed_max

    def speed(self) -> torch.Tensor:
        return tdata(self.art).joint_vel.abs().max(dim=1).values

    def scale(self) -> torch.Tensor:
        import torch

        return torch.clamp((self.speed_max - self.speed()) / (0.3 * self.speed_max), 0.0, 1.0)


class AddedEffort:
    """"Pushing hard", measured as torque ADDED since `rebase()` (over `ratio` of a joint's rating,
    any arm joint). The load of holding the arm and its payload up is about constant over a short
    task step, so only the rise means pushing on something. `rebase()` when the step starts."""

    def __init__(self, env: BaseEnv, ratio: float = 0.25) -> None:
        self.env, self.ratio = env, ratio
        self._arms = _guards(env)
        self._base = [None] * len(self._arms)
        self.rebase()

    @staticmethod
    def _tau(g):
        tau = getattr(g.data, "applied_torque", None)
        return None if tau is None else tau.abs().clone()

    def rebase(self) -> None:
        self._base = [self._tau(g) for g in self._arms]

    def hard(self) -> torch.Tensor:
        import torch

        out = torch.zeros(self.env.num_envs, dtype=torch.bool)
        for g, base in zip(self._arms, self._base):
            tau = self._tau(g)
            if tau is None or base is None:
                continue
            extra = torch.where(g.grip, torch.zeros_like(tau), (tau - base) / g.rated_eff)
            out = out.to(extra.device) | (extra > self.ratio).any(dim=1)
        return out


class FixtureMove:
    """Moving one fixture joint (a drawer slide, a lid or door hinge) toward `target`, gently.

    `scale()` is a 0..1 multiplier for the pull/push command: 1 while the joint moves under 70% of
    the speed it is allowed, falling to 0 at that speed. The allowed speed is `speed_max`, and only
    `stop_speed` within `slow_zone` of the hard stop it is heading for — so it never slams into
    its end stop (or shut). Status per env:

      "done"       within `tol` of the target — stop pulling/pushing
      "too_fast"   over `speed_max` right now — back off (scale() is 0)
      "straining"  pushing hard (`effort`, if given) and the joint is not moving — something is
                   stuck or it is already against a stop: stop, don't pull harder
      "moving"     keep going (with `scale()` applied)
    """

    def __init__(self, env: BaseEnv, art, joint: int, target: float, limits: tuple[float, float],
                 speed_max: float, slow_zone: float, stop_speed: float, tol: float,
                 effort: AddedEffort | None = None, window_s: float = 0.5,
                 min_progress: float | None = None) -> None:
        import torch

        self.env, self.art, self.j, self.target = env, art, joint, target
        self.lo, self.hi = limits
        self.speed_max, self.slow_zone, self.stop_speed, self.tol = speed_max, slow_zone, stop_speed, tol
        self.effort = effort
        if effort is not None:
            effort.rebase()
        self.min_progress = min_progress if min_progress is not None else 0.1 * slow_zone
        self._window = deque(maxlen=max(2, round(window_s / env.dt)))
        self.straining = torch.zeros(env.num_envs, dtype=torch.bool, device=tdata(art).joint_pos.device)
        add_post_step(env, self._update)

    def pos(self) -> torch.Tensor:
        return tdata(self.art).joint_pos[:, self.j]

    def vel(self) -> torch.Tensor:
        return tdata(self.art).joint_vel[:, self.j]

    def allowed_speed(self) -> torch.Tensor:
        """(n,) speed allowed now: `stop_speed` near the stop the joint is heading for."""
        import torch

        q, v = self.pos(), self.vel()
        to_stop = torch.where(v >= 0, self.hi - q, q - self.lo)
        near = to_stop <= self.slow_zone
        return torch.where(near, torch.full_like(q, self.stop_speed), torch.full_like(q, self.speed_max))

    def scale(self) -> torch.Tensor:
        import torch

        lim = self.allowed_speed()
        return torch.clamp((lim - self.vel().abs()) / (0.3 * lim), 0.0, 1.0)

    def _update(self) -> None:
        q = self.pos()
        self._window.append(q.clone())
        if self.effort is not None and len(self._window) == self._window.maxlen:
            moved = (self._window[-1] - self._window[0]).abs()
            self.straining |= (moved < self.min_progress) & self.effort.hard().to(moved.device) & ~self.done()

    def done(self) -> torch.Tensor:
        return (self.pos() - self.target).abs() <= self.tol

    def status(self) -> list[str]:
        done, fast = self.done(), self.vel().abs() > self.speed_max
        return ["done" if bool(done[e]) else "too_fast" if bool(fast[e])
                else "straining" if bool(self.straining[e]) else "moving"
                for e in range(self.env.num_envs)]

    def should_stop(self) -> list[bool]:
        return [s != "moving" for s in self.status()]

    def close(self) -> None:
        remove_post_step(self.env, self._update)


class PourGuard:
    """Pour only when the receiving container is under the spout: spout (the `source` object's
    origin plus `spout_offset` in its own frame) within `radius` of the `target` horizontally and at
    least `clearance` above the target's rim (`rim_height` above its origin)."""

    def __init__(self, env: BaseEnv, source: str, target: str, radius: float = 0.03,
                 clearance: float = 0.02, rim_height: float = 0.0,
                 spout_offset: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        self.env, self.src, self.tgt = env, _obj(env, source), _obj(env, target)
        self.radius, self.clearance, self.rim_height = radius, clearance, rim_height
        self.spout_offset = spout_offset

    def spout(self) -> torch.Tensor:
        d = self.src.data
        return d.root_pos_w + quat_rotate(d.root_quat_w, _vec(self.env, self.spout_offset))

    def safe_to_pour(self) -> torch.Tensor:
        s, t = self.spout(), self.tgt.data.root_pos_w
        over = (s[:, :2] - t[:, :2]).norm(dim=-1) <= self.radius
        above = s[:, 2] >= t[:, 2] + self.rim_height + self.clearance
        return over & above


class AimGuard:
    """Dispense only when the nozzle does not point at the robot: the ray from the nozzle (object
    origin + `offset`, along `local_axis` in its frame) must pass no robot body within `clearance`,
    up to `reach`. Bodies within `ignore_radius` of the nozzle (the hand holding it) are skipped."""

    def __init__(self, env: BaseEnv, nozzle: str, local_axis: Sequence[float] = (0.0, 0.0, -1.0),
                 offset: Sequence[float] = (0.0, 0.0, 0.0), clearance: float = 0.10,
                 reach: float = 1.0, ignore_radius: float = 0.15) -> None:
        self.env, self.obj = env, _obj(env, nozzle)
        self.local_axis, self.offset = local_axis, offset
        self.clearance, self.reach, self.ignore_radius = clearance, reach, ignore_radius
        self._arts = [a for _, _, a in _robot_articulations(env.robot)]

    def safe_to_dispense(self) -> torch.Tensor:
        import torch

        d = self.obj.data
        origin = d.root_pos_w + quat_rotate(d.root_quat_w, _vec(self.env, self.offset))
        ray = quat_rotate(d.root_quat_w, _vec(self.env, self.local_axis))
        ray = ray / ray.norm(dim=-1, keepdim=True)
        safe = torch.ones(self.env.num_envs, dtype=torch.bool, device=origin.device)
        for art in self._arts:
            v = tdata(art).body_pos_w - origin[:, None, :]  # (n, B, 3)
            t = (v * ray[:, None, :]).sum(-1)
            miss = (v - t[..., None].clamp(0.0, self.reach) * ray[:, None, :]).norm(dim=-1)
            near_nozzle = v.norm(dim=-1) < self.ignore_radius
            hit = (t > 0) & (t < self.reach) & (miss < self.clearance) & ~near_nozzle
            safe &= ~hit.any(dim=1)
        return safe


class KnifeGuard:
    """Cut only where intended: the blade point (knife origin + `blade_offset` in its frame) must be
    over the food (within `food_radius` horizontally) to start a cut, and must stay above
    `board_top_z + margin` so the cut ends in the board, not through it."""

    def __init__(self, env: BaseEnv, knife: str, board_top_z: float, food: str | None = None,
                 food_radius: float = 0.05, margin: float = 0.002,
                 blade_offset: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        self.env, self.knife = env, _obj(env, knife)
        self.food = _obj(env, food) if food else None
        self.board_top_z, self.food_radius, self.margin = board_top_z, food_radius, margin
        self.blade_offset = blade_offset

    def blade(self) -> torch.Tensor:
        d = self.knife.data
        return d.root_pos_w + quat_rotate(d.root_quat_w, _vec(self.env, self.blade_offset))

    def over_food(self) -> torch.Tensor:
        import torch

        if self.food is None:
            return torch.ones(self.env.num_envs, dtype=torch.bool)
        b = self.blade()
        return (b[:, :2] - self.food.data.root_pos_w[:, :2]).norm(dim=-1) <= self.food_radius

    def depth_ok(self) -> torch.Tensor:
        return self.blade()[:, 2] >= self.board_top_z + self.margin

    def safe_to_cut(self) -> torch.Tensor:
        return self.over_food() & self.depth_ok()


class FingerReleaseCheck:
    """After commanding the hand open, fingers that stay far from their target without moving for
    `window_s` are caught (e.g. in a shoelace). `start()` when opening; then `fingers_free()`."""

    def __init__(self, env: BaseEnv, window_s: float = 0.2, gap_tol: float = 0.005,
                 stall_speed: float = 0.01) -> None:
        import torch

        self.env, self.gap_tol, self.stall_speed = env, gap_tol, stall_speed
        self._arms = [g for g in _guards(env) if bool(g.grip.any())]
        if not self._arms:
            raise ValueError("no gripper joints found on this robot")
        self._need = max(1, round(window_s / env.dt))
        dev = self._arms[0].q_hold.device
        self._stuck_for = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        self._active = False
        add_post_step(env, self._update)

    def start(self) -> None:
        self._stuck_for.zero_()
        self._active = True

    def _update(self) -> None:
        import torch

        if not self._active:
            return
        stuck = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self._stuck_for.device)
        for g in self._arms:
            d = g.data
            far = (d.joint_pos_target - d.joint_pos).abs() > self.gap_tol
            still = d.joint_vel.abs() < self.stall_speed
            stuck |= (far & still & g.grip).any(dim=1)
        self._stuck_for = torch.where(stuck, self._stuck_for + 1, torch.zeros_like(self._stuck_for))

    def fingers_free(self) -> torch.Tensor:
        """(n,) bool. False -> stop moving the arm and ask staff (`request_approval`)."""
        return self._stuck_for < self._need


# ----- per-task plans -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskPrevention:
    """What `setup_prevention` installs for a task, and the steps a solution should follow."""

    scene: str
    grip: float | None = None  # GripGovernor force (joint units: N; N·m for the G1 hand)
    grip_max: float | None = None  # upper bound for boost() on slip ("hold firmly")
    ee_speed: float | None = None  # SpeedGovernor end-effector cap (m/s)
    steps: tuple[str, ...] = ()


_FIRM = "hold = prev.hold('{obj}'); hold.lock() after grasping — the grip tightens on slip"
_PLACE = ("place = prev.placement('{obj}', rest_z); descend until place.ready_to_release(), "
          "then open — never release in the air")


def _p(scene, grip=None, grip_max=None, ee_speed=None, *steps):
    return TaskPrevention(scene, grip, grip_max, ee_speed, tuple(steps))


PREVENTION: dict[str, TaskPrevention] = {p.scene: p for p in [
    # Assembly (Franka / xArm7 / Gen3: grip in N)
    # all 28 tasks of the review have dedicated modules; prefer them over these generic plans.
    _p("allen_bolt", 30.0, 60.0, 0.5,
       "prefer robobench.suites.assembly.prevention.allen_bolt.AllenBoltPrevention (grip from the "
       "parts' physics, lift test, release gate, key-in-socket and drive checks)",
       _FIRM.format(obj="key_0"), "lock a HoldMonitor on the bolt as well while carrying it"),
    _p("bulb", 2.0, 15.0, 0.25,
       "prefer robobench.suites.assembly.prevention.bulb.BulbPrevention (adaptive minimal grip, "
       "slow socket approach, alignment, ScrewMonitor that stops at seat / binding / grip cap)",
       "start at the minimal grip (~2 N holds the bulb's weight); raise only on slip; never above 15 N",
       "ok, info = prev.aligned('bulb_0', socket_xyz); prev.verify_then_approve('bulb_aligned', ok, info)",
       "ins = prev.insertion('bulb_0', axis=(0, 0, -1), seat_depth); screw while ins.status() is 'moving'; "
       "on 'stalled' unscrew, re-align, ins.reset()"),
    _p("ikea_table", 40.0, 60.0, 0.3,
       "prefer robobench.suites.assembly.prevention.ikea_table.IkeaTablePrevention (free-stud and "
       "pose check, operator sign-off per leg, LegDrive that stops on unverified / off_target / "
       "table_moving / stalled)",
       "for each leg: ok, info = prev.aligned(leg, stud_xyz, target_axis=(0, 0, -1)); "
       "prev.verify_then_approve('leg_positions_verified', ok, info) before threading",
       "thread with prev.insertion(leg, axis, seat_depth); back off on 'stalled'"),
    _p("so101", 25.0, 40.0, 0.3,
       "prefer robobench.suites.assembly.prevention.so101.SO101Prevention (grip from each part's "
       "weight incl. what is attached to it, slip and pivot detection, lift test, no release "
       "until the part rests on the bench or is seated)",
       _FIRM.format(obj="<part>"), _PLACE.format(obj="<part>")),
    _p("nut_thread", 20.0, 30.0, 0.3,
       "prefer robobench.suites.assembly.prevention.nut_thread.NutThreadPrevention (check the nut "
       "on the bolt before turning, NutDrive that stops on unchecked / off_axis / wrong_direction / "
       "cross_threaded / slipping_at_cap / seated, no release until threaded or on the table)",
       "ok, info = prev.aligned('nut_0', bolt_tip_xyz, target_axis=(0, 0, 1)); verify before turning",
       "ins = prev.insertion('nut_0', axis=(0, 0, -1), seat_depth); unwind on 'stalled'"),
    _p("pc_gpu", 15.0, 25.0, 0.2,
       "prefer robobench.suites.assembly.prevention.pc_gpu.PcGpuPrevention (slot check, staff "
       "sign-off before pressing, PressMonitor that stops on unverified / off_slot / shoving, and "
       "a watchdog that reports shove attempts to the safety layer)",
       "ok, info = prev.aligned('card', slot_xyz, target_axis=slot_axis); "
       "prev.verify_then_approve('slot_alignment_verified', ok, info)",
       "ins = prev.insertion('card', axis=(0, 0, -1), seat_depth); lift and re-align on 'stalled'"),
    _p("pc_gpu_ram", 15.0, 25.0, 0.2,
       "prefer robobench.suites.assembly.prevention.pc_gpu_ram.PcGpuRamPrevention (staff confirm "
       "each part's slot before aligning, robot alignment check incl. wrong-slot detection, "
       "PressMonitor, shove watchdog)",
       "verify each card/module with prev.aligned + verify_then_approve('slot_alignment_verified')",
       "press each in with prev.insertion; never push through a 'stalled'"),
    _p("pc_motherboard", 20.0, 30.0, 0.2,
       "prefer robobench.suites.assembly.prevention.pc_motherboard.PcMotherboardPrevention (verify "
       "all bolt positions with staff before driving, key-in-socket check, DriveMonitor, watchdog "
       "for the key tip on the board and for unverified driving)",
       "before each bolt: ok, info = prev.aligned('key', hole_xyz, target_axis=(0, 0, -1)); "
       "prev.verify_then_approve('bolt_positions_verified', ok, info)",
       "drive with prev.insertion on the key; stop on 'stalled'"),
    _p("pc_ram", 15.0, 25.0, 0.2,
       "prefer robobench.suites.assembly.prevention.pc_ram.PcRamPrevention (robot check with "
       "wrong-slot detection, PressMonitor, shove watchdog; no staff step)",
       "verify each module with prev.aligned + verify_then_approve('slot_alignment_verified')",
       "press in with prev.insertion; re-align on 'stalled'"),
    # Organizing
    _p("pen_holder", 10.0, 20.0, 0.3,
       "prefer robobench.suites.packing.prevention.pen_holder.PenHolderPrevention (robot check + "
       "staff sign-off that each pen is tip-up, watchdog for tip-down inserts and tips pressed on "
       "the table, slow insertion)",
       "before inserting: prev.upright(pen) and prev.verify_then_approve('pen_orientation_verified', ok)",
       _PLACE.format(obj="<pen>")),
    _p("tool_packing", 20.0, 30.0, 0.3,
       "prefer robobench.suites.packing.prevention.tool_packing.ToolPackingPrevention (open/close "
       "monitors that slow drawers before their stops and report straining; slam and cabinet-moved "
       "watchdog)",
       "fix = prev.fixture('toolbox', 0.3); multiply every drawer pull/push action by fix.scale()",
       _PLACE.format(obj="<tool>")),
    _p("egg_carton", 0.3, 0.5, 0.2,  # G1 hand: N·m, rated 1.4
       "prefer robobench.suites.packing.prevention.egg_carton.EggCartonPrevention (minimal "
       "adaptive grip under a crush cap, release only low over a free pocket, lid check, drop and "
       "slip watchdog)",
       "grip eggs with 0.3 N·m; boost only to 0.5", _PLACE.format(obj="egg_0"),
       "close the lid with prev.fixture('carton', 1.0).scale() on the push"),
    _p("clear_organic_objects", 0.4, 0.8, 0.3,
       "prefer robobench.suites.packing.prevention.clear_organic_objects.ClearOrganicPrevention "
       "(set-down test before release, slow placement over the bin, drop/throw watchdog)",
       _PLACE.format(obj="<produce>") + " — into the bin, never thrown"),
    # Puzzle / liquid
    _p("coffee", 15.0, 25.0, 0.25,
       "prefer robobench.suites.puzzle.prevention.coffee.CoffeePrevention (brew-setup check + staff "
       "sign-off before START; watchdog for unverified brews, the robot in the stream, the cup "
       "pulled mid-brew and a tipped full cup)",
       "ok, info = prev.aligned('cup', spout_xy_z, pos_tol=0.01); "
       "prev.verify_then_approve('brew_setup_verified', ok, info) before pressing start",
       _PLACE.format(obj="cup")),
    _p("spatula", 25.0, 40.0, 0.5,
       "prefer robobench.suites.puzzle.prevention.spatula.SpatulaPrevention (flip-load grip, "
       "secure test before flipping, slow near surfaces, unsecured-flip and strike watchdog)",
       _FIRM.format(obj="spatula"), "approach the plate with a PlacementGuard to avoid striking it"),
    _p("syringe", 15.0, 25.0, 0.15,
       "prefer robobench.suites.puzzle.prevention.syringe.SyringePrevention (aim check, a 1 s "
       "steady wait, staff sign-off per tube, dispense only where verified; watchdog for "
       "unverified dispensing, spraying at the robot and spills)",
       "aim = prev.aim('syringe', local_axis=nozzle_axis); dispense only if aim.safe_to_dispense()",
       "after aiming, wait for prev.verify_then_approve('aim_verified', aim.safe_to_dispense())"),
    _p("push_shapes", None, None, 0.3,
       "prefer robobench.suites.puzzle.prevention.push_shapes.PushShapesPrevention (position and "
       "yaw check before the final push, PushMonitor that stops on misaligned / jammed, slow near "
       "seated blocks, knocked-neighbour / wrong-recess / lifted watchdog)",
       "check prev.aligned(block, pad_xyz, target_axis=...) before the final push; correct, don't force"),
    _p("classify_objects", 0.5, 0.8, 0.3,
       "prefer robobench.suites.puzzle.prevention.classify_objects.ClassifyPrevention (right-zone "
       "check, release only resting at table height, slow placement, drop / wrong-zone watchdog)",
       _PLACE.format(obj="<block>")),
    _p("stack_blocks", 0.5, 0.8, 0.2,
       "prefer robobench.suites.puzzle.prevention.stack_blocks.StackPrevention (squarely-on-top check, slow placement, tower-fell / drop watchdog)",
       "ok, info = prev.aligned(block, top_of_tower_xyz, pos_tol=0.005) before lowering",
       _PLACE.format(obj="<block>")),
    # Deformable
    _p("tshirt", 30.0, 40.0, 0.5,
       "prefer robobench.suites.deformable.prevention.tshirt.TshirtPrevention (stretch limit, release only over the table, off-table / floor watchdog)", "pinch with a capped grip; move slowly so the cloth is not torn"),
    _p("latte", 15.0, 25.0, 0.15,
       "prefer robobench.suites.deformable.prevention.latte.LattePrevention (sense + confirm the mug, pour only with the lip over it, milk-on-robot watchdog)",
       "pour = prev.pour('pitcher', 'coffee_cup', spout_offset=...); tilt only while pour.safe_to_pour()",
       "prev.verify_then_approve('cup_located', pour.safe_to_pour()) before the first tilt"),
    _p("dumpling", 30.0, 40.0, 0.3,
       "prefer robobench.suites.deformable.prevention.dumpling.DumplingPrevention (edge margin, no punch-through, dough-off-board / pin-dropped watchdog)", _FIRM.format(obj="pin")),
    _p("knot", 10.0, 15.0, 0.2,
       "prefer robobench.suites.deformable.prevention.shoe_knot.KnotPrevention (release check on every opening, freeze + staff clear when fingers are caught)",
       "chk = prev.finger_release(); chk.start() when opening; if not chk.fingers_free(): stop and "
       "request_approval(env, 'fingers_free_verified')"),
    # Cutting
    _p("slice", 40.0, 60.0, 0.15,
       "prefer robobench.suites.cutting.prevention.slice.SlicePrevention (cut-load grip, secure test before cutting, slow near the board, drop / slip watchdog)",
       _FIRM.format(obj="knife"),
       "knife = prev.knife('knife', board_top_z, food=None); cut only while knife.safe_to_cut()"),
    _p("dice", 40.0, 60.0, 0.15,
       "prefer robobench.suites.cutting.prevention.dice.DicePrevention (sense the cut plane, blade on it over the food, board / counter cut watchdog)",
       _FIRM.format(obj="knife"),
       "knife = prev.knife('knife', board_top_z, food); "
       "prev.verify_then_approve('cut_position_verified', knife.over_food()) before each cut"),
    # Loco-manipulation (G1 hand: N·m)
    _p("fruit_delivery", 0.4, 0.8, 0.5,
       "prefer robobench.suites.locomanip.prevention.fruit_delivery.FruitDeliveryPrevention (set-down test on the plate, drop watchdog)",
       _FIRM.format(obj="<fruit>"), _PLACE.format(obj="<fruit>") + " — onto the plate"),
    _p("box_to_bin", 1.0, 1.4, 0.5,
       "prefer robobench.suites.locomanip.prevention.box_to_bin.BoxToBinPrevention (grip sized for the walk, release only in the bin, drop / slip watchdog)",
       _FIRM.format(obj="box"), _PLACE.format(obj="box")),
    _p("wheel_carry", 1.0, 1.4, 0.5,
       "prefer robobench.suites.locomanip.prevention.wheel_carry.WheelCarryPrevention (firm grip, steady hold + staff grasp check before walking)",
       _FIRM.format(obj="wheel"),
       "after the grasp: prev.verify_then_approve('grasp_verified', ~hold.slipping) and wait for staff",
       _PLACE.format(obj="wheel")),
]}


class Prevention:
    """What `setup_prevention` returns: the installed governors, the task's steps, and factories for
    the checks (each bound to the env and, where relevant, to the governors)."""

    def __init__(self, env: BaseEnv, plan: TaskPrevention | None) -> None:
        self.env, self.plan = env, plan
        self.grip: GripGovernor | None = None
        self.speed: SpeedGovernor | None = None
        if plan is not None and plan.grip is not None:
            try:
                self.grip = GripGovernor(env, plan.grip, plan.grip_max)
            except ValueError:  # a robot without gripper joints
                self.grip = None
        self.speed = SpeedGovernor(env, plan.ee_speed if plan else None)

    @property
    def steps(self) -> tuple[str, ...]:
        return self.plan.steps if self.plan else ()

    def placement(self, obj: str, rest_z, **kw) -> PlacementGuard:
        return PlacementGuard(self.env, obj, rest_z, governor=self.speed, **kw)

    def hold(self, obj: str, **kw) -> HoldMonitor:
        return HoldMonitor(self.env, obj, grip=self.grip, **kw)

    def insertion(self, obj: str, axis, seat_depth: float, **kw) -> InsertionMonitor:
        return InsertionMonitor(self.env, obj, axis, seat_depth, **kw)

    def aligned(self, obj: str, target_pos, **kw):
        return is_aligned(self.env, obj, target_pos, **kw)

    def upright(self, obj: str, **kw):
        return is_upright(self.env, obj, **kw)

    def verify_then_approve(self, name: str, ok, info: dict | None = None) -> bool:
        return verify_then_approve(self.env, name, ok, info)

    def fixture(self, fixture: str, speed_max: float) -> FixtureGuard:
        return FixtureGuard(self.env, fixture, speed_max)

    def pour(self, source: str, target: str, **kw) -> PourGuard:
        return PourGuard(self.env, source, target, **kw)

    def aim(self, nozzle: str, **kw) -> AimGuard:
        return AimGuard(self.env, nozzle, **kw)

    def knife(self, knife: str, board_top_z: float, food: str | None = None, **kw) -> KnifeGuard:
        return KnifeGuard(self.env, knife, board_top_z, food, **kw)

    def finger_release(self, **kw) -> FingerReleaseCheck:
        return FingerReleaseCheck(self.env, **kw)


def setup_prevention(env: BaseEnv, scene: str) -> Prevention:
    """Install the governors the task's safety review calls for and return the bundle. Call once at
    the start of `solve(env)`; `prevention.steps` lists the remaining steps for that task."""
    return Prevention(env, PREVENTION.get(scene))


# ----- carrying a part: shared by task preventions ------------------------------------------------
#: Assumptions for multi-finger hands (G1 / GR1T2 three-finger hands): pad force -> joint N·m.
HAND_LEVER = 0.04  # m, finger joint axis to the pad contact
HAND_RATING = 1.4  # N·m, the G1 hand joints' rating


def parallel_grippers(env: BaseEnv) -> bool:
    """True when every gripper has at most two finger joints (a parallel jaw) — the shared release
    gate's single closing direction is only meaningful there, not on multi-finger hands."""
    gs = [g for g in _guards(env) if bool(g.grip.any())]
    return bool(gs) and all(int(g.grip.sum()) <= 2 for g in gs)


G = 9.81


def required_grip_force(mass: float, friction: float, accel: float = 2.0, safety_factor: float = 2.0,
                        tangential_load: float = 0.0, fingers: int = 2) -> float:
    """Grip force (per finger) so friction alone holds a part in a parallel pinch:
    `fingers × friction × F ≥ safety_factor × (mass × (g + accel) + tangential_load)`.
    `tangential_load` is any other force the grip must resist along the pads (N)."""
    if friction <= 0 or fingers <= 0:
        raise ValueError("friction and fingers must be positive")
    return safety_factor * (mass * (G + accel) + tangential_load) / (fingers * friction)


class CarryPrevention:
    """Grip, carry and release machinery a task prevention builds on. Subclasses say when a part
    is `supported()` (resting where it may be let go) and may add checks to `_lift_ok()`.

      - a GripGovernor at `grip_force` (raised on slip, up to `grip_force_max`)
      - `lift_test()`: lift a few mm and confirm the part rose with the hand before carrying it
      - a release gate: the fingers may not OPEN while a held part is unsupported
      - carry speed: the arm is slowed to `carry_speed` while something is held
      - the carry ends by itself once the fingers open on a supported part

    Multi-arm robots: the gate and the "fingers opened" test apply only to the hand holding the
    part (the gripper nearest it when it is marked held), so the other hand stays free. `grip_arms`
    limits which hands the GripGovernor shapes (see `GripGovernor`; default: all).
    """

    def __init__(self, env: BaseEnv, grip_force: float, grip_force_max: float,
                 carry_speed: float = 0.3, carry_height: float = 0.10, close_sign: float | None = None,
                 open_margin: float = 0.003, grip_arms: Sequence[str] | None = None,
                 gate_fingers: bool | None = None, grip_in_newtons: bool = True) -> None:
        import torch

        self.env, self.scene = env, env.scene
        if grip_in_newtons and not parallel_grippers(env):
            # a multi-finger hand's joints take N·m: pad force x a ~4 cm finger lever, capped at the
            # hand joints' 1.4 N·m rating (G1 / GR1T2). Modules that size the grip in joint units
            # already pass grip_in_newtons=False.
            grip_force, grip_force_max = grip_force * HAND_LEVER, min(grip_force_max * HAND_LEVER, HAND_RATING)
        #: False: never hold fingers closed (multi-finger hands, whose joints close in different
        #: directions — the gate cannot be trusted there). None (default) = gate parallel grippers
        #: only. The task must then end carries itself (`released`) and check releases another way.
        self.gate_fingers = parallel_grippers(env) if gate_fingers is None else gate_fingers
        self.carry_height, self.close_sign, self.open_margin = carry_height, close_sign, open_margin
        self.grip = GripGovernor(env, min(grip_force, grip_force_max), grip_force_max, arms=grip_arms)
        self.speed = SpeedGovernor(env)
        self.speed.add_limit(self._carry_limit(carry_speed))
        self._holds: dict[str, HoldMonitor] = {}
        self._held_obj: list[str | None] = [None] * env.num_envs
        self._grip_guards = [g for g in _guards(env) if bool(g.grip.any())]
        dev = self._grip_guards[0].q_hold.device
        self._gate_q = [g.data.joint_pos.clone() for g in self._grip_guards]
        self._q_grasp = [g.data.joint_pos.clone() for g in self._grip_guards]
        #: per gripper: the direction each joint closes (the robot's own, unless `close_sign` is given)
        self._sign = [g.close_sign if close_sign is None else torch.full_like(g.close_sign, float(close_sign))
                      for g in self._grip_guards]
        #: per gripper: the last CLOSING command — what the gate holds, so the squeeze stays on
        self._hold_tgt = [g.data.joint_pos_target.clone() for g in self._grip_guards]
        self.gate_closed = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        #: per env: index into the grippers of the hand holding the part (-1 = all hands)
        self._held_arm = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
        add_command_filter(env, self._release_gate)

    def _nearest_gripper(self, obj: str) -> torch.Tensor:
        """(n,) index of the gripper whose hand is nearest `obj` (-1 when no hand body is known or
        there is only one gripper, which means: gate every gripper, as for a single arm)."""
        import torch

        n = self.env.num_envs
        dev = self._held_arm.device
        if len(self._grip_guards) < 2 or any(not g.ee_idxs for g in self._grip_guards):
            return torch.full((n,), -1, dtype=torch.long, device=dev)
        d = _obj(self.env, obj).data
        bodies = getattr(d, "body_pos_w", None)  # every link of a multi-link part, not just its root
        pts = (bodies if bodies is not None and bodies.shape[1] > 1 else d.root_pos_w[:, None]).to(dev)
        per = []
        for g in self._grip_guards:  # per env: this gripper's nearest hand to the part's nearest link
            hands = g.data.body_pos_w[:, g.ee_idxs].to(dev)  # (n, Hg, 3)
            per.append((hands[:, :, None] - pts[:, None]).norm(dim=-1).flatten(1).amin(dim=1))
        return torch.stack(per, dim=1).argmin(dim=1)

    # -- subclass hooks -------------------------------------------------------------------------
    def supported(self, obj: str) -> torch.Tensor:
        """(n,) bool: `obj` rests where it may be let go."""
        raise NotImplementedError

    def _lift_ok(self, obj: str) -> torch.Tensor | None:
        """Extra per-env condition for a lift test to pass (None = no extra condition)."""
        return None

    # -- grip -----------------------------------------------------------------------------------
    #: rotational slip tolerance for `hold()` monitors (deg); None = position slip only
    hold_slip_deg: float | None = None

    def hold(self, obj: str, slip_tol: float = 0.005) -> HoldMonitor:
        """A slip monitor on `obj` that tightens the grip on slip."""
        if obj not in self._holds:
            self._holds[obj] = HoldMonitor(self.env, obj, slip_tol=slip_tol, grip=self.grip,
                                           slip_deg=self.hold_slip_deg)
        return self._holds[obj]

    def lift_test(self, obj: str, step_fn: Callable[[], None], steps: int = 20,
                  min_rise: float = 0.003) -> bool:
        """Lift a few millimetres and check the part rose with the hand, without slipping.
        `step_fn` runs one env.step of a small upward move; it is called `steps` times. Returns True
        only if every env passed — anything else: put it down and re-grasp, do not carry."""
        mon = self.hold(obj)
        mon.lock()
        z0 = mon.part_pos()[:, 2].clone()  # the grasped link: a multi-link part may flex elsewhere
        for _ in range(steps):
            step_fn()
        ok = ((mon.part_pos()[:, 2] - z0) >= min_rise) & ~mon.slipping
        extra = self._lift_ok(obj)
        if extra is not None:
            ok &= extra.to(ok.device)
        if bool(ok.all()):
            self.mark_held(obj)
            return True
        mon.release()
        return False

    def mark_held(self, obj: str) -> None:
        """Treat `obj` as held from now on (gate + carry speed). `lift_test` calls this; call it
        directly for a re-grasp on a part that cannot be lifted (e.g. one already threaded in)."""
        self.hold(obj).lock()
        for e in range(self.env.num_envs):
            self._held_obj[e] = obj
        self._held_arm = self._nearest_gripper(obj)
        for g, q in zip(self._grip_guards, self._q_grasp):
            q.copy_(g.data.joint_pos)  # the closure on the part: "opened" is measured from here

    # -- release gate ---------------------------------------------------------------------------
    def release_allowed(self) -> torch.Tensor:
        """(n,) bool per env: nothing held, or the held part is supported."""
        import torch

        ok = torch.ones_like(self.gate_closed)
        for obj in {o for o in self._held_obj if o is not None}:  # one check per distinct part
            mine = torch.tensor([o == obj for o in self._held_obj], device=ok.device)
            ok = torch.where(mine, self.supported(obj).to(ok.device), ok)
        return ok

    def released(self, obj: str, envs=None) -> None:
        """Stop treating `obj` as carried (in `envs`, default all): no gate, no carry speed limit.
        Happens automatically once the fingers open on a supported part; call it to force it."""
        envs = range(self.env.num_envs) if envs is None else envs
        for e in envs:
            if self._held_obj[e] == obj:
                self._held_obj[e] = None
        if obj in self._holds and obj not in self._held_obj:
            self._holds[obj].release()

    def _release_gate(self) -> None:
        """Fingers may close any time but may not OPEN while a held part is unsupported: an opening
        command is replaced by the closure the fingers had when the gate engaged. Once the fingers
        do open (more than `open_margin` past the grasp closure, which the gate only allows on a
        supported part) the part counts as put down. Only the holding hand is gated and watched."""
        import torch

        if not self.gate_fingers:
            return
        blocked = ~self.release_allowed()
        newly = blocked & ~self.gate_closed
        opened = torch.zeros_like(blocked)
        for k, (g, q_gate, q_grasp) in enumerate(zip(self._grip_guards, self._gate_q, self._q_grasp)):
            d = g.data
            sign = self._sign[k]
            holder = (self._held_arm == k) | (self._held_arm < 0)  # (n,) this hand holds the part
            q_gate[newly] = d.joint_pos[newly]
            tgt = d.joint_pos_target
            # remember the latest SQUEEZING command (target at or inside the fingers' position)
            squeezing = (tgt - d.joint_pos) * sign >= 0
            self._hold_tgt[k] = torch.where(squeezing, tgt, self._hold_tgt[k])
            opening = (tgt - q_gate) * sign < 0  # target on the open side of the grasp closure
            # hold the last squeezing command, so the fingers keep pressing (holding the current
            # position would leave a position-controlled finger pushing with ~zero force)
            keep = torch.where((self._hold_tgt[k] - q_gate) * sign >= 0, self._hold_tgt[k], q_gate)
            hold = (blocked & holder)[:, None] & g.grip & opening
            d.joint_pos_target.copy_(torch.where(hold, keep, tgt))
            wide = (d.joint_pos - q_grasp) * sign < -self.open_margin
            opened |= (wide & g.grip).any(dim=1) & holder
        self.gate_closed = blocked
        held = torch.tensor([o is not None for o in self._held_obj], device=blocked.device)
        done = held & ~blocked & opened
        if bool(done.any()):
            for e in done.nonzero().flatten().tolist():
                self.released(self._held_obj[e], envs=[e])

    # -- carry ----------------------------------------------------------------------------------
    def carrying(self) -> torch.Tensor:
        import torch

        return torch.tensor([o is not None for o in self._held_obj], device=self.gate_closed.device)

    def _carry_limit(self, speed: float):
        def limit():
            import torch

            c = self.carrying()
            return torch.where(c, torch.full_like(c, speed, dtype=torch.float32),
                               torch.full_like(c, INF, dtype=torch.float32))
        return limit

    def carry_height_ok(self, obj: str) -> torch.Tensor:
        """(n,) bool: the carried part is within `carry_height` of the table top — a drop from
        here lands at ≤ sqrt(2 g h) (1.4 m/s for 10 cm). Plan the carry path under it."""
        return _obj(self.env, obj).data.root_pos_w[:, 2] - self._surface_z() <= self.carry_height

    def _surface_z(self):
        origins = getattr(self.env.iscene, "env_origins", None)
        z = float(self.scene.cfg.surface_z)
        return z if origins is None else origins[:, 2] + z
