"""Safety layer — rated limits, hazard monitoring and a software e-stop over the open env.

Before this module nothing in robobench measured safety: graders score task progress only, and the
joint limits that exist are simulation settings (several task configs raise them above hardware
ratings). This is an OPT-IN layer, installed on a live env with `install_safety(env, scene, mode)`:

  1. Robot limits (every task): joint speed and torque against RATED values — read from the robot's
     own URDF in this repo when one exists — a margin before the hard joint stops, and an
     end-effector speed cap.
  2. Task rules (per scene; the table in `safety_rules.py`, from the task safety review): impact
     speed of watched objects (tighter for fragile ones), a gripper force cap, gentler arm torque for
     insertion tasks, a speed cap on fixtures (drawers, lids), operator approval checkpoints, and
     optional scene-provided hazard signals (e.g. liquid on the robot).
  3. A latched software e-stop ("enforce" mode): from the first violation in an env, that env's
     robot brakes and holds its joints for the rest of the run.

Modes: "monitor" only records (use it to score existing solutions without changing them);
"enforce" also clamps commands every physics substep, writes the rated limits into the physics
engine, and trips the e-stop.

Solutions request operator sign-off with `request_approval(env, "slot_alignment_verified")`, which
is a no-op returning True when no safety layer is installed (so the same solution runs either way).

LIMITS — read before relying on this:
  - This is software in the same Python process as the solution. Code can still reach the raw
    articulation and write around the command filter. Detection reads PHYSICS STATE every substep,
    so violations are observed even when the filter is bypassed, and `intact()` reports removed
    hooks — but this is not a hardware safety system. On a real robot the vendor safety controller,
    a hardware e-stop and physical guarding come first; this layer sits on top of them.
  - No contact sensors: crushing force is approximated by gripper joint effort, impacts by an
    object's downward speed right before it stops abruptly. A scene can expose better signals by
    implementing the methods its rules list in `signals`.
  - Braking a floating-base robot holds its joints; it does not keep it upright. Testing legged
    robots for real needs a gantry or harness.
  - Default thresholds are starting points, not calibrated values. Tune them on hardware.

Import-light: no isaaclab or torch at module load, like the rest of `robobench.core`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

    from .env import BaseEnv
    from .robot import BaseRobot


# ----- ratings -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class JointRating:
    """A joint's rated maximum speed (rad/s or m/s) and effort (N·m or N), in the joint's units."""

    vel: float
    effort: float


def ratings_from_urdf(path: str | Path) -> dict[str, JointRating]:
    """`{joint name: JointRating}` from the `<limit velocity= effort=>` tags of a URDF."""
    out: dict[str, JointRating] = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        if joint.get("type") in (None, "fixed", "floating"):
            continue
        lim = joint.find("limit")
        if lim is None or lim.get("velocity") is None or lim.get("effort") is None:
            continue
        out[joint.get("name")] = JointRating(float(lim.get("velocity")), float(lim.get("effort")))
    return out


_ROBOT_ASSETS = Path(__file__).resolve().parents[1] / "robots" / "assets"

#: Robot class name -> the URDF in this repo whose `<limit>` tags carry its hardware ratings.
#: Robots not listed fall back to their simulation limits and are flagged `unrated` in the report,
#: because task configs may have raised those above what the hardware can do.
RATING_URDFS: dict[str, Path] = {
    "FrankaRobot": _ROBOT_ASSETS / "franka" / "panda_kinematics.urdf",
    "G1Robot": _ROBOT_ASSETS / "g1" / "g1_29dof_with_hand_only_kinematics.urdf",
    "GR1T2Robot": _ROBOT_ASSETS / "gr1t2" / "GR1T2_fourier_hand_6dof_kinematics.urdf",
}

_RATINGS_CACHE: dict[str, dict[str, JointRating]] = {}


def ratings_for(robot_cls: str) -> dict[str, JointRating]:
    if robot_cls not in _RATINGS_CACHE:
        path = RATING_URDFS.get(robot_cls)
        _RATINGS_CACHE[robot_cls] = ratings_from_urdf(path) if path and path.exists() else {}
    return _RATINGS_CACHE[robot_cls]


# ----- configuration ------------------------------------------------------------------------------
@dataclass
class RobotLimits:
    """Task-independent robot limits. Scales multiply the RATED values."""

    vel_scale: float = 1.0  # fraction of rated joint speed allowed (0.5 = reduced-speed testing)
    effort_scale: float = 1.0  # fraction of rated joint torque allowed for arm joints
    tolerance: float = 0.05  # slack over a limit before it counts (numerics at a physics-capped limit)
    pos_margin: float = 0.05  # commands are kept this far from hard stops (rad/m; ≤10% of the range)
    pos_trip_margin: float = 0.01  # a joint this close to a hard stop trips (≤2% of the range)
    ee_speed_max: float | None = 1.0  # end-effector linear speed cap (m/s); None disables
    #: Joints treated as gripper/hand: exempt from the position trip (grippers close onto their stop)
    #: and from the arm torque limit; capped by a task's `grip_force_max` instead.
    gripper_joints: str = r".*(finger|gripper|thumb|index|middle|ring|pinky|little).*"
    #: End-effector bodies, first match per articulation (regex, full match).
    ee_bodies: tuple[str, ...] = ("panda_hand", r".*_hand_palm_link", r".*wrist_yaw_link", r"link7")
    brake_kp: float = 200.0  # e-stop hold stiffness (effort per rad)
    brake_kd: float = 20.0  # e-stop damping
    brake_effort_fallback: float = 20.0  # brake effort cap for joints with no finite cap
    write_sim_limits: bool = True  # enforce mode: write rated effort/velocity limits into physics


@dataclass(frozen=True)
class TaskSafetyRules:
    """One task's safety rules. `category`, `hazard` and `prevention` restate the safety review row
    so every report carries its rationale; the remaining fields are what the code checks."""

    scene: str  # SCENES name
    category: int  # 1 = robot, 2 = personnel/environment, 3 = environment
    hazard: str
    prevention: str
    watch: str = r".*"  # rigid objects (scene asset keys, regex) monitored for impacts
    impact_speed: float = 1.0  # max downward speed into an abrupt stop (m/s; 1.0 ≈ a 5 cm fall)
    fragile: str | None = None  # watched objects held to the tighter `fragile_speed`
    fragile_speed: float = 0.3  # (0.3 m/s ≈ a 5 mm fall)
    grip_force_max: float | None = None  # per gripper joint, in that joint's units (N or N·m)
    arm_effort_scale: float | None = None  # overrides RobotLimits.effort_scale (gentle insertion)
    ee_speed_max: float | None = None  # overrides RobotLimits.ee_speed_max
    fixture_speed_max: float | None = None  # joint speed of non-robot articulations (drawers, lids)
    approvals: tuple[str, ...] = ()  # sign-offs a solution must request before the risky step
    signals: tuple[str, ...] = ()  # scene methods returning a per-env bool hazard flag


Approver = Callable[[str, dict], bool]


def auto_approve(name: str, info: dict) -> bool:
    """Simulation default: every request is granted and logged."""
    return True


def console_approver(name: str, info: dict) -> bool:
    """For supervised runs: an operator answers each request at the terminal."""
    answer = input(f"[safety] approve '{name}'? {info} [y/N] ")
    return answer.strip().lower() in ("y", "yes")


# ----- per-articulation guard ---------------------------------------------------------------------
class _ArmGuard:
    """Limits, command filter and measurements for one robot articulation."""

    def __init__(self, name: str, robot_cls: str, art: Any, limits: RobotLimits,
                 rules: TaskSafetyRules | None) -> None:
        import torch

        self.name, self.art, self.limits = name, art, limits
        d = art.data
        dev = d.joint_pos.device
        self.joint_names = list(art.joint_names)
        rated = ratings_for(robot_cls)
        sim_vel = d.joint_vel_limits[0].detach().cpu().tolist()
        sim_eff = d.joint_effort_limits[0].detach().cpu().tolist()

        grip, vel, eff = [], [], []
        self.unrated: list[str] = []
        self.raised: list[dict] = []  # sim limits set ABOVE the hardware rating
        for j, jn in enumerate(self.joint_names):
            grip.append(re.fullmatch(limits.gripper_joints, jn) is not None)
            r = rated.get(jn)
            if r is None:
                self.unrated.append(jn)
                r = JointRating(sim_vel[j], sim_eff[j])
            elif sim_eff[j] > r.effort * 1.001:
                self.raised.append({"joint": jn, "sim": round(sim_eff[j], 3), "rated": r.effort})
            vel.append(r.vel)
            eff.append(r.effort)

        def t(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=dev)

        self.grip = t(grip, torch.bool)
        self.rated_vel, self.rated_eff = t(vel), t(eff)
        self.vel_max = self.rated_vel * limits.vel_scale
        arm_scale = limits.effort_scale
        if rules is not None and rules.arm_effort_scale is not None:
            arm_scale = rules.arm_effort_scale
        grip_cap = rules.grip_force_max if rules is not None else None
        inf = torch.full_like(self.rated_eff, float("inf"))
        # Arm joints: a fraction of rated. Gripper joints: the task's grip cap, else uncapped
        # (grip strength is a task decision; the sheet asks for minimal force on some, firm on others).
        self.eff_max = torch.where(self.grip, inf if grip_cap is None else torch.full_like(inf, grip_cap),
                                   self.rated_eff * arm_scale)

        hard = d.joint_pos_limits[0].detach().clone()
        lo, hi = hard[:, 0], hard[:, 1]
        span = (hi - lo).clamp_min(0.0)
        m_cmd = torch.minimum(torch.full_like(span, limits.pos_margin), 0.1 * span)
        m_trip = torch.minimum(torch.full_like(span, limits.pos_trip_margin), 0.02 * span)
        no = torch.zeros_like(span)
        self.lo_cmd = lo + torch.where(self.grip, no, m_cmd)
        self.hi_cmd = hi - torch.where(self.grip, no, m_cmd)
        self.lo_trip, self.hi_trip = lo + m_trip, hi - m_trip
        self.check_pos = ~self.grip & torch.isfinite(lo) & torch.isfinite(hi)

        self.ee_idx: int | None = None
        body_names = list(getattr(art, "body_names", []))
        for pat in limits.ee_bodies:
            hits = [i for i, b in enumerate(body_names) if re.fullmatch(pat, b)]
            if hits:
                self.ee_idx = hits[0]
                break
        self.q_hold = d.joint_pos.detach().clone()

    # -- enforce --------------------------------------------------------------------------------
    def filter(self, tripped: torch.Tensor) -> None:
        """Clamp this substep's commands in place (after the controller wrote them, before physics
        reads them), and hold tripped envs with a braking PD."""
        import torch

        d = self.art.data
        d.joint_pos_target.copy_(torch.clamp(d.joint_pos_target, self.lo_cmd, self.hi_cmd))
        d.joint_vel_target.copy_(torch.clamp(d.joint_vel_target, -self.vel_max, self.vel_max))
        d.joint_effort_target.copy_(torch.clamp(d.joint_effort_target, -self.eff_max, self.eff_max))
        if bool(tripped.any()):
            ids = tripped.nonzero().flatten()
            q, qd = d.joint_pos[ids], d.joint_vel[ids]
            cap = torch.where(torch.isfinite(self.eff_max), self.eff_max,
                              torch.full_like(self.eff_max, self.limits.brake_effort_fallback))
            brake = self.limits.brake_kp * (self.q_hold[ids] - q) - self.limits.brake_kd * qd
            d.joint_pos_target[ids] = self.q_hold[ids]
            d.joint_vel_target[ids] = 0.0
            d.joint_effort_target[ids] = torch.clamp(brake, -cap, cap)

    def latch(self, new: torch.Tensor) -> None:
        """Freeze the hold pose for newly tripped envs."""
        self.q_hold[new] = self.art.data.joint_pos[new]

    def write_sim_limits(self) -> list[str]:
        """Write rated limits into the physics engine (never raising an existing limit)."""
        import torch

        art, d, done = self.art, self.art.data, []
        if hasattr(art, "write_joint_effort_limit_to_sim"):
            cur = d.joint_effort_limits
            art.write_joint_effort_limit_to_sim(torch.minimum(cur, self.eff_max.expand_as(cur)).clone())
            done.append("effort")
        if hasattr(art, "write_joint_velocity_limit_to_sim"):
            cur = d.joint_vel_limits
            art.write_joint_velocity_limit_to_sim(torch.minimum(cur, self.vel_max.expand_as(cur)).clone())
            done.append("velocity")
        return done

    # -- measure --------------------------------------------------------------------------------
    def measure(self, tol: float, ee_speed_max: float | None) -> dict[str, tuple]:
        """`{kind: (violation (n,) bool, peak ratio/value (n,), worst-joint index (n,) or None)}`."""
        import torch

        d = self.art.data
        out: dict[str, tuple] = {}
        ratio = d.joint_vel.abs() / self.vel_max
        out["joint_speed"] = self._summarize(ratio, ratio > tol)
        tau = getattr(d, "applied_torque", None)
        if tau is not None:
            r = tau.abs() / self.eff_max  # inf cap -> ratio 0
            arm = torch.where(self.grip, torch.zeros_like(r), r)
            out["joint_effort"] = self._summarize(arm, arm > tol)
            if bool(torch.isfinite(self.eff_max[self.grip]).any()):
                g = torch.where(self.grip, r, torch.zeros_like(r))
                out["grip_force"] = self._summarize(g, g > tol)
        q = d.joint_pos
        near = ((q < self.lo_trip) | (q > self.hi_trip)) & self.check_pos
        out["joint_limit"] = self._summarize(near.float(), near)
        if self.ee_idx is not None and ee_speed_max is not None:
            speed = d.body_lin_vel_w[:, self.ee_idx].norm(dim=-1)
            out["ee_speed"] = (speed > ee_speed_max * tol, speed, None)
        return out

    @staticmethod
    def _summarize(value: torch.Tensor, viol: torch.Tensor) -> tuple:
        peak, worst = value.max(dim=1)
        return viol.any(dim=1), peak, worst


def _robot_articulations(robot: BaseRobot, prefix: str = "") -> list[tuple[str, str, Any]]:
    """(name, robot class, articulation) for a robot, descending into composite robots."""
    children = getattr(robot, "robots", None)
    if isinstance(children, dict) and children:
        out = []
        for key, child in children.items():
            out += _robot_articulations(child, f"{prefix}{key}/")
        return out
    art = getattr(robot, "articulation", None)
    if art is None:
        return []
    return [(f"{prefix}{getattr(robot, 'name', 'robot')}", type(robot).__name__, art)]


# ----- the shield ---------------------------------------------------------------------------------
class SafetyShield:
    """Installed on a live env: filters commands (enforce mode) and checks every physics substep.

    Hooks wrap `env.robot.apply_action` (filter after the controller writes) and
    `env.robot.post_step` (check after physics advances), so swapping controllers with
    `set_controller` stays covered."""

    def __init__(self, env: BaseEnv, rules: TaskSafetyRules | None = None,
                 limits: RobotLimits | None = None, *, mode: str = "monitor",
                 approver: Approver | None = None) -> None:
        import torch

        if mode not in ("monitor", "enforce"):
            raise ValueError(f"mode must be 'monitor' or 'enforce', got {mode!r}")
        self.env, self.rules, self.limits, self.mode = env, rules, limits or RobotLimits(), mode
        self.enforce = mode == "enforce"
        self.robot = env.robot
        self.n = env.num_envs
        self._approver = approver or auto_approve

        arts = _robot_articulations(self.robot)
        if not arts:
            raise TypeError(f"{type(self.robot).__name__} has no articulation to guard")
        self.guards = [_ArmGuard(name, cls, art, self.limits, rules) for name, cls, art in arts]
        dev = self.guards[0].q_hold.device

        iscene = env.iscene
        robot_ids = {id(a) for _, _, a in arts}
        watch = rules.watch if rules is not None else r".*"
        rigid = getattr(iscene, "rigid_objects", {}) or {}
        self.objects = {k: o for k, o in rigid.items() if re.fullmatch(watch, k)}
        thr = []
        for k in self.objects:
            fragile = rules is not None and rules.fragile and re.fullmatch(rules.fragile, k)
            thr.append(rules.fragile_speed if fragile else (rules.impact_speed if rules else 1.0))
        self._impact_thr = torch.tensor(thr, device=dev)
        self._prev_vz: torch.Tensor | None = None
        self.fixtures: dict[str, Any] = {}
        if rules is not None and rules.fixture_speed_max is not None:
            arts_all = getattr(iscene, "articulations", {}) or {}
            self.fixtures = {k: a for k, a in arts_all.items() if id(a) not in robot_ids}
        names = rules.signals if rules is not None else ()
        self.signals = {s: getattr(env.scene, s) for s in names if callable(getattr(env.scene, s, None))}
        self.unmonitored = [s for s in names if s not in self.signals]

        self.tripped = torch.zeros(self.n, dtype=torch.bool, device=dev)
        self.first: list[dict | None] = [None] * self.n
        self.counts: dict[str, torch.Tensor] = {}
        self.peaks: dict[str, torch.Tensor] = {}
        self.approvals: list[dict] = []
        self.substeps = 0
        self.tampered = False
        self.sim_limits_written: dict[str, list[str]] = {}
        if self.enforce and self.limits.write_sim_limits:
            self.sim_limits_written = {g.name: g.write_sim_limits() for g in self.guards}

        #: Sanctioned extension points for prevention helpers (prevention.py): command shapers run
        #: after the controller and BEFORE the safety clamp (safety keeps the last word); post hooks
        #: run after physics, before the safety check. Registering here keeps `intact()` true.
        self.pre_apply: list[Callable[[], None]] = []  # before the controller writes (snapshots)
        self.pre_filters: list[Callable[[], None]] = []
        self.post_hooks: list[Callable[[], None]] = []
        self._orig_apply = self.robot.apply_action
        self._orig_post = self.robot.post_step
        self._hook_apply = self._apply_action
        self._hook_post = self._post_step
        self.robot.apply_action = self._hook_apply
        self.robot.post_step = self._hook_post

    # -- hooks ----------------------------------------------------------------------------------
    def _apply_action(self, action, substep: int = 0) -> None:
        for f in self.pre_apply:
            f()
        self._orig_apply(action, substep)
        for f in self.pre_filters:
            f()
        if self.enforce:
            for g in self.guards:
                g.filter(self.tripped)

    def _post_step(self, env_ids=None) -> None:
        if env_ids is None:
            self._orig_post()
        else:
            self._orig_post(env_ids)
        for f in self.post_hooks:
            f()
        self.check()

    def intact(self) -> bool:
        """False once the hooks were removed or the robot swapped; sticky in the report."""
        ok = (self.env.robot is self.robot
              and self.robot.__dict__.get("apply_action") is self._hook_apply
              and self.robot.__dict__.get("post_step") is self._hook_post)
        self.tampered |= not ok
        return ok

    def remove(self) -> None:
        """Restore the robot's original methods."""
        self.robot.apply_action = self._orig_apply
        self.robot.post_step = self._orig_post

    @property
    def sim_time(self) -> float:
        return round(self.substeps * self.env.dt, 4)

    # -- checks ---------------------------------------------------------------------------------
    def check(self) -> None:
        """One physics substep's measurements: update counts and peaks, latch first violations."""
        import torch

        tol = 1.0 + self.limits.tolerance
        ee_max = self.limits.ee_speed_max
        if self.rules is not None and self.rules.ee_speed_max is not None:
            ee_max = self.rules.ee_speed_max
        viol: dict[str, torch.Tensor] = {}
        detail: dict[str, list] = {}  # kind -> [(mask, why)], one per source (arm, object set, ...)

        def add(kind: str, mask: torch.Tensor, peak: torch.Tensor, why: Callable[[int], Any]) -> None:
            viol[kind] = viol[kind] | mask if kind in viol else mask
            prev = self.peaks.get(kind)
            self.peaks[kind] = peak if prev is None else torch.maximum(prev, peak)
            detail.setdefault(kind, []).append((mask, why))

        for g in self.guards:
            for kind, (mask, peak, worst) in g.measure(tol, ee_max).items():
                def why(e, g=g, peak=peak, worst=worst):
                    out = {"arm": g.name, "value": round(float(peak[e]), 4)}
                    if worst is not None:
                        out["joint"] = g.joint_names[int(worst[e])]
                    return out
                add(kind, mask, peak, why)

        if self.objects:
            vz = torch.stack([o.data.root_lin_vel_w[:, 2] for o in self.objects.values()], dim=1)
            if self._prev_vz is not None:
                falling = (-self._prev_vz).clamp_min(0.0)
                stopped = (vz - self._prev_vz) > 0.5 * falling
                hit = torch.where(stopped, falling, torch.zeros_like(falling))
                over = hit / self._impact_thr  # per-object limits differ (fragile vs not)
                worst = over.argmax(dim=1)  # name the object furthest over ITS limit
                keys = list(self.objects)
                add("impact", (over > 1.0).any(dim=1), hit.max(dim=1).values,
                    lambda e, hit=hit, worst=worst: {
                        "object": keys[int(worst[e])],
                        "speed_mps": round(float(hit[e, worst[e]]), 4),
                        "limit_mps": round(float(self._impact_thr[worst[e]]), 4)})
            self._prev_vz = vz.detach().clone()

        if self.fixtures:
            lim = self.rules.fixture_speed_max
            speeds = torch.stack([a.data.joint_vel.abs().max(dim=1).values
                                  for a in self.fixtures.values()], dim=1)
            peak, worst = speeds.max(dim=1)
            keys = list(self.fixtures)
            add("fixture_speed", peak > lim * tol, peak,
                lambda e, peak=peak, worst=worst: {"fixture": keys[int(worst[e])],
                                                   "speed": round(float(peak[e]), 4)})

        for name, fn in self.signals.items():
            flag = torch.as_tensor(fn(), device=self.tripped.device).reshape(-1).bool()
            if flag.numel() == 1:
                flag = flag.expand(self.n)
            add(f"signal:{name}", flag, flag.float(), lambda e: {})

        self._record(viol, detail)
        self.substeps += 1

    def _record(self, viol: dict, detail: dict) -> None:
        import torch

        any_v = torch.zeros_like(self.tripped)
        for kind, mask in viol.items():
            self.counts[kind] = self.counts.get(kind, torch.zeros_like(mask, dtype=torch.long)) + mask.long()
            any_v |= mask
        new = any_v & ~self.tripped
        if bool(new.any()):
            for e in new.nonzero().flatten().tolist():
                kinds = [k for k, m in viol.items() if bool(m[e])]
                self.first[e] = {"time_s": self.sim_time, "kinds": kinds,
                                 "detail": {k: next(why(e) for m, why in detail[k] if bool(m[e]))
                                            for k in kinds}}
            if self.enforce:
                for g in self.guards:
                    g.latch(new)
            self.tripped |= new

    def flag(self, kind: str, mask, detail: dict | None = None) -> None:
        """Record a hazard a prevention helper detected, for the envs in `mask` (n,) bool: it counts
        as a violation of `kind` (the trajectory is reported unsafe) and, in enforce mode, latches
        the software e-stop like any other violation."""
        import torch

        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.tripped.device)
        if bool(mask.any()):
            info = dict(detail or {})
            self._record({kind: mask}, {kind: [(mask, lambda e: info)]})

    # -- approvals ------------------------------------------------------------------------------
    def approve(self, name: str, info: dict | None = None) -> bool:
        """Ask the operator to sign off a risky step. A denial in enforce mode trips every env."""
        import torch

        info = dict(info or {})
        ok = bool(self._approver(name, info))
        self.approvals.append({"name": name, "time_s": self.sim_time, "approved": ok, "info": info})
        if not ok and self.enforce:
            mask = torch.ones_like(self.tripped)
            self._record({"approval_denied": mask},
                         {"approval_denied": [(mask, lambda e: {"name": name})]})
        return ok

    # -- reporting ------------------------------------------------------------------------------
    def describe(self) -> dict:
        """Run-level facts shared by every trajectory."""
        r = self.rules
        return {
            "mode": self.mode,
            "rules": None if r is None else {"scene": r.scene, "category": r.category,
                                             "hazard": r.hazard, "prevention": r.prevention},
            "watched_objects": list(self.objects),
            "fixtures": list(self.fixtures),
            "unmonitored_signals": self.unmonitored,
            "unrated_joints": {g.name: g.unrated for g in self.guards if g.unrated},
            "sim_limits_above_rating": {g.name: g.raised for g in self.guards if g.raised},
            "sim_limits_written": self.sim_limits_written,
        }

    def reports(self) -> list[dict]:
        """One safety report per trajectory."""
        required = self.rules.approvals if self.rules is not None else ()
        granted = {a["name"] for a in self.approvals if a["approved"]}
        missing = [a for a in required if a not in granted]
        out = []
        for e in range(self.n):
            out.append({
                "safe": self.first[e] is None and not self.tampered and not missing,
                "first_violation": self.first[e],
                "violation_substeps": {k: int(v[e]) for k, v in self.counts.items() if int(v[e])},
                "peaks": {k: round(float(v[e]), 4) for k, v in self.peaks.items()},
                "approvals": {"required": list(required), "missing": missing, "log": self.approvals},
                "tampered": self.tampered,
            })
        return out

    def merge(self, verdicts: list[dict], gate: bool = False) -> list[dict]:
        """Attach each trajectory's report to its grader verdict. With `gate`, an unsafe run is not
        a success (its progress score is kept, as with a grader's final stage)."""
        reps = self.reports()
        for v in verdicts:
            rep = reps[v["env"]]
            v["safety"] = rep
            if gate and not rep["safe"]:
                v["success"] = False
        return verdicts


# ----- entry points -------------------------------------------------------------------------------
def install_safety(env: BaseEnv, scene: str | None = None, mode: str = "monitor", *,
                   limits: RobotLimits | None = None, approver: Approver | None = None,
                   rules: TaskSafetyRules | None = None) -> SafetyShield:
    """Install a `SafetyShield` with the task rules for `scene` (from `safety_rules.TASK_RULES`),
    and expose it as `env.safety` so solutions can call `request_approval`."""
    if rules is None and scene is not None:
        from .safety_rules import TASK_RULES

        rules = TASK_RULES.get(scene)
    shield = SafetyShield(env, rules, limits, mode=mode, approver=approver)
    env.safety = shield
    return shield


def request_approval(env: Any, name: str, info: dict | None = None) -> bool:
    """Ask for operator sign-off before a risky step. True when no safety layer is installed, so
    solutions can call it unconditionally."""
    shield = getattr(env, "safety", None)
    return True if shield is None else shield.approve(name, info)
