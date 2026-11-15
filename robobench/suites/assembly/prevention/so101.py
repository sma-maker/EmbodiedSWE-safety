"""SO101 — preventing a dropped robot-arm part.

Safety review row:
    What could go wrong: dropping the robot arm may dent or damage the arm.
    How to prevent:      firmly hold the robot arm.
    Category:            1 — Robot.

What gets carried in this task (scenes/so101_assembly.py) — the parts of the SO101 arm:
  - `motor`: the bare elbow servo (STS3215, 61 g), a free rigid body.
  - `distal`: the forearm-to-gripper half, a free ARTICULATION (several links, three servos). Long
    and off-centre: held at one end it wants to swing in the fingers, and its own wrist joints
    flex a little when carried.
  - `proximal`: the base-shoulder-upper-arm half, a free articulation. Usually dragged or held in
    place; after assembly the WHOLE robot (proximal + seated servo + attached forearm) may be
    lifted, and then all of it hangs from the grip.
No part is welded to the hand: every grasp is friction alone. What holds a part once it is put
where it belongs is the scene's own mechanics: the servo CLICKS into its pocket (a detent) or is
screwed in; the forearm SNAPS onto the horn or is screwed on. Near is not enough: the base rests
tilted (up to ~32°) and the pocket is a slip fit, so an unclicked servo slides out and falls.

"Firmly hold", as code:

  1. Grip for what is actually in the hand: `needed_force(part)` is the pinch force for friction
     to carry the part's weight (read from the simulator, plus everything fastened to it) under
     carry acceleration, with a safety factor. Grasping sets the grip to it.
  2. Refuse a hold that cannot be firm: `lift_test` will not lift a part that needs more than
     `grip_force_max` (the assembled robot can) — `refusal` says why. With `hands=2` the load is
     shared by both hands (four fingers) and BOTH hands are kept closed until it is put down.
  3. Confirm the hold before carrying: `lift_test(part, ...)` raises it a few mm and checks the
     link in the fingers rose with the hand, without sliding or turning.
  4. Hold firmly while carrying: the grip tightens a step at a time if the grasped link slides
     (`slip_tol`) or pivots (`slip_deg`) in the fingers, up to `grip_force_max`.
  5. Never let go in the air: the fingers stay closed until the part rests on the bench, or the
     scene says it is held where it belongs (servo clicked in or screwed; forearm snapped or
     screwed on — AND that servo in its pocket: the forearm snaps onto the servo, not the base,
     so on a loose servo the pair just tips over).
  6. One carried part at a time: a new grasp is refused while another part is still carried and
     unsupported, so the first part never silently loses its protection.
  7. Carry low and slow: the arm is speed-limited while holding a part, and `carry_height_ok`
     checks the part stays within `carry_height` of the bench, so a drop is short.

Usage in a solution:

    from robobench.suites.assembly.prevention.so101 import SO101Prevention

    def solve(env):
        prev = SO101Prevention(env)
        ...                                        # close the fingers on the servo
        if not prev.lift_test("motor", lambda: env.step(small_lift)):
            print(prev.refusal)                    # why: slipped, too heavy, already carrying
            ...                                    # re-grasp; do not carry
        ...                                        # carry low; push it into the pocket
        ...                                        # the fingers open once it has clicked in
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, _quat_mul, _obj, quat_rotate,
                                       required_grip_force)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["SO101Prevention", "PARTS"]

PARTS = ("motor", "distal", "proximal")
#: The servo's mass is from the repo (smokes/so101_smoke.py: "the 61 g servo"). The two halves'
#: masses live in their USDs and are read from the simulator when it is running; these are only
#: fallbacks (assumptions: printed shells + three servos each, ~0.3 / 0.4 kg).
FALLBACK_MASS = {"motor": 0.061, "distal": 0.30, "proximal": 0.40}
#: Assumption, not from the repo: rubber finger pads on printed PLA. Calibrate on hardware.
PAD_FRICTION = 0.35


class SO101Prevention(CarryPrevention):
    """Installs the so101 prevention on a live env and exposes the per-step checks.

    `safety_factor` scales the friction grip over what the weight needs at `carry_accel`.
    `slip_deg` is how far the grasped link may turn in the fingers before it counts as slipping.
    `rest_tol` is how far above its resting height on the bench a still part may be and still
    count as put down (a drop from there is at most that far)."""

    hold_slip_deg = 5.0

    def __init__(self, env: BaseEnv, grip_force_max: float = 40.0, pad_friction: float = PAD_FRICTION,
                 safety_factor: float = 2.0, carry_accel: float = 2.0, carry_speed: float = 0.25,
                 carry_height: float = 0.10, slip_deg: float = 5.0, rest_tol: float = 0.02,
                 still: float = 0.02, still_ang: float = 0.2, close_sign: float | None = None,
                 open_margin: float = 0.003, grip_arms=None) -> None:
        self.hold_slip_deg = slip_deg
        self.pad_friction, self.safety_factor, self.carry_accel = pad_friction, safety_factor, carry_accel
        self.rest_tol, self.still, self.still_ang = rest_tol, still, still_ang
        self.refusal: str | None = None  # why the last lift_test / mark_held was refused
        self._hands_on = 1  # hands on the part for the lift test in progress
        self.env, self.scene = env, env.scene  # needed_force() reads them before super().__init__
        super().__init__(env, self.needed_force("motor"), grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms)

    # -- 1. grip for what is in the hand --------------------------------------------------------
    def mass(self, part: str) -> float:
        """The part's own mass (kg): all its links' masses from the simulator, else the fallback."""
        view = getattr(_obj(self.env, part), "root_physx_view", None)
        if view is not None:
            try:
                return float(view.get_masses()[0].sum())
            except Exception:  # noqa: BLE001 - no physics view yet (e.g. before play): fallback
                pass
        return FALLBACK_MASS[part]

    def load_mass(self, part: str) -> float:
        """What hangs from the grip when holding `part` (kg): the part plus whatever is held to it
        in any env (planned for the worst env) — the base carries a fastened servo and forearm,
        the servo a forearm on its horn."""
        m = self.mass(part)
        if part == "proximal" and bool(self.motor_in_pocket().any()):
            m += self.mass("motor")
            if bool(self.distal_on_horn().any()):
                m += self.mass("distal")
        elif part == "motor" and bool(self.distal_on_horn().any()):
            m += self.mass("distal")
        return m

    def needed_force(self, part: str, hands: int = 1) -> float:
        """Pinch force per finger for friction to carry `part` and what is held to it, the load
        shared by `hands` parallel grippers (two fingers each)."""
        return required_grip_force(self.load_mass(part), self.pad_friction, accel=self.carry_accel,
                                   safety_factor=self.safety_factor, fingers=2 * hands)

    def grip_report(self) -> dict:
        """Per part: the load, the force it needs, and whether that is over the cap."""
        out = {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max, "parts": {}}
        for p in PARTS:
            need = self.needed_force(p)
            two = self.needed_force(p, hands=2)
            out["parts"][p] = {"load_kg": round(self.load_mass(p), 3), "needed_N": round(need, 2),
                               "over_cap": need > self.grip.force_max,
                               "needed_N_two_hands": round(two, 2),
                               "over_cap_two_hands": two > self.grip.force_max}
        return out

    # -- 2 + 6. refuse a hold that cannot be firm, or that would drop another part's protection --
    def _refuse(self, obj: str, hands: int = 1) -> str | None:
        if hands not in (1, 2):
            return f"hands must be 1 or 2, got {hands}"
        if hands == 2 and len(self._grip_guards) < 2:
            return "hands=2 needs a robot with two grippers"
        need = self.needed_force(obj, hands)
        if need > self.grip.force_max:
            hint = ("use both hands (hands=2)" if hands == 1 and len(self._grip_guards) > 1
                    and self.needed_force(obj, 2) <= self.grip.force_max else "a different grasp")
            return (f"{obj} needs {need:.1f} N per finger ({self.load_mass(obj):.3f} kg held, "
                    f"{hands} hand{'s' if hands > 1 else ''}), over the {self.grip.force_max:.1f} N "
                    f"cap: {hint}")
        others = {o for o in self._held_obj if o is not None and o != obj}
        for o in others:
            mine = [h == o for h in self._held_obj]
            sup = self.supported(o)
            if any(m and not bool(sup[e]) for e, m in enumerate(mine)):
                return f"still carrying {o} and it is not put down: put it down first"
        return None

    def _set_grip(self, obj: str, hands: int = 1) -> None:
        """The grip for `obj`. A re-grasp of the part already held keeps a force that slip has
        boosted; a different part starts from its own need (not the last part's force)."""
        need = self.needed_force(obj, hands)
        same = any(o == obj for o in self._held_obj)
        self.grip.force = min(self.grip.force_max, max(self.grip.force, need) if same else need)

    def _two_hand_gate(self, hands: int) -> None:
        """With two hands on the part, keep BOTH closed until it is put down (the shared gate
        otherwise locks only the hand nearest the part)."""
        if hands == 2:
            self._held_arm.fill_(-1)

    # -- 3 + 4. confirm, then hold firmly (slip monitors from CarryPrevention) -------------------
    def lift_test(self, obj: str, step_fn, steps: int = 20, min_rise: float = 0.003,
                  hands: int = 1) -> bool:
        """Refuses (returns False, without lifting) a hold that cannot be firm; otherwise sets the
        grip for `obj` and runs the shared lift test on the link in the fingers. `hands=2`: both
        grippers are on the part (the grip is set for four fingers, both stay closed)."""
        self.refusal = self._refuse(obj, hands)
        if self.refusal is not None:
            return False
        self._set_grip(obj, hands)
        self._hands_on = hands
        ok = super().lift_test(obj, step_fn, steps=steps, min_rise=min_rise)
        self._hands_on = 1
        if not ok:
            self.refusal = f"{obj} did not rise with the hand, or slid or turned in the fingers"
        return ok

    def mark_held(self, obj: str, hands: int | None = None) -> None:
        """Treat `obj` as held (a re-grasp that cannot be lifted). Raises if the hold is refused."""
        hands = self._hands_on if hands is None else hands
        self.refusal = self._refuse(obj, hands)
        if self.refusal is not None:
            raise RuntimeError(self.refusal)
        self._set_grip(obj, hands)
        super().mark_held(obj)
        self._two_hand_gate(hands)
        self._hands_on = 1

    def released(self, obj: str, envs=None) -> None:
        super().released(obj, envs)
        if not bool(self.carrying().any()):
            self.grip.force = self.needed_force("motor")  # the next grasp sets its own force

    # -- 5. where a part may be let go ----------------------------------------------------------
    def _env_flags(self, attr: str):
        import torch

        flags = getattr(self.scene, attr, None)
        if flags is None:
            return None
        return torch.tensor([bool(f) for f in flags], dtype=torch.bool)

    def _fastened(self, horn: bool):
        """(n,) bool from the scene's fastening table: any tab screw (horn=False) or horn screw
        (horn=True) driven home. None when the scene does not expose it."""
        f = getattr(self.scene, "fastened", None)
        if f is None:
            return None
        ne = int(self.scene.cfg.num_elbow_holes)
        return ((f >= ne) if horn else ((f >= 0) & (f < ne))).any(dim=1).cpu()

    def _upper_arm_pose(self):
        prox = _obj(self.env, "proximal")
        b = getattr(self.scene, "b_ua", None)
        if b is None:
            b = prox.find_bodies("upper_arm")[0][0]
        return prox.data.body_pos_w[:, b], prox.data.body_quat_w[:, b]

    @staticmethod
    def _angle_deg(qa, qb):
        import torch

        dot = (qa * qb).sum(-1).abs().clamp(max=1.0)
        return torch.rad2deg(2.0 * torch.arccos(dot))

    def motor_in_pocket(self) -> torch.Tensor:
        """(n,) bool: the pocket HOLDS the servo — clicked in (detent) or screwed. From the scene's
        own state when it exposes it; otherwise from the pose, within the detent's capture
        tolerance (tighter than the screw gate's), since only a clicked-in servo stays put."""
        c = self.scene.cfg
        screwed, clicked = self._fastened(horn=False), self._env_flags("_detent_on")
        if screwed is not None or clicked is not None:
            import torch

            out = torch.zeros(self.env.num_envs, dtype=torch.bool)
            for flags in (screwed, clicked):
                if flags is not None:
                    out |= flags
            return out
        ap, aq = self._upper_arm_pose()
        m = _obj(self.env, "motor").data
        if not getattr(c, "seat_detent", True):  # no click-in: only screws hold it
            return (m.root_pos_w[:, 0] != m.root_pos_w[:, 0])  # all False
        return (((m.root_pos_w - ap).norm(dim=-1) <= getattr(c, "detent_pos", 0.0035))
                & (self._angle_deg(m.root_quat_w, aq) <= getattr(c, "detent_deg", 5.0)))

    def distal_on_horn(self) -> torch.Tensor:
        """(n,) bool: the forearm is HELD on the servo's horn — snapped on or screwed. From the
        scene's state when exposed; otherwise from the pose, within the snap capture tolerance."""
        c = self.scene.cfg
        screwed, snapped = self._fastened(horn=True), self._env_flags("_fork_snap_on")
        if screwed is not None or snapped is not None:
            import torch

            out = torch.zeros(self.env.num_envs, dtype=torch.bool)
            for flags in (screwed, snapped):
                if flags is not None:
                    out |= flags
            return out
        m = _obj(self.env, "motor").data
        if not getattr(c, "fork_snap", True):  # no snap: only screws hold it
            return (m.root_pos_w[:, 0] != m.root_pos_w[:, 0])  # all False
        n = m.root_pos_w.shape[0]
        seat_p = m.root_pos_w + quat_rotate(m.root_quat_w, m.root_pos_w.new_tensor(c.elbow_lower_arm_seat_pos).expand(n, 3))
        seat_q = _quat_mul(m.root_quat_w, m.root_quat_w.new_tensor(c.elbow_lower_arm_seat_quat).expand(n, 4))
        d = _obj(self.env, "distal").data
        return (((d.root_pos_w - seat_p).norm(dim=-1) <= c.fork_snap_pos)
                & (self._angle_deg(d.root_quat_w, seat_q) <= c.fork_snap_deg))

    def rest_height(self, part: str) -> float:
        """The part's root height above the bench when lying there (its spawn rest pose)."""
        c = self.scene.cfg
        return {"motor": c.motor_spawn_z, "distal": c.distal_spawn_z, "proximal": 0.0}[part]

    def on_bench(self, part: str) -> torch.Tensor:
        """(n,) bool: `part` is down on the bench — no higher than its rest height + `rest_tol`."""
        z = _obj(self.env, part).data.root_pos_w[:, 2] - self._surface_z()
        return z <= self.rest_height(part) + self.rest_tol

    def supported(self, obj: str) -> torch.Tensor:
        """(n,) bool: `obj` rests on the bench, or the scene holds it where it belongs (servo in
        its pocket; forearm on the horn of a servo that is in its pocket) — and it is not moving."""
        d = _obj(self.env, obj).data
        still = d.root_lin_vel_w.norm(dim=-1) < self.still
        ang = getattr(d, "root_ang_vel_w", None)
        if ang is not None:
            still &= ang.norm(dim=-1) < self.still_ang
        ok = self.on_bench(obj)
        if obj == "motor":
            ok |= self.motor_in_pocket().to(ok.device)
        elif obj == "distal":  # snapped onto the servo, and the servo held by the pocket
            ok |= (self.distal_on_horn() & self.motor_in_pocket()).to(ok.device)
        return ok & still
