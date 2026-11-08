"""Allen bolt — preventing the drop.

Safety review row:
    What could go wrong: the robot may lose its grip and the bolt may fall and dent the floor/table.
    How to prevent:      apply enough pressure on the part that friction keeps the grip.
    Category:            3 — Environment.

What actually gets carried in this task (scenes/allen_bolt_assembly.py):
  - the KEY (80 g steel L-key). In simulation it is WELDED to the hand once the fingers pinch
    either arm of its hex (`scene.grasp_held`), so it cannot slip; it falls only if the fingers
    open while it is in the air, or if the weld never engaged and the arm lifts anyway.
  - the BOLT (50 g M16 cap screw), unless the scene is staged with the bolt already started in
    its hole. The bolt has NO weld: it is held by friction alone, and the scene makes it slick
    (friction 0.01) so it turns easily in the thread. It is the part that can really slip.
On real hardware both are friction grips, so "enough pressure" is the core of the prevention.

The prevention, as code (each step is something the robot does differently, not only a report):

  1. Enough pressure: `required_grip_force()` turns the physics (mass, friction, acceleration,
     a safety factor) into a grip force; `AllenBoltPrevention` sets a GripGovernor to it — firm
     enough not to slip, capped so it doesn't crush — and boosts it if a slip is detected.
  2. Confirm the hold before lifting: `lift_test()` raises the part a few millimetres and checks
     it rose with the hand (and, for the key, that the weld engaged). No lift on a failed test.
  3. Never let go in the air: a release gate holds the fingers closed while the held part is
     unsupported — opening is allowed only once the part is resting (table, hole, or socket).
  4. Carry low and slow: while something is held, the arm is speed-limited and the carry height
     is checked, so a drop that does happen falls a short way at low speed.
  5. Seat the key before turning: `key_in_socket()` checks the key tip is in the bolt's hex
     socket, so the turn does not lever the key out (a common way to fling it off).
  6. Drive without forcing: an InsertionMonitor on the bolt reports "stalled" (cross-threaded
     or binding) instead of letting the arm shove harder.

Usage in a solution:

    from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

    def solve(env):
        prev = AllenBoltPrevention(env)
        ...                                   # approach + close the fingers on the key
        up = lambda: env.step(small_lift_action)
        if not prev.lift_test("key_0", up):   # weld engaged and the key rose with the hand?
            ...                               # re-grasp; do not carry
        ...                                   # carry low: prev.carry_height_ok("key_0")
        if not prev.key_in_socket(0).all():
            ...                               # re-seat before turning
        drive = prev.drive(0)
        while drive.status()[0] == "moving":
            env.step(turn_and_press_action)
        ...                                   # opening the fingers is blocked until supported
"""


from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, InsertionMonitor, quat_rotate,
                                       quat_rotate_inverse, required_grip_force, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["AllenBoltPrevention", "required_grip_force"]

#: Part properties from scenes/allen_bolt_assembly.py (masses) and real-world pad friction.
KEY_MASS, BOLT_MASS = 0.08, 0.05  # kg
SIM_FRICTION = {"key": 0.6, "bolt": 0.01}  # the scene's authored friction (bolt is slick on purpose)
REAL_PAD_FRICTION = 0.3  # rubber-padded fingers on oily steel: a conservative real-world value
M16_HEAD_HEIGHT = 0.016  # ISO 4762 M16 socket-head height (m): the socket opens at its top


class AllenBoltPrevention(CarryPrevention):
    """Installs the allen_bolt prevention on a live env and exposes the per-step checks.

    Steps 1-4 (grip, lift test, release gate, carry low and slow) come from `CarryPrevention`;
    this class sets the grip force from the parts' physics and says when each part is supported.

    `grip_force` defaults to the larger of the key and bolt requirements under `friction_model`:
    "sim" uses the scene's authored friction (the bolt is slick: ~59 N), "real" uses rubber pads on
    steel (~3 N). The key is welded in simulation, so its sim requirement uses real friction as a
    floor. Capped at `grip_force_max`.

    `drive_torque` / `lever`: if the key is pinched so that turning it pushes ALONG the finger pads,
    that reaction (torque / lever) must also be carried by friction — at 2 N·m and 10 cm that adds
    20 N and the requirement jumps to ~70 N per finger, well over a Franka finger's 20 N rating.
    The prevention for that is grasp choice, not force: pinch across the handle so the turning
    force pushes INTO a finger. Hence the default 0 (that grasp); set it to plan for the other."""

    def __init__(self, env: BaseEnv, grip_force: float | None = None, grip_force_max: float = 60.0,
                 friction_model: str = "sim", carry_speed: float = 0.3, carry_height: float = 0.10,
                 close_sign: float = -1.0, drive_torque: float = 0.0, lever: float = 0.10,
                 open_margin: float = 0.003) -> None:
        if friction_model not in ("sim", "real"):
            raise ValueError("friction_model must be 'sim' or 'real'")
        tangential = drive_torque / lever  # the key handle's reaction while driving (N)
        self.needed_real = {
            "key": required_grip_force(KEY_MASS, REAL_PAD_FRICTION, tangential_load=tangential),
            "bolt": required_grip_force(BOLT_MASS, REAL_PAD_FRICTION),
        }
        self.needed_sim = {
            "key": self.needed_real["key"],  # welded on closure; keep the real-world floor
            "bolt": required_grip_force(BOLT_MASS, SIM_FRICTION["bolt"]),
        }
        self.needed = self.needed_sim if friction_model == "sim" else self.needed_real
        force = grip_force if grip_force is not None else max(self.needed.values())
        super().__init__(env, force, grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin)

    # -- 1. enough pressure ---------------------------------------------------------------------
    def grip_report(self) -> dict:
        """Required vs. set grip force, per finger, for both friction models."""
        return {
            "set_N": round(self.grip.force, 2), "max_N": self.grip.force_max,
            "required_real_N": {k: round(v, 2) for k, v in self.needed_real.items()},
            "required_sim_N": {"key": "welded on closure (real-world floor kept)",
                               "bolt": round(self.needed_sim["bolt"], 2)},
            "capped": max(self.needed.values()) > self.grip.force_max,
        }

    # -- 2. confirm the hold before lifting -----------------------------------------------------
    def key_held(self, i: int = 0) -> torch.Tensor:
        """(n,) bool: the scene's grasp contract welded key `i` (either grip band) to the hand."""
        import torch

        held = getattr(self.scene, "grasp_held", None)
        if held is None:
            return torch.zeros(self.env.num_envs, dtype=torch.bool)
        return held[:, 2 * i: 2 * i + 2].any(dim=1)

    def _lift_ok(self, obj: str):
        """A key only counts as held once the scene's weld engaged."""
        return self.key_held(int(obj.split("_")[1])) if obj.startswith("key_") else None

    # -- 3. never let go in the air -------------------------------------------------------------
    def supported(self, obj: str, tol: float = 0.003, still: float = 0.02) -> torch.Tensor:
        """(n,) bool: `obj` is resting — on the table, in the plate's hole, or (the key) in the
        bolt's socket — and not moving."""
        d = _obj(self.env, obj).data
        slow = d.root_lin_vel_w.norm(dim=-1) < still
        on_table = d.root_pos_w[:, 2] - self._surface_z() < self._rest_height(obj) + tol
        if obj.startswith("bolt_"):
            on_table |= self.bolt_in_hole(int(obj.split("_")[1]))
        if obj.startswith("key_"):
            on_table |= self.key_in_socket(int(obj.split("_")[1]))
        return on_table & slow

    # -- 5. seat the key before turning ---------------------------------------------------------
    def socket_pos(self, i: int = 0) -> torch.Tensor:
        """(n, 3) world position of bolt `i`'s hex socket mouth (top of the head, on its axis)."""
        c = self.scene.cfg
        d = _obj(self.env, f"bolt_{i}").data
        up = d.root_pos_w.new_tensor((0.0, 0.0, c.thread_len + M16_HEAD_HEIGHT)).expand_as(d.root_pos_w)
        return d.root_pos_w + quat_rotate(d.root_quat_w, up)

    def key_in_socket(self, i: int = 0, xy_tol: float = 0.0015, depth_min: float = 0.004) -> torch.Tensor:
        """(n,) bool: key `i`'s tip is on bolt `i`'s axis (within `xy_tol`) and at least
        `depth_min` down into the socket. The socket only leaves 0.75 mm of play per side, so a
        tip that is really inside sits well within `xy_tol`; the tolerance just absorbs pose noise."""
        bolt = _obj(self.env, f"bolt_{i}").data
        tip = _obj(self.env, f"key_{i}").data.root_pos_w  # the key's origin is its short-arm tip
        axis = quat_rotate(bolt.root_quat_w, tip.new_tensor((0.0, 0.0, 1.0)).expand_as(tip))
        v = tip - self.socket_pos(i)
        along = (v * axis).sum(-1)  # negative = below the mouth, i.e. inside
        off_axis = (v - along[:, None] * axis).norm(dim=-1)
        return (off_axis <= xy_tol) & (along <= -depth_min)

    def _bolt_offset(self, i: int) -> torch.Tensor:
        """(n, 3) bolt `i`'s tip in the frame of its NEAREST platform (hole axis = platform origin,
        +z up), the way the scene's own `engaged()` / `seated()` measure it."""
        import torch

        bolt = _obj(self.env, f"bolt_{i}").data.root_pos_w
        offs = []
        for k in range(int(getattr(self.scene.cfg, "num_pairs", 1))):
            plat = _obj(self.env, f"platform_{k}").data
            offs.append(quat_rotate_inverse(plat.root_quat_w, bolt - plat.root_pos_w))
        offs = torch.stack(offs, dim=1)  # (n, P, 3)
        near = offs[..., :2].norm(dim=-1).argmin(dim=1)
        return offs[torch.arange(offs.shape[0], device=offs.device), near]

    def bolt_depth(self, i: int = 0) -> torch.Tensor:
        """(n,) bolt `i`'s tip depth below the plate top (m; negative = above the plate)."""
        return self.scene.cfg.plate_top - self._bolt_offset(i)[:, 2]

    def bolt_in_hole(self, i: int = 0, min_depth: float = 0.002) -> torch.Tensor:
        """(n,) bool: bolt `i` stands IN a hole — on its axis (within the scene's `align_xy`) and at
        least `min_depth` into the thread. Position alone is not enough: a bolt lying on the table
        beside the platform is also 'below the plate top'."""
        off = self._bolt_offset(i)
        on_axis = off[:, :2].norm(dim=-1) <= self.scene.cfg.align_xy
        return on_axis & (self.scene.cfg.plate_top - off[:, 2] >= min_depth)

    # -- 6. drive without forcing ---------------------------------------------------------------
    def drive(self, i: int = 0, **kw) -> InsertionMonitor:
        """Monitor bolt `i` going down its thread: "seated", "moving", or "stalled" (stop and back
        off: cross-threaded or binding — pressing harder is how parts get damaged and keys slip).
        The remaining travel is measured from the bolt's ACTUAL depth now, per env, so a bolt that
        starts a few millimetres in is "seated" at the scene's `seat_depth`, not deeper."""
        remaining = (self.scene.cfg.seat_depth - self.bolt_depth(i)).clamp_min(0.0)
        return InsertionMonitor(self.env, f"bolt_{i}", axis=(0.0, 0.0, -1.0), seat_depth=remaining, **kw)

    def _rest_height(self, obj: str) -> float:
        c = self.scene.cfg
        return c.key_init_z if obj.startswith("key_") else c.bolt_init_z
