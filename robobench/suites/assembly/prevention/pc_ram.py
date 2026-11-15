"""PC RAM — preventing a memory module from going into the wrong spot.

Safety review row:
    What could go wrong: "Nothing much could happen that would be a safety issue"
    How to prevent:      "just make sure not to place the module in the wrong spot"
    Category:            3 — Environment.

How the task works (scenes/pc_ram_assembly.py): two RAM sticks go into two of the board's four
DIMM slots — stick k into seat k, the outermost and the second-from-socket slot (the pair a
2-stick dual-channel kit fills). Only those two slots are physical. Each stick locks to the hand
when grasped. Seated: blade >= 3.7 mm into its slot, within 3 mm, 6° tilt and 3° heading.

The prevention is the shared `slot_insert.py` flow. The row asks for no staff step, so the robot's
own check is the gate:

  1. Check before pressing: `check(stick)` — over ITS slot within 1.5 mm and 3°, at the slot
     mouth. A stick over the other stick's slot, or over one of the two empty DIMM slots, is
     reported as the wrong slot. `verify(stick)` records a passing check.
  2. Press only a checked stick: `press(stick)` stops on "unverified", "off_slot", "shoving" and at
     "seated"; any stop uses the check up.
  3. Watchdog: a hard push on a held stick at the case that is not checked and over its slot is a
     `shove_attempt` (in `incidents`, reported to the safety layer).
  4. Don't drop it: lift test, no release until seated or back down as it started.

Usage:

    prev = PcRamPrevention(env)
    for stick in ("ram_0", "ram_1"):
        ...                                      # grasp, prev.lift_test(stick, ...), carry, align
        if not prev.verify(stick):
            ok, info = prev.check(stick)         # e.g. "over the empty DIMM slot ...: wrong slot"
            ...
        press = prev.press(stick)
        while not press.should_stop()[0]:
            env.step(press_down_action)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from robobench.suites.assembly.prevention.pc_gpu_ram import empty_dimm_slots, ram_parts
from robobench.suites.assembly.prevention.slot_insert import PAD_FRICTION, PressMonitor, SlotInsertPrevention

if TYPE_CHECKING:
    from robobench.core import BaseEnv

__all__ = ["PcRamPrevention", "PressMonitor"]


def _as_ram_cfg(c):
    """pc_ram names its fields without the `ram_` prefix pc_gpu_ram uses; map them across."""
    return SimpleNamespace(ram_seat_pos=c.seat_pos, ram_slot_mouth_z=c.slot_mouth_z, ram_seat_depth=c.seat_depth,
                           ram_align_xy=c.align_xy, ram_align_axis_deg=c.align_axis_deg,
                           ram_align_yaw_deg=c.align_yaw_deg, ram_mass=c.ram_mass, ram_init_z=c.ram_init_z,
                           ram_init_quat=c.ram_init_quat)


class PcRamPrevention(SlotInsertPrevention):
    """pc_ram prevention: both sticks, robot-checked (no staff step — the row asks for none)."""

    def __init__(self, env: BaseEnv, align_xy: float = 0.0015, align_deg: float = 3.0,
                 pad_friction: float = PAD_FRICTION, **kw) -> None:
        rc = _as_ram_cfg(env.scene.cfg)
        super().__init__(env, ram_parts(rc, align_xy=align_xy, align_deg=align_deg, grasp_col0=0),
                         empty_slots=empty_dimm_slots(rc), staff_confirms_target=False,
                         staff_verifies_alignment=False, pad_friction=pad_friction, **kw)
