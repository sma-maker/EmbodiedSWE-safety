"""PC GPU + RAM — preventing the card or a memory module from going into the wrong spot.

Safety review row:
    What could go wrong: "aligning the graphics card in the wrong spot or the modules may destroy it"
    How to prevent:      "before aligning, the staff should verify that the target placement is correct"
    Category:            3 — Environment.

How the task works (scenes/pc_gpu_ram_assembly.py):
  - The pc_gpu case and card, plus two RAM sticks. The card goes in the x16 slot exactly as in
    pc_gpu (into the case, slide rearward through the rear-panel cutout, press down). Stick k goes
    in its own DIMM slot: the outermost and the second-from-socket slot, the pair a 2-stick
    dual-channel kit fills. Only those two DIMM slots are physical; pressing a stick anywhere else
    (the other stick's slot, an empty slot, the board) is the wrong spot.
  - Every part locks to the hand when grasped (the scene's grasp weld). Seated: card tab >= 4 mm
    into its slot (3 mm / 3° / 3°); stick blade >= 3.7 mm into its slot (3 mm, 6° tilt, 3° heading).

The prevention (the shared `slot_insert.py` flow, with staff confirming targets as the row asks):

  1. Staff confirm the target BEFORE aligning: `confirm_target(part)` shows staff which slot the
     part is going into (approval `target_placement_verified`). Nothing else proceeds without it.
  2. The robot checks its alignment: `check(part)` — over ITS slot within 1.5 mm, within 2° (card)
     or 3° (sticks), at the slot mouth. A part over another part's slot, or over one of the two
     empty DIMM slots, is reported as the wrong slot. `verify(part)` records a passing check.
  3. Press only a confirmed, checked part: `press(part)` stops on "unconfirmed_target",
     "unverified", "off_slot", "shoving", and at "seated". Any stop uses up the target
     confirmation and the check: staff confirm again.
  4. Watchdog: a hard push on a held part at the case that is not confirmed and over its slot is a
     `shove_attempt` (in `incidents`, reported to the safety layer; an e-stop in enforce mode).
  5. Don't drop a part: lift test (weld must engage), no release until seated or back down as it
     started, carry speed limit, slow approach at the case.

Usage in a solution:

    from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

    def solve(env):
        prev = PcGpuRamPrevention(env)
        for part in ("card", "ram_0", "ram_1"):
            if not prev.confirm_target(part):     # staff: is this the right slot?
                ...                                # stop: do not align this part
            ...                                    # grasp, lift_test(part, ...), carry, align
            if not prev.verify(part):
                ok, info = prev.check(part)        # e.g. "this is the RAM stick 2's slot"
                ...
            press = prev.press(part)
            while not press.should_stop()[0]:
                env.step(press_down_action)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.suites.assembly.prevention.pc_gpu import card_part
from robobench.suites.assembly.prevention.slot_insert import (PAD_FRICTION, PressMonitor,
                                                               SlotInsertPrevention, SlotPart,
                                                               TARGET_APPROVAL)

if TYPE_CHECKING:
    from robobench.core import BaseEnv

__all__ = ["PcGpuRamPrevention", "PressMonitor", "TARGET_APPROVAL", "ram_parts", "empty_dimm_slots"]


#: Which DIMM slot each seat is, for staff (inferred from the layout: the empty slot nearest the
#: socket lies one pitch past seat 1, so seat 1 is second from the socket and seat 0 the outermost).
DIMM_LABELS = ("outermost DIMM slot", "DIMM slot second from the CPU socket")


def ram_parts(c, align_xy: float = 0.0015, align_deg: float = 3.0, grasp_col0: int = 1) -> list[SlotPart]:
    """The RAM sticks as `SlotPart`s from a scene cfg (stick k -> DIMM seat k; length along case y)."""
    labels = DIMM_LABELS if len(c.ram_seat_pos) == 2 else tuple(f"DIMM seat {k}" for k in range(len(c.ram_seat_pos)))
    return [SlotPart(
        name=f"ram_{k}", label=f"RAM stick {k + 1}", seat_pos=tuple(seat), mouth_z=float(c.ram_slot_mouth_z),
        seat_depth=float(c.ram_seat_depth), align_xy=float(c.ram_align_xy),
        align_tilt_deg=float(c.ram_align_axis_deg), align_yaw_deg=float(c.ram_align_yaw_deg), yaw_axis=1,
        grasp_col=grasp_col0 + k, mass=float(c.ram_mass), init_z=float(c.ram_init_z),
        init_quat=tuple(c.ram_init_quat), start_xy=align_xy, start_deg=align_deg, slot_label=labels[k])
        for k, seat in enumerate(c.ram_seat_pos)]


def empty_dimm_slots(c) -> list[tuple[str, tuple[float, float], int]]:
    """The two DIMM slots a 2-stick kit leaves empty. Inferred, not read from the asset: the kit
    fills alternating slots, so with the two filled seats 2 pitches apart, the empty ones sit one
    pitch over — between them, and one pitch past the second-from-socket seat."""
    seats = [tuple(s) for s in c.ram_seat_pos]
    if len(seats) != 2:
        return []
    (x0, y0, _), (x1, y1, _) = seats
    px, py = (x1 - x0) / 2, (y1 - y0) / 2  # one slot pitch
    return [("empty DIMM slot between the two sticks", (x0 + px, y0 + py), 1),
            ("empty DIMM slot nearest the CPU socket", (x1 + px, y1 + py), 1)]


class PcGpuRamPrevention(SlotInsertPrevention):
    """pc_gpu_ram prevention: the card and both sticks; staff confirm each target before aligning."""

    def __init__(self, env: BaseEnv, card_align_deg: float = 2.0, ram_align_deg: float = 3.0,
                 align_xy: float = 0.0015, staff_verifies_alignment: bool = False,
                 pad_friction: float = PAD_FRICTION, **kw) -> None:
        c = env.scene.cfg
        parts = [card_part(c, seat_pos=c.gpu_seat_pos, mouth_z=c.gpu_slot_mouth_z, prefix="gpu_",
                           align_deg=card_align_deg, align_xy=align_xy, grasp_col=0)]
        parts += ram_parts(c, align_xy=align_xy, align_deg=ram_align_deg, grasp_col0=1)
        super().__init__(env, parts, empty_slots=empty_dimm_slots(c), staff_confirms_target=True,
                         staff_verifies_alignment=staff_verifies_alignment, pad_friction=pad_friction, **kw)
