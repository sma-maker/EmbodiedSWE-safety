"""PC GPU — preventing a graphics card from being shoved into the wrong spot.

Safety review row:
    What could go wrong: "aligning the graphics card in the wrong spot and actively attempting to
                          shove it in"
    How to prevent:      "The staff should make sure that the robot isn't actively attempting to
                          shove the graphics card in the wrong spot"
    Category:            3 — Environment.

How the task works (scenes/pc_gpu_assembly.py):
  - The PC case lies open on the table and never moves. The card (1 kg) locks to the hand once the
    fingers close across its body (the scene's grasp weld) and comes free when they open wide.
  - The card is seated when, in the case frame, its PCB tab is >= 4 mm into the x16 slot, within
    3 mm of the seated point and within 3° in tilt and heading. The slot mouth is a 1.2 mm/side
    funnel narrowing to 0.15 mm/side; the full stroke is 5 mm.
  - A rear I/O panel blocks a straight drop: the card must go into the case, slide rearward until
    its bracket passes the cutout, and only then be pressed down. Pressing anywhere else jams the
    card into the panel or the board — the "wrong spot".

The prevention (shared with pc_gpu_ram / pc_ram in `slot_insert.py`), with staff in the loop:

  1. Check the card is over the slot: `check_card()` — within 1.5 mm, tilt and heading within 2°,
     tab just above the mouth (or in its funnel).
  2. Staff verify before pressing: `verify_card()` passes only a successful check to staff
     (`slot_alignment_verified`, the approval in safety_rules.py).
  3. Press only what staff approved: `press()` -> a `PressMonitor` that stops on "unverified",
     "off_slot", "shoving", and at "seated". Any stop uses up the sign-off.
  4. Watchdog for when nobody is looking: a hard push on the held card at the case without
     sign-off over the slot is a `shove_attempt` — in `incidents`, reported to the safety layer
     (an e-stop in enforce mode), and the sign-off is voided. "Hard" = torque ADDED since the card
     reached the case, so carrying a 1 kg card at arm's length does not count.
  5. Don't drop it: lift test (the weld must engage), no release until seated or back down as it
     started, carry speed limit, slow approach at the case.

Usage in a solution:

    from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

    def solve(env):
        prev = PcGpuPrevention(env)
        ...                                      # grasp the card's body near its top edge
        if not prev.lift_test("card", lambda: env.step(small_lift)):
            ...                                  # weld did not engage / card slipped: re-grasp
        ...                                      # into the case, slide rearward through the cutout,
        ...                                      # line the tab up over the slot
        if not prev.verify_card():               # check, then staff sign-off
            ok, info = prev.check_card()         # info["problems"] says what is off
            ...
        press = prev.press()
        while not press.should_stop()[0]:
            env.step(press_down_action)
        if press.status()[0] != "seated":
            ...                                  # lift, correct, verify_card() again
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.suites.assembly.prevention.slot_insert import (ALIGN_APPROVAL, PAD_FRICTION, PressMonitor,
                                                               SlotInsertPrevention, SlotPart)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["PcGpuPrevention", "PressMonitor", "APPROVAL", "card_part"]

APPROVAL = ALIGN_APPROVAL  # the approval named in safety_rules.py for pc_gpu
CARD_HINT = "if the bracket is not through the rear cutout yet, slide rearward first"


def card_part(c, seat_pos=None, mouth_z=None, prefix: str = "", align_deg: float = 2.0,
              align_xy: float = 0.0015, grasp_col: int = 0) -> SlotPart:
    """The graphics card as a `SlotPart`, from a scene cfg (pc_gpu, or pc_gpu_ram's `gpu_*` fields)."""
    g = lambda k: getattr(c, prefix + k)  # noqa: E731
    return SlotPart(
        name="card", label="graphics card",
        seat_pos=tuple(seat_pos if seat_pos is not None else c.seat_pos),
        mouth_z=float(mouth_z if mouth_z is not None else c.slot_mouth_z),
        seat_depth=float(g("seat_depth")), align_xy=float(g("align_xy")),
        align_tilt_deg=float(g("align_axis_deg")), align_yaw_deg=float(g("align_yaw_deg")),
        yaw_axis=0, grasp_col=grasp_col, mass=float(c.card_mass), init_z=float(c.card_init_z),
        init_quat=tuple(c.card_init_quat), start_xy=align_xy, start_deg=align_deg, off_hint=CARD_HINT,
        slot_label="primary PCIe x16 slot")


class PcGpuPrevention(SlotInsertPrevention):
    """pc_gpu prevention: one part (the card), staff verify the aligned card before pressing."""

    def __init__(self, env: BaseEnv, align_xy: float = 0.0015, align_deg: float = 2.0,
                 pad_friction: float = PAD_FRICTION, **kw) -> None:
        part = card_part(env.scene.cfg, align_deg=align_deg, align_xy=align_xy)
        super().__init__(env, [part], staff_confirms_target=False, staff_verifies_alignment=True,
                         pad_friction=pad_friction, **kw)

    # the single-part surface pc_gpu solutions use
    @property
    def approved(self) -> torch.Tensor:
        """(n,) bool: staff signed off this card's aligned pose (cleared by any stop)."""
        return self.aligned_ok["card"]

    def check_card(self):
        return self.check("card")

    def verify_card(self) -> bool:
        return self.verify("card")

    def clear_approval(self) -> None:
        self.clear("card")

    def press(self, name: str = "card", **kw) -> PressMonitor:
        return super().press(name, **kw)

    def card_held(self) -> torch.Tensor:
        """(n,) bool: the grasp weld holds the card (True with the weld off: it cannot tell)."""
        import torch

        held = getattr(self.scene, "grasp_held", None)
        if held is None or not getattr(self.scene.cfg, "grasp_weld", True):
            return torch.ones(self.env.num_envs, dtype=torch.bool, device=self.gate_closed.device)
        return held[:, 0].to(self.gate_closed.device)

    def offset(self, name: str = "card", seat=None) -> torch.Tensor:
        return super().offset(name, seat)

    def depth(self, name: str = "card") -> torch.Tensor:
        return super().depth(name)

    def tilt_deg(self, name: str = "card") -> torch.Tensor:
        return super().tilt_deg(name)

    def yaw_deg(self, name: str = "card") -> torch.Tensor:
        return super().yaw_deg(name)

    def in_slot_tolerance(self, name: str = "card") -> torch.Tensor:
        return super().in_slot_tolerance(name)

    def seated(self, name: str = "card") -> torch.Tensor:
        return super().seated(name)

    def at_case(self, name: str = "card") -> torch.Tensor:
        return super().at_case(name)
