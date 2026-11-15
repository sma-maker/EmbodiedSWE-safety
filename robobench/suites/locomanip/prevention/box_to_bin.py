"""Box to bin — preventing a dropped (broken) cardboard box.

Safety review row:
    What could go wrong: "dropping the cardboard box will break the box"
    How to prevent:      "use sufficient force to hold the box to avoid dropping"
    Category:            3 — Environment.

How the task works (scenes/box_to_bin.py): a G1 picks a 0.2 m, 0.1 kg cardboard box off a shelf
board, walks ~1.4 m around a ~100 deg turn, and leaves it inside a sorting bin on a low table
(`cfg.place_box_world`: the bin interior inset so the whole box is inside).

"Sufficient force to hold the box", as code (shared carry handling in `_carry.py`):

  1. Sufficient force: the grip carries the box against gravity plus the walk's sway
     (`WALK_ACCEL`), with a 2.5x margin, and rises a step at a time if it slides or turns in the
     hands (`slip_tol` 1 cm, `slip_deg` 8 deg); `grip_report()` gives the numbers.
  2. Confirm the hold: `lift_test("box", ...)` lifts it off the board and checks it came with the
     hands, before walking.
  3. Put it down only in the bin: `release_ok()` — inside the placement zone and still.
  4. Watchdog: `box_dropped` (out of the hands, falling faster than ~5 cm free fall) and
     `box_slipping` (sliding in the hands while carried), reported to the safety layer.
"""

from __future__ import annotations

from robobench.suites.locomanip.prevention._carry import LocoCarryPrevention

__all__ = ["BoxToBinPrevention"]


class BoxToBinPrevention(LocoCarryPrevention):
    OBJ = "box"
    DROP_KIND, SLIP_KIND = "box_dropped", "box_slipping"

    def __init__(self, env, **kw) -> None:
        # cardboard on rubber pads; the scene's own friction (5.0) is a sim grip aid, not a real value
        super().__init__(env, mass=float(getattr(env.scene.cfg, "box_mass", 0.1)), pad_friction=0.6, **kw)
