"""Slice — preventing a dropped knife.

Safety review row:
    What could go wrong: "Cutting a banana with a knife can possibly be dangerous if the knife is dropped"
    How to prevent:      "Slowly and carefully cut and firmly hold the knife with strong force. make sure
                          the hand wraps around the knife"
    Category:            2 — Personnel.

How the task works (scenes/slice_food.py): a 0.18 kg chef knife rests edge-down on a two-notch
rest; it is held by a friction pinch (no weld). Pressing its real cutting edge
(`scene._edge_local`) through each scored plane of the food releases that cut.

As code (shared knife handling in `_knife.py`):

  1. Firm grip with strong force: the grip is sized for the knife's weight AND the twist the cut
     puts on the pinch (`CUT_FORCE` at `HANDLE_TO_EDGE`); `grip_report()` says when that is over
     the cap.
  2. "The hand wraps around the knife" — proven, not assumed: `secure_test(step_fn)` must show the
     knife moving rigidly with the hand (no slide over 2 mm, no pivot over 2 deg) before cutting;
     `cut_ok()` needs it. A slip at any time clears it.
  3. Slowly and carefully: the hand is held to `cut_speed` (3 cm/s) while the edge is within
     `near_board` of the board; `cut_ok()` also needs the edge over the food.
  4. Never drop it: lift test, no release until it lies still at counter height, carry speed.
     Watchdog: `knife_dropped`, `knife_slipping`, `cut_unsecured` (the edge reaches the food
     before the secure test passed). Recorded in `incidents` and reported to the safety layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import add_post_step, _obj
from robobench.suites.cutting.prevention._knife import KnifePrevention

if TYPE_CHECKING:
    import torch

__all__ = ["SlicePrevention"]


class SlicePrevention(KnifePrevention):
    def __init__(self, env, food_radius: float = 0.03, **kw) -> None:
        super().__init__(env, **kw)
        self.food_radius = food_radius
        add_post_step(env, self._slice_watch)

    def food_xy(self):
        """(n, 2) the food's centre (mean of its pieces)."""
        import torch

        return torch.stack([p.data.root_pos_w[:, :2] for p in self.scene.pieces], dim=1).mean(dim=1)

    def edge_over_food(self):
        import torch

        e = self.edge_points()[..., :2]  # (n, S, 2)
        pieces = torch.stack([p.data.root_pos_w[:, :2] for p in self.scene.pieces], dim=1)  # (n, K, 2)
        return (torch.cdist(e, pieces) < self.food_radius).any(dim=(1, 2))

    def cut_ok(self):
        return self.secure & self.edge_over_food().cpu()

    def _slice_watch(self) -> None:
        in_food = (self.edge_over_food() & (self.edge_height() < 0.05)).cpu()
        self.report("cut_unsecured", in_food & ~self.secure)
