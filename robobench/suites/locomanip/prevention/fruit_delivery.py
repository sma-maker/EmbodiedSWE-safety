"""Fruit delivery — preventing fruit from being dropped on the plate (or on the way).

Safety review row:
    What could go wrong: "dropping the fruit on the plate may break the plate or damage the fruit"
    How to prevent:      "place the fruit on the plate and do not drop"
    Category:            3 — Environment.

How the task works (scenes/fruit_delivery.py, built on packing's fruits_on_plate): a G1 humanoid
picks fruit on one side of a 1.0 m-deep table and WALKS around it (~2.5 m per delivery) to a
clay plate on the far side; produce distractors must stay off the plate. A fruit counts on the
plate inside its counted volume (plate body frame, measured from the dish centre), settled.

It is the clear_organic_objects prevention with the plate as the target:

  1. Set it down before letting go: `set_down_test(fruit, step_fn)` — the fruit stopped
     descending while pressing on something (the plate, or fruit already on it); then
     `release_ok(fruit)`.
  2. Lower it slowly onto the plate: `place_speed` within `slow_above` of the rim.
  3. Carried while walking: the shared carry speed limit; a fruit that leaves the hand mid-walk is
     caught by the watchdog.
  4. Watchdog: `fruit_dropped` (out of the hand, falling faster than ~5 cm free fall would) and
     `fruit_thrown` (flying sideways), reported to the safety layer.
"""

from __future__ import annotations

from robobench.core.prevention import quat_rotate_inverse, _obj
from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

__all__ = ["FruitDeliveryPrevention"]


class FruitDeliveryPrevention(ClearOrganicPrevention):
    DROP_KIND, THROW_KIND = "fruit_dropped", "fruit_thrown"

    def __init__(self, env, carry_speed: float = float("inf"), **kw) -> None:
        # no carry speed limit while walking: the hand's world speed includes the walk itself
        super().__init__(env, carry_speed=carry_speed, **kw)

    PRESENT_FN = "fruits_present"  # organics() / organic_present() list the fruit, per env

    # the plate replaces the bin: same interface, plate body frame measured from the dish centre
    def bin_local(self, item: str):
        p = _obj(self.env, "plate").data
        loc = quat_rotate_inverse(p.root_quat_w, _obj(self.env, item).data.root_pos_w - p.root_pos_w)
        off = loc.new_tensor(self.scene.cfg.plate_center_offset)
        return loc - loc.new_tensor((float(off[0]), float(off[1]), 0.0))

    def _bin_box(self):
        c = self.scene.cfg
        r = c.radius_frac * c.plate_radius
        return r, r, c.plate_rim_z

    def over_bin(self, item: str):
        r, _, _ = self._bin_box()
        return self.bin_local(item)[:, :2].norm(dim=-1) <= r

    def in_bin(self, item: str):
        c = self.scene.cfg
        z = self.bin_local(item)[:, 2]
        return self.over_bin(item) & (z >= c.floor_local_z) & (z <= c.plate_rim_z + c.stack_local_z)

    over_plate, on_plate = over_bin, in_bin
