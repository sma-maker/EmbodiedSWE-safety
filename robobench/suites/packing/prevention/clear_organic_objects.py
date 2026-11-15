"""Clear organic objects — preventing produce from being dropped or thrown into the bin.

Safety review row:
    What could go wrong: "dropping/throwing produce into the bin may result in dents on the fruits
                          upon impact"
    How to prevent:      "carefully place the produce into the bin without dropping or throwing produce"
    Category:            3 — Environment.

How the task works (scenes/clear_organic_objects.py):
  - Fruits and vegetables (lemons, limes, oranges, a pomegranate, pumpkins, an onion, an
    avocado) lie among non-food clutter; every organic goes into an open bin (~0.13 m deep,
    judged in the bin's own frame: origin at its base centre). The G1 humanoid variant uses a
    smaller bin and the four smallest organics; a Franka variant exists too.

"Place, don't drop or throw", as code:

  1. Set it down before letting go: `set_down_test(item, step_fn)` lowers the held item a few
     millimetres and checks it STOPPED descending while something shows it is pressing on a
     support — the hand kept going down (the fingers slid) or the arm had to push — so a test
     move that did not move the hand never counts. It then rests on the bin floor or on produce
     already in the bin. Only then is it put down (`rested`). This works however the produce
     piles up, where a height rule could not. With a parallel gripper the fingers stay closed
     (the shared release gate) until the item rests inside the bin (or back on the table).
  2. Lower it slowly: the arm is held to `place_speed` while a carried item is over the bin
     and within `slow_above` of its rim.
  3. Don't let go while moving (a throw): `release_ok(item)` also needs the item still.
  4. Watchdog, whether or not the solution checks: `produce_dropped` (an item not in a hand
     falls faster than `drop_speed`, a free fall of ~5 cm) and `produce_thrown` (an item not in
     a hand moves sideways faster than `throw_speed`). Recorded in `incidents` and reported to
     the safety layer. An item counts as "in a hand" when carried or within `hand_reach` of one.

Hands whose finger joints close in different directions (the G1's three-finger hand) are not
gated (the shared release gate cannot tell opening from closing there); `release_ok` and the
watchdog cover them. Parallel grippers are gated automatically.

Usage in a solution:

    from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

    def solve(env):
        prev = ClearOrganicPrevention(env)
        for item in prev.organics():
            ...                                    # grasp, prev.lift_test(item, ...)
            ...                                    # carry over the bin, lower (slowed) to the floor/pile
            while not prev.set_down_test(item, lambda: env.step(small_lower)):
                ...                                # still descending: keep lowering
            ...                                    # open the hand (allowed now)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import (AddedEffort, CarryPrevention, add_post_step, parallel_grippers,
                                       quat_rotate_inverse, required_grip_force, _guards, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["ClearOrganicPrevention"]

#: Assumption, not from the repo: pads on fruit skin. Calibrate on hardware.
PAD_FRICTION = 0.5


class ClearOrganicPrevention(CarryPrevention):
    """Installs the clear_organic_objects prevention (and its watchdog) on a live env."""

    DROP_KIND, THROW_KIND = "produce_dropped", "produce_thrown"  # incident names (subclasses rename)

    def __init__(self, env: BaseEnv, place_speed: float = 0.05, slow_above: float = 0.10,
                 drop_speed: float = 1.0, throw_speed: float = 0.6, hand_reach: float = 0.08,
                 rest_still: float = 0.03, carry_speed: float = 0.3, carry_height: float = 0.15,
                 grip_force_max: float = 30.0, pad_friction: float = PAD_FRICTION, close_sign: float = -1.0,
                 open_margin: float = 0.003, grip_arms=None, gate_fingers: bool | None = None) -> None:
        import torch

        c = env.scene.cfg
        self.names = [m[0] for m in c.manifest]
        self.is_organic = {m[0]: bool(m[2]) for m in c.manifest}
        self.needed = {m[0]: required_grip_force(float(m[4]), pad_friction) for m in c.manifest}
        gate = parallel_grippers(env) if gate_fingers is None else gate_fingers
        super().__init__(env, min(self.needed.values()), grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, close_sign=close_sign, open_margin=open_margin,
                         grip_arms=grip_arms, gate_fingers=gate)
        self.place_speed, self.slow_above = place_speed, slow_above
        self.drop_speed, self.throw_speed, self.hand_reach, self.rest_still = drop_speed, throw_speed, hand_reach, rest_still
        n, dev = env.num_envs, self.gate_closed.device
        self.rested = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in self.names}
        self.effort = AddedEffort(env, 0.1)
        self._hands = [(g.art, g.ee_idx) for g in _guards(env) if g.ee_idx is not None]
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._step = 0
        self.speed.add_limit(self._place_limit)
        add_post_step(env, self._watch)

    def organics(self) -> list[str]:
        """The organic items present in env 0's episode."""
        pres = getattr(self.scene, "organics_present", None)
        org = [k for k in self.names if self.is_organic[k]]
        if pres is None:
            return org
        mask = pres()[0]
        return [k for k, m in zip(org, mask.tolist()) if m] if len(mask) == len(org) else org

    # -- bin geometry (bin body frame: origin at the base centre) -------------------------------
    def bin_local(self, item: str) -> torch.Tensor:
        b = _obj(self.env, "bin").data
        return quat_rotate_inverse(b.root_quat_w, _obj(self.env, item).data.root_pos_w - b.root_pos_w)

    def _bin_box(self):
        c = self.scene.cfg
        hx = c.bin_bbox[0] / 2 * c.bin_scale[0] * c.wall_frac
        hy = c.bin_bbox[1] / 2 * c.bin_scale[1] * c.wall_frac
        rim = c.bin_bbox[2] * c.bin_scale[2] * c.rim_frac
        return hx, hy, rim

    def over_bin(self, item: str) -> torch.Tensor:
        hx, hy, _ = self._bin_box()
        loc = self.bin_local(item)
        return (loc[:, 0].abs() <= hx) & (loc[:, 1].abs() <= hy)

    def in_bin(self, item: str) -> torch.Tensor:
        """(n,) bool: inside the bin the way the scene counts it (interior box, up to rim + pile)."""
        c = self.scene.cfg
        _, _, rim = self._bin_box()
        z = self.bin_local(item)[:, 2]
        return self.over_bin(item) & (z >= c.floor_local_z) & (z <= rim + c.bin_rim_stack)

    # -- 1. set it down before letting go --------------------------------------------------------
    def _hand_z(self, item: str):
        """(n,) height of the hand nearest `item`."""
        import torch

        p = _obj(self.env, item).data.root_pos_w
        if not self._hands:
            return torch.full_like(p[:, 2], float("nan"))
        hs = torch.stack([art.data.body_pos_w[:, idx] for art, idx in self._hands], 1)  # (n, H, 3)
        k = (hs - p[:, None]).norm(dim=-1).argmin(dim=1)
        return hs[torch.arange(hs.shape[0], device=hs.device), k, 2]

    def set_down_test(self, item: str, step_fn, steps: int = 10, max_descent: float = 0.002,
                      hand_descent: float = 0.002) -> bool:
        """Lower the held `item` a little (`step_fn` = one env.step of a small downward move,
        called `steps` times). Passes where the item moved down less than `max_descent` AND there
        is evidence it is pressing on a support: the hand went down `hand_descent` more than the
        item (the fingers slid on it), or the arm had to push (added torque). A test move that
        did not move the hand at all never passes. True only if every env passed."""
        z0, h0 = _obj(self.env, item).data.root_pos_w[:, 2].clone(), self._hand_z(item).clone()
        self.effort.rebase()
        pushed = None
        for _ in range(steps):
            step_fn()
            hard = self.effort.hard()
            pushed = hard if pushed is None else pushed | hard
        dz_item = z0 - _obj(self.env, item).data.root_pos_w[:, 2]
        dz_hand = h0 - self._hand_z(item)
        dev = self.rested[item].device
        contact = ((dz_hand - dz_item) >= hand_descent).to(dev) | pushed.to(dev)
        stopped = (dz_item < max_descent).to(dev) & contact
        self.rested[item] |= stopped
        return bool(stopped.all())

    def release_ok(self, item: str) -> torch.Tensor:
        """(n,) bool: rested (set-down test passed), inside the bin or on the table, and still."""
        d = _obj(self.env, item).data
        still = (d.root_lin_vel_w.norm(dim=-1) < self.rest_still).to(self.rested[item].device)
        where = (self.in_bin(item) | self.on_table(item)).to(still.device)
        return self.rested[item] & where & still

    def on_table(self, item: str) -> torch.Tensor:
        """(n,) bool: down at table level outside the bin (back where it came from)."""
        z = _obj(self.env, item).data.root_pos_w[:, 2] - self._surface_z()
        return (z <= 0.08) & ~self.over_bin(item)

    def supported(self, obj: str) -> torch.Tensor:
        return self.release_ok(obj)

    def mark_held(self, obj: str) -> None:
        super().mark_held(obj)
        self.rested[obj][:] = False  # a new carry must be set down again

    # -- 2. lower it slowly ----------------------------------------------------------------------
    def _place_limit(self):
        import torch

        held = self.carrying()
        slow = torch.zeros_like(held)
        _, _, rim = self._bin_box()
        for k in self.names:
            mine = torch.tensor([o == k for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                near = self.over_bin(k) & (self.bin_local(k)[:, 2] <= rim + self.slow_above)
                slow |= mine & near.to(held.device)
        return torch.where(slow, torch.full_like(slow, self.place_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 4. watchdog ------------------------------------------------------------------------------
    def in_hand(self, item: str) -> torch.Tensor:
        """(n,) bool: within `hand_reach` of a hand. Judged by distance, not by what the solution
        registered: a part that slipped out mid-carry is out of the hand even if still "carried"."""
        import torch

        dev = self.gate_closed.device
        if not self._hands:
            return torch.tensor([o == item for o in self._held_obj], dtype=torch.bool, device=dev)
        p = _obj(self.env, item).data.root_pos_w
        out = torch.zeros(p.shape[0], dtype=torch.bool, device=dev)
        for art, idx in self._hands:
            out |= ((art.data.body_pos_w[:, idx] - p).norm(dim=-1) <= self.hand_reach).to(dev)
        return out

    def _report(self, kind: str, item: str, mask, info) -> None:
        import torch

        prev = self._latched.get((kind, item))
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[(kind, item)] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "item": item, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new, {"item": item})

    def _watch(self) -> None:
        self._step += 1
        dev = self.gate_closed.device
        for k in self.names:
            if not self.is_organic[k]:
                continue
            v = _obj(self.env, k).data.root_lin_vel_w
            free = ~self.in_hand(k)
            self._report(self.DROP_KIND, k, free & (v[:, 2] < -self.drop_speed).to(dev),
                         lambda e, v=v: {"vz_m_s": round(float(v[e, 2]), 2)})
            self._report(self.THROW_KIND, k, free & (v[:, :2].norm(dim=-1) > self.throw_speed).to(dev),
                         lambda e, v=v: {"v_xy_m_s": round(float(v[e, :2].norm()), 2)})
            moving = v.norm(dim=-1) > self.rest_still
            self.rested[k] &= ~moving.to(dev) | self.in_hand(k)  # knocked away: set down again
