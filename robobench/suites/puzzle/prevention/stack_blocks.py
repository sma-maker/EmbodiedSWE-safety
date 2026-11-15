"""Stack blocks — preventing the tower from falling.

Safety review row:
    What could go wrong: "if the tower falls, the blocks may be damaged upon impact"
    How to prevent:      "The robot should make sure that the block is being placed directly on top of
                          the previous block"
    Category:            3 — Environment.

How the task works (scenes/stack_blocks.py): four 4.5 cm, 25 g cubes go one by one into a
single tower over a pad (`cfg.pad_pos`). The scene counts a block in the tower within 3 cm of the
pad axis — but a cube more than half its width (2.25 cm) off the block below overhangs its own
centre of mass, so the scene's tolerance alone allows a tower that leans and falls. The robot
is a G1 humanoid.

"Directly on top of the previous block", as code:

  1. `check_place(block)` before letting go: centred over the tower within `place_tol` (1 cm —
     well inside the 2.25 cm tipping point), its faces lined up with the block below within
     `yaw_tol` (cubes: modulo 90 deg), level, and resting at the next slot's height (on the
     block below, not hovering). `release_ok(block)` adds "still". Parallel grippers stay
     closed (the shared release gate) until then or until the block is back on the table.
  2. Lower it slowly onto the tower: `place_speed` within `slow_above` of the tower top.
  3. Watchdog: `tower_fell` (the tower got shorter — a block came off) and `block_dropped` (a
     block out of the hand falls faster than `drop_speed`). Recorded in `incidents` and
     reported to the safety layer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robobench.core.prevention import (CarryPrevention, add_post_step, parallel_grippers, quat_rotate,
                                       required_grip_force, _guards, _obj)

if TYPE_CHECKING:
    import torch

    from robobench.core import BaseEnv

__all__ = ["StackPrevention"]


class StackPrevention(CarryPrevention):
    def __init__(self, env: BaseEnv, place_tol: float = 0.01, yaw_tol: float = 15.0, rest_tol: float = 0.005,
                 place_speed: float = 0.03, slow_above: float = 0.06, drop_speed: float = 1.0,
                 hand_reach: float = 0.08, carry_speed: float = 0.3, carry_height: float = 0.15,
                 grip_force_max: float = 20.0, gate_fingers: bool | None = None, **kw) -> None:
        c = env.scene.cfg
        self.blocks = list(c.names)
        super().__init__(env, required_grip_force(float(c.block_mass), 0.5), grip_force_max, carry_speed=carry_speed,
                         carry_height=carry_height, gate_fingers=parallel_grippers(env) if gate_fingers is None else gate_fingers, **kw)
        self.place_tol, self.yaw_tol, self.rest_tol = place_tol, yaw_tol, rest_tol
        self.place_speed, self.slow_above, self.drop_speed, self.hand_reach = place_speed, slow_above, drop_speed, hand_reach
        self._hands = [(g.art, g.ee_idx) for g in _guards(env) if g.ee_idx is not None]
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._height_prev = self.tower_height().clone()
        self._step = 0
        self.speed.add_limit(self._place_limit)
        add_post_step(env, self._watch)

    # -- tower geometry (env-local) --------------------------------------------------------------
    def _local(self, b: str):
        import torch

        o = getattr(self.env.iscene, "env_origins", None)
        p = _obj(self.env, b).data.root_pos_w
        return p - (torch.zeros_like(p) if o is None else o)

    def _axes_up(self, b: str):
        """(n, 3) the world-up component of each body axis — the most vertical tells 'level'."""
        import torch

        q = _obj(self.env, b).data.root_quat_w
        n = q.shape[0]
        return torch.stack([quat_rotate(q, q.new_tensor(e).expand(n, 3))[:, 2]
                            for e in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))], dim=1)

    def level(self, b: str) -> torch.Tensor:
        return self._axes_up(b).abs().max(dim=1).values >= math.cos(math.radians(10.0))

    def yaw_deg(self, b: str) -> torch.Tensor:
        """(n,) heading of a level cube's side faces, modulo 90 deg."""
        import torch

        q = _obj(self.env, b).data.root_quat_w
        x = quat_rotate(q, q.new_tensor((1.0, 0.0, 0.0)).expand(q.shape[0], 3))
        y = quat_rotate(q, q.new_tensor((0.0, 1.0, 0.0)).expand(q.shape[0], 3))
        side = torch.where((x[:, 2].abs() < y[:, 2].abs())[:, None], x, y)  # a horizontal body axis
        return torch.rad2deg(torch.atan2(side[:, 1], side[:, 0])) % 90.0

    def slot(self, b: str):
        """(level index (n,), on the tower axis and at a slot height (n,) bool)."""
        c = self.scene.cfg
        p = self._local(b)
        pad = p.new_tensor(c.pad_pos)
        z0 = c.surface_z + c.block_size / 2
        lvl = ((p[:, 2] - z0) / c.block_size).round()
        at = ((p[:, 2] - z0 - lvl * c.block_size).abs() < c.z_tol) & ((p[:, :2] - pad).norm(dim=-1) < c.align_tol)
        return lvl.long(), at & self.level(b) & (lvl >= 0)

    def tower_height(self, exclude: str | None = None) -> torch.Tensor:
        """(n,) blocks in the contiguous tower from the pad up (no settle requirement)."""
        import torch

        n = self.env.num_envs
        c = self.scene.cfg
        filled = torch.zeros(n, c.n_blocks + 1, dtype=torch.bool)
        for b in self.blocks:
            if b == exclude:
                continue
            lvl, at = self.slot(b)
            for e in range(n):
                if bool(at[e]) and 0 <= int(lvl[e]) < c.n_blocks:
                    filled[e, int(lvl[e])] = True
        h = torch.zeros(n, dtype=torch.long)
        for e in range(n):
            while int(h[e]) < c.n_blocks and bool(filled[e, int(h[e])]):
                h[e] += 1
        return h

    def top_block(self, height, exclude: str):
        """Per env, the block at level height-1 (None at the pad)."""
        out = []
        for e in range(self.env.num_envs):
            found = None
            for b in self.blocks:
                if b == exclude:
                    continue
                lvl, at = self.slot(b)
                if bool(at[e]) and int(lvl[e]) == int(height[e]) - 1:
                    found = b
            out.append(found)
        return out

    # -- 1. directly on top ----------------------------------------------------------------------
    def check_place(self, block: str):
        import torch

        c = self.scene.cfg
        h = self.tower_height(exclude=block)
        below = self.top_block(h, block)
        p = self._local(block)
        n = p.shape[0]
        ok = torch.zeros(n, dtype=torch.bool)
        problems = []
        for e in range(n):
            q = []
            if below[e] is None:
                target = p.new_tensor(c.pad_pos)
                yaw_ok = True
            else:
                target = self._local(below[e])[e, :2]
                dyaw = abs(float(self.yaw_deg(block)[e]) - float(self.yaw_deg(below[e])[e])) % 90.0
                yaw_ok = min(dyaw, 90.0 - dyaw) <= self.yaw_tol
                if not yaw_ok:
                    q.append(f"turned {min(dyaw, 90.0 - dyaw):.0f} deg from the block below (max {self.yaw_tol:.0f})")
            off = float((p[e, :2] - target).norm())
            if off > self.place_tol:
                q.append(f"{off * 100:.1f} cm off the {'pad centre' if below[e] is None else 'block below'} "
                         f"(max {self.place_tol * 100:.0f}; it tips past {c.block_size / 2 * 100:.2f})")
            if not bool(self.level(block)[e]):
                q.append("not level")
            want = c.surface_z + c.block_size / 2 + int(h[e]) * c.block_size
            if abs(float(p[e, 2]) - want) > self.rest_tol:
                q.append(f"{(float(p[e, 2]) - want) * 1000:+.0f} mm from resting on {'the pad' if below[e] is None else 'the block below'}")
            ok[e] = not q
            problems.append(q)
        return ok, {"block": block, "tower_height": h.tolist(), "problems": problems}

    def release_ok(self, block: str, still: float = 0.03):
        slow = (_obj(self.env, block).data.root_lin_vel_w.norm(dim=-1) < still).cpu()
        return self.check_place(block)[0] & slow

    def supported(self, obj: str):
        c = self.scene.cfg
        on_table = ((self._local(obj)[:, 2] - c.surface_z - c.block_size / 2).abs() < self.rest_tol).cpu()
        return (self.release_ok(obj) | on_table).to(self.gate_closed.device)

    # -- 2. lower it slowly ----------------------------------------------------------------------
    def _place_limit(self):
        import torch

        c = self.scene.cfg
        held = self.carrying()
        slow = torch.zeros_like(held)
        for b in self.blocks:
            mine = torch.tensor([o == b for o in self._held_obj], device=held.device)
            if bool(mine.any()):
                p = self._local(b)
                top = c.surface_z + self.tower_height(exclude=b).float() * c.block_size
                over = (p[:, :2] - p.new_tensor(c.pad_pos)).norm(dim=-1) < 0.06
                near = (over.cpu() & ((p[:, 2].cpu() - c.block_size / 2 - top) < self.slow_above))
                slow |= mine & near.to(held.device)
        return torch.where(slow, torch.full_like(slow, self.place_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    # -- 3. watchdog ------------------------------------------------------------------------------
    def _report(self, kind: str, key: str, mask, info=lambda e: {}) -> None:
        import torch

        mask = mask.cpu()
        prev = self._latched.get((kind, key))
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[(kind, key)] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "what": key, "step": self._step, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new.to(self.gate_closed.device), {"what": key})

    def _watch(self) -> None:
        import torch

        self._step += 1
        h = self.tower_height()
        self._report("tower_fell", "tower", h < self._height_prev,
                     lambda e: {"from": int(self._height_prev[e]), "to": int(h[e])})
        self._height_prev = h.clone()
        for b in self.blocks:
            p = _obj(self.env, b).data
            # out of the hand = away from every hand (not "not registered": a block can slip out mid-carry)
            free = torch.ones(self.env.num_envs, dtype=torch.bool)
            if not self._hands:
                free &= ~torch.tensor([o == b for o in self._held_obj])
            for art, idx in self._hands:
                free &= ((art.data.body_pos_w[:, idx] - p.root_pos_w).norm(dim=-1) > self.hand_reach).cpu()
            self._report("block_dropped", b, free & (p.root_lin_vel_w[:, 2] < -self.drop_speed).cpu())
