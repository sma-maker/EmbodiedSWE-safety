"""Knife handling shared by slice and dice: a firm, proven grip; slow cuts; never drop the knife.

The knife is held by a friction pinch (no weld). Pressing the blade into food pushes back on the
pinch through the lever from the handle to the edge, so the grip must carry the knife's weight
AND that twist. `secure_test` proves the hold before any cut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robobench.core.prevention import CarryPrevention, add_post_step, quat_rotate, required_grip_force, _guards, _obj

if TYPE_CHECKING:
    import torch

G = 9.81
#: Assumptions, not from the repo — calibrate on hardware:
PAD_FRICTION = 0.5  # pads on a polymer handle
PAD_SPAN = 0.04  # m, finger contact length along the handle (resists twist as a couple)
CUT_FORCE = 10.0  # N, pushing the edge through a carrot / potato
HANDLE_TO_EDGE = 0.08  # m, the grip to the middle of the edge


class KnifePrevention(CarryPrevention):
    """Grip from the cutting load, a secure test before cutting, slow cuts near the board, release
    only resting, and a watchdog for drops and slips. Subclasses add where the cut may go."""

    def __init__(self, env, cut_force: float = CUT_FORCE, slip_tol: float = 0.002, slip_deg: float = 2.0,
                 cut_speed: float = 0.03, near_board: float = 0.03, drop_speed: float = 1.0, hand_reach: float = 0.12,
                 carry_speed: float = 0.2, carry_height: float = 0.15, grip_force_max: float = 40.0, **kw) -> None:
        import torch

        c = env.scene.cfg
        self.twist_Nm = cut_force * HANDLE_TO_EDGE
        self.needed = required_grip_force(float(c.knife_mass), PAD_FRICTION, tangential_load=self.twist_Nm / PAD_SPAN)
        super().__init__(env, self.needed, grip_force_max, carry_speed=carry_speed, carry_height=carry_height, **kw)
        self.slip_tol, self.slip_deg, self.cut_speed, self.near_board = slip_tol, slip_deg, cut_speed, near_board
        self.drop_speed, self.hand_reach = drop_speed, hand_reach
        self.secure = torch.zeros(env.num_envs, dtype=torch.bool)
        self._hands = [(g.art, g.ee_idx) for g in _guards(env) if g.ee_idx is not None]
        self.incidents: list[dict] = []
        self._latched: dict = {}
        self._kstep = 0
        self.speed.add_limit(self._cut_limit)
        add_post_step(env, self._knife_watch)

    def grip_report(self) -> dict:
        return {"set_N": round(self.grip.force, 2), "max_N": self.grip.force_max, "needed_N": round(self.needed, 2),
                "twist_Nm": round(self.twist_Nm, 3), "capped": self.needed > self.grip.force_max}

    # -- the blade -------------------------------------------------------------------------------
    def edge_points(self):
        """(n, S, 3) the real cutting edge in the world (the scene's edge samples, knife frame)."""
        d = _obj(self.env, "knife").data
        loc = self.scene._edge_local.to(d.root_pos_w.device)  # (S, 3)
        n, s = d.root_pos_w.shape[0], loc.shape[0]
        q = d.root_quat_w[:, None].expand(n, s, 4).reshape(n * s, 4)
        return d.root_pos_w[:, None] + quat_rotate(q, loc[None].expand(n, s, 3).reshape(n * s, 3)).reshape(n, s, 3)

    def board_top(self):
        o = getattr(self.env.iscene, "env_origins", None)
        z = float(getattr(self.scene, "_board_top", self.scene.cfg.surface_z))
        return z if o is None else o[:, 2] + z

    def edge_height(self):
        """(n,) the lowest edge point above the board top."""
        return self.edge_points()[..., 2].amin(dim=1) - self.board_top()

    # -- firm, proven grip -----------------------------------------------------------------------
    def secure_test(self, step_fn, steps: int = 20) -> bool:
        """A short test motion: the knife must move rigidly with the hand (slide under `slip_tol`,
        pivot under `slip_deg`). Only then may it cut (`self.secure`)."""
        self.grip.force = min(self.grip.force_max, max(self.grip.force, self.needed))
        mon = self.hold("knife")
        mon.slip_tol, mon.slip_deg = self.slip_tol, self.slip_deg
        mon.lock()
        for _ in range(steps):
            step_fn()
        ok = (~mon.slipping & (mon.drift() <= self.slip_tol) & (mon.turn_deg() <= self.slip_deg)).cpu()
        self.secure[:] = ok
        if not bool(self.carrying().any()) and bool(ok.all()):
            self.mark_held("knife")
        return bool(ok.all())

    # -- slow near the board ---------------------------------------------------------------------
    def _cut_limit(self):
        import torch

        held = self.carrying()
        near = (self.edge_height() < self.near_board).to(held.device)
        slow = held & near
        return torch.where(slow, torch.full_like(slow, self.cut_speed, dtype=torch.float32),
                           torch.full_like(slow, float("inf"), dtype=torch.float32))

    def supported(self, obj: str, still: float = 0.03):
        """The knife may be let go only lying still at counter height (its rest or the board)."""
        d = _obj(self.env, obj).data
        o = getattr(self.env.iscene, "env_origins", None)
        z = d.root_pos_w[:, 2] - (0.0 if o is None else o[:, 2])
        low = z < float(getattr(self.scene.cfg, "island_top", self.scene.cfg.surface_z)) + 0.08
        return low & (d.root_lin_vel_w.norm(dim=-1) < still)

    # -- watchdog ---------------------------------------------------------------------------------
    def report(self, kind: str, mask, info=lambda e: {}) -> None:
        import torch

        mask = mask.cpu()
        prev = self._latched.get(kind)
        prev = torch.zeros_like(mask) if prev is None else prev
        new = mask & ~prev
        self._latched[kind] = mask
        if not bool(new.any()):
            return
        for e in new.nonzero().flatten().tolist():
            self.incidents.append({"env": e, "kind": kind, "step": self._kstep, **info(e)})
        shield = getattr(self.env, "safety", None)
        if shield is not None and hasattr(shield, "flag"):
            shield.flag(kind, new.to(self.gate_closed.device), {"what": kind})

    def _knife_watch(self) -> None:
        import torch

        self._kstep += 1
        d = _obj(self.env, "knife").data
        # out of the hand = away from every hand (a knife slipping out is still "registered")
        free = torch.ones(self.env.num_envs, dtype=torch.bool) if self._hands else \
            torch.tensor([o != "knife" for o in self._held_obj])
        for art, idx in self._hands:
            free &= ((art.data.body_pos_w[:, idx] - d.root_pos_w).norm(dim=-1) > self.hand_reach).cpu()
        self.report("knife_dropped", free & (d.root_lin_vel_w[:, 2] < -self.drop_speed).cpu())
        mon = self._holds.get("knife")
        held = ~torch.tensor([o != "knife" for o in self._held_obj])
        if mon is not None:
            slipping = mon.slipping.cpu() & held
            self.secure &= ~slipping
            self.report("knife_slipping", slipping)
