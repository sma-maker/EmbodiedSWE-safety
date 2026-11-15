"""Prevention helpers (robobench/core/prevention.py) and the allen_bolt prevention.

Like test_safety.py: no Isaac Sim — fake articulations / rigid objects carry the tensors the
helpers read, and each test drives apply_action -> (state change) -> post_step by hand.
"""
import math
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from test_safety import FakeArticulation, FrankaRobot, make_env


def rigid(n, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)):
    return SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([pos] * n, dtype=torch.float32),
        root_quat_w=torch.tensor([quat] * n, dtype=torch.float32),
        root_lin_vel_w=torch.tensor([vel] * n, dtype=torch.float32)))


def prev_env(n=2, rigid_objects=None, articulations=None):
    env = make_env(n, rigid=rigid_objects or {}, articulations=articulations or {})
    env.device = "cpu"
    art = env.robot.articulation
    art.data.body_pos_w = torch.zeros(n, 2, 3)
    art.data.body_pos_w[:, 1] = torch.tensor([0.4, 0.0, 0.3])  # panda_hand
    return env


def substep(env, action=None):
    d = env.robot.articulation.data
    env.robot.apply_action(action if action is not None else d.joint_pos_target.clone(), 0)
    env.robot.post_step()


@unittest.skipIf(torch is None, "needs torch")
class GovernorTests(unittest.TestCase):
    def test_grip_governor_caps_the_squeeze(self):
        from robobench.core.prevention import GripGovernor

        env = prev_env()
        GripGovernor(env, force=10.0)  # default stiffness 2000 -> at most 5 mm of squeeze
        d = env.robot.articulation.data
        action = d.joint_pos.clone()
        action[:, 7:9] = 0.0  # controller slams the fingers shut from 0.02
        d.joint_effort_target[:, 7:9] = 300.0
        substep(env, action)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], torch.full((2, 2), 0.015)))
        self.assertTrue(torch.allclose(d.joint_effort_target[:, 7:9], torch.full((2, 2), 10.0)))
        self.assertAlmostEqual(float(d.joint_pos_target[0, 0]), float(action[0, 0]))  # arm untouched

    def test_speed_governor_slows_an_overspeeding_hand(self):
        from robobench.core.prevention import SpeedGovernor

        env = prev_env()
        SpeedGovernor(env, ee_speed_max=0.5)
        d = env.robot.articulation.data
        d.body_lin_vel_w[0, 1] = torch.tensor([2.0, 0.0, 0.0])  # env 0's hand at 2 m/s
        d.joint_vel[0, 0] = 1.0
        action = d.joint_pos.clone()
        action[:, 0] += 0.1
        substep(env, action)
        lead = d.joint_pos_target[:, 0] - d.joint_pos[:, 0]
        self.assertAlmostEqual(float(lead[0]), 0.1 * 0.25, places=4)  # scaled by 0.5 / 2.0
        self.assertAlmostEqual(float(lead[1]), 0.1, places=4)  # env 1 under the limit
        self.assertLess(float(d.joint_effort_target[0, 0]), 0.0)  # braking torque added

    def test_helpers_register_with_the_safety_shield_without_tripping_tamper(self):
        from robobench.core.prevention import GripGovernor
        from robobench.core.safety import install_safety

        env = prev_env()
        shield = install_safety(env)
        GripGovernor(env, force=10.0)
        self.assertEqual(len(shield.pre_filters), 1)
        self.assertTrue(shield.intact())


@unittest.skipIf(torch is None, "needs torch")
class CheckTests(unittest.TestCase):
    def test_placement_guard_slows_near_the_destination_and_gates_release(self):
        from robobench.core.prevention import PlacementGuard

        egg = rigid(2, pos=(0.0, 0.0, 0.40))  # 13 cm above its rest height: outside the slow zone
        env = prev_env(rigid_objects={"egg_0": egg})
        guard = PlacementGuard(env, "egg_0", rest_z=0.27)
        self.assertTrue(math.isinf(float(guard.speed_limit()[0])))
        egg.data.root_pos_w[:, 2] = 0.275
        egg.data.root_lin_vel_w[0, 2] = -0.04
        self.assertAlmostEqual(float(guard.speed_limit()[0]), 0.05, places=6)
        self.assertEqual(guard.ready_to_release().tolist(), [False, True])  # env 0 still moving

    def test_alignment_and_upright_checks(self):
        from robobench.core.prevention import is_aligned, is_upright

        s = math.sqrt(0.5)
        card = rigid(2, pos=(0.5, 0.0, 0.1), quat=(s, s, 0.0, 0.0))  # tipped 90° about x
        env = prev_env(rigid_objects={"card": card})
        ok, info = is_aligned(env, "card", (0.5, 0.0, 0.1), target_axis=(0, 0, 1))
        self.assertFalse(bool(ok[0]))
        self.assertAlmostEqual(info["ang_err_deg"][0], 90.0, places=2)
        card.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        ok, _ = is_aligned(env, "card", (0.501, 0.0, 0.1), pos_tol=0.003, target_axis=(0, 0, 1))
        self.assertTrue(bool(ok.all()))
        self.assertTrue(bool(is_upright(env, "card").all()))

    def test_verify_then_approve_never_asks_about_a_failed_check(self):
        from robobench.core.prevention import verify_then_approve
        from robobench.core.safety import install_safety

        env = prev_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        self.assertFalse(verify_then_approve(env, "slot", torch.tensor([True, False])))
        self.assertEqual(asked, [])
        self.assertTrue(verify_then_approve(env, "slot", torch.tensor([True, True])))
        self.assertEqual(asked, ["slot"])

    def test_insertion_monitor_reports_a_stall_instead_of_shoving(self):
        from robobench.core.prevention import InsertionMonitor

        card = rigid(2, pos=(0.5, 0.0, 0.10))
        env = prev_env(rigid_objects={"card": card})
        mon = InsertionMonitor(env, "card", axis=(0, 0, -1), seat_depth=0.01, window_s=0.05)
        d = env.robot.articulation.data
        d.applied_torque[0, 1] = 80.0  # env 0 pushes at 92% of the 87 N·m rating ...
        for k in range(10):
            card.data.root_pos_w[1, 2] = 0.10 - 0.002 * (k + 1)  # ... env 1 slides in
            substep(env)
        self.assertEqual(mon.status(), ["stalled", "seated"])
        mon.reset()
        self.assertEqual(mon.status()[0], "moving")

    def test_fixture_guard_scales_the_pull_down_before_the_limit(self):
        from robobench.core.prevention import FixtureGuard

        drawer = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.tensor([[0.1], [0.27]])))
        env = prev_env(articulations={"toolbox": drawer})
        scale = FixtureGuard(env, "toolbox", 0.3).scale()
        self.assertAlmostEqual(float(scale[0]), 1.0)
        self.assertAlmostEqual(float(scale[1]), 1 / 3, places=4)

    def test_aim_guard_blocks_a_nozzle_pointed_at_the_robot(self):
        from robobench.core.prevention import AimGuard

        syringe = rigid(2, pos=(0.4, 0.0, 0.6))  # nozzle axis local -z: straight down at the hand
        env = prev_env(rigid_objects={"syringe": syringe})
        guard = AimGuard(env, "syringe", ignore_radius=0.05)
        self.assertFalse(bool(guard.safe_to_dispense()[0]))
        syringe.data.root_pos_w[:, 0] = 0.9  # moved 0.5 m to the side
        self.assertTrue(bool(guard.safe_to_dispense().all()))

    def test_every_reviewed_task_has_a_prevention_plan_that_installs(self):
        from robobench.core.prevention import PREVENTION, setup_prevention
        from robobench.core.safety_rules import TASK_RULES

        self.assertEqual(set(PREVENTION), set(TASK_RULES))
        for scene in PREVENTION:
            prev = setup_prevention(prev_env(), scene)
            self.assertTrue(prev.steps, scene)


# ----- allen_bolt ----------------------------------------------------------------------------------
PLATE_TOP = 0.038  # plate top above the platform origin (the scene's default)


def allen_env(n=2, staged=False):
    """The allen_bolt layout: platform (hole axis at its origin) at the table origin, bolt lying
    on its side 14 cm away, key lying flat beyond it."""
    cfg = SimpleNamespace(surface_z=0.0, thread_len=0.0248, seat_depth=0.022, bolt_staged=staged,
                          bolt_stage_depth=0.004898, key_init_z=0.0075, bolt_init_z=0.013,
                          plate_top=PLATE_TOP, align_xy=0.004, num_pairs=1)
    key = rigid(n, pos=(0.26, 0.0, 0.0075))
    bolt = rigid(n, pos=(0.14, 0.0, 0.013))
    platform = rigid(n, pos=(0.0, 0.0, 0.0))
    env = prev_env(n, rigid_objects={"key_0": key, "bolt_0": bolt, "platform_0": platform})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, grasp_held=torch.zeros(n, 2, dtype=torch.bool))
    return env, key, bolt


def lift_with(env, part):
    """A step_fn that raises the hand and `part` together by 2 mm."""
    hand = env.robot.articulation.data.body_pos_w

    def lift():
        hand[:, 1, 2] += 0.002
        part.data.root_pos_w[:, 2] += 0.002
        substep(env)
    return lift


@unittest.skipIf(torch is None, "needs torch")
class AllenBoltTests(unittest.TestCase):
    def test_required_grip_force_physics(self):
        from robobench.suites.assembly.prevention.allen_bolt import required_grip_force

        # 2 × (0.08 kg × (9.81 + 2) m/s²) / (2 fingers × 0.3) = 3.15 N per finger
        self.assertAlmostEqual(required_grip_force(0.08, 0.3), 3.149, places=3)
        # the simulated bolt is slick (μ = 0.01): ~59 N
        self.assertAlmostEqual(required_grip_force(0.05, 0.01), 59.05, places=2)
        # turning the key against the pads (2 N·m at 10 cm = 20 N) makes the key need ~70 N
        self.assertAlmostEqual(required_grip_force(0.08, 0.3, tangential_load=20.0), 69.82, places=2)

    def test_grip_is_set_from_the_friction_model(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, _, _ = allen_env()
        sim = AllenBoltPrevention(env)
        self.assertAlmostEqual(sim.grip.force, 59.05, places=2)
        env2, _, _ = allen_env()
        real = AllenBoltPrevention(env2, friction_model="real")
        self.assertAlmostEqual(real.grip.force, 3.149, places=3)
        self.assertFalse(real.grip_report()["capped"])

    def test_lift_test_needs_the_weld_and_the_part_rising(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, key, _ = allen_env()
        prev = AllenBoltPrevention(env)
        hand = env.robot.articulation.data.body_pos_w

        def lift():
            hand[:, 1, 2] += 0.001
            key.data.root_pos_w[:, 2] += 0.001
            substep(env)
        self.assertFalse(prev.lift_test("key_0", lift, steps=5))  # rose, but no weld
        env.scene.grasp_held[:, 0] = True
        self.assertTrue(prev.lift_test("key_0", lift, steps=5))

    def test_fingers_cannot_open_while_the_key_is_in_the_air(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, key, _ = allen_env()
        env.scene.grasp_held[:, 0] = True
        prev = AllenBoltPrevention(env, friction_model="real")
        hand = env.robot.articulation.data.body_pos_w

        def lift():
            hand[:, 1, 2] += 0.002
            key.data.root_pos_w[:, 2] += 0.002
            substep(env)
        self.assertTrue(prev.lift_test("key_0", lift, steps=10))
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        action = d.joint_pos.clone()
        action[:, 7:9] = 0.04  # try to open wide mid-air
        substep(env, action)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))
        # set it down on the table and stop: now opening goes through
        key.data.root_pos_w[:, 2] = 0.0075
        substep(env, action)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))

    def test_key_must_be_down_in_the_socket_before_turning(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, key, bolt = allen_env(staged=True)
        bolt.data.root_pos_w[:] = torch.tensor([0.0, 0.0, 0.0331])  # standing in the hole
        prev = AllenBoltPrevention(env)
        mouth = prev.socket_pos(0)[0]
        self.assertAlmostEqual(float(mouth[2]), 0.0331 + 0.0248 + 0.016, places=4)
        key.data.root_pos_w[0] = mouth + torch.tensor([0.0, 0.0, -0.006])  # 6 mm in, on axis
        key.data.root_pos_w[1] = mouth + torch.tensor([0.003, 0.0, -0.006])  # 3 mm off axis
        self.assertEqual(prev.key_in_socket(0).tolist(), [True, False])

    def test_drive_measures_travel_from_the_bolts_actual_depth(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, _, bolt = allen_env()
        bolt.data.root_pos_w[:] = torch.tensor([0.0, 0.0, PLATE_TOP - 0.005])  # stood in, 5 mm deep
        prev = AllenBoltPrevention(env)
        drive = prev.drive(0)
        bolt.data.root_pos_w[:, 2] = PLATE_TOP - 0.020
        substep(env)
        self.assertEqual(drive.status(), ["moving", "moving"])
        bolt.data.root_pos_w[:, 2] = PLATE_TOP - 0.022  # the scene's seat depth, absolute
        substep(env)
        # regression: the old code wanted 22 mm of TRAVEL (27 mm absolute), past where the head
        # bottoms out (24.8 mm) — never "seated", and "stalled" once the arm pressed on the head
        self.assertEqual(drive.status(), ["seated", "seated"])

    # -- regressions found in review ----------------------------------------------------------
    def test_bolt_lying_beside_the_platform_is_not_in_the_hole(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, _, bolt = allen_env()
        prev = AllenBoltPrevention(env)
        # lying 14 cm away: its tip is 25 mm "below the plate top", but it is not in the hole
        self.assertEqual(prev.bolt_in_hole(0).tolist(), [False, False])
        bolt.data.root_pos_w[0] = torch.tensor([0.001, 0.0, PLATE_TOP - 0.004])  # on axis, 4 mm in
        self.assertEqual(prev.bolt_in_hole(0).tolist(), [True, False])
        self.assertAlmostEqual(float(prev.bolt_depth(0)[0]), 0.004, places=6)

    def test_carrying_the_bolt_gates_release_until_it_stands_in_the_hole(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, _, bolt = allen_env()
        prev = AllenBoltPrevention(env, friction_model="real")
        # regression: this used to raise (bool |= float) on the first substep while carrying
        self.assertTrue(prev.lift_test("bolt_0", lift_with(env, bolt), steps=10))
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        bolt.data.root_pos_w[:] = torch.tensor([0.0, 0.0, PLATE_TOP + 0.03])  # above the hole
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))
        bolt.data.root_pos_w[:] = torch.tensor([0.0, 0.0, PLATE_TOP - 0.004])  # stood in the thread
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))

    def test_opening_on_a_supported_part_ends_the_carry_automatically(self):
        from robobench.suites.assembly.prevention.allen_bolt import AllenBoltPrevention

        env, key, _ = allen_env()
        env.scene.grasp_held[:, 0] = True
        prev = AllenBoltPrevention(env, friction_model="real")
        self.assertTrue(prev.lift_test("key_0", lift_with(env, key), steps=10))
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.3, places=6)  # carry speed on
        key.data.root_pos_w[:, 2] = 0.0075  # set down on the table
        d = env.robot.articulation.data
        d.joint_pos[:, 7:9] += 0.005  # the fingers have opened 5 mm
        substep(env)
        self.assertEqual(prev._held_obj, [None, None])
        self.assertTrue(math.isinf(float(prev.speed.limit()[0])))  # carry speed off again


class PeriodicFranka:
    """A controller that writes only every `period` substeps, like a decimated OSC: in between,
    the actuator buffers keep the last command."""

    def __init__(self, art, period=2):
        self.articulation, self.name, self.period = art, "robot", period

    def apply_action(self, action, substep=0):
        if substep % self.period == 0:
            self.articulation.data.joint_effort_target[:] = action

    def post_step(self, env_ids=None):
        pass


@unittest.skipIf(torch is None, "needs torch")
class DecimatedControllerTests(unittest.TestCase):
    def test_speed_governor_brake_does_not_compound_between_controller_writes(self):
        from robobench.core.prevention import SpeedGovernor

        art = FakeArticulation()
        art.data.body_pos_w = torch.zeros(2, 2, 3)
        robot = PeriodicFranka(art)  # not named FrankaRobot: ratings fall back to sim limits
        env = make_env(robot=robot)
        env.device = "cpu"
        SpeedGovernor(env, ee_speed_max=0.5, damping=5.0)
        art.data.body_lin_vel_w[:, 1] = torch.tensor([1.0, 0.0, 0.0])  # hand at 2x the limit
        art.data.joint_vel[:, 0] = 1.0
        command = torch.full((2, 9), 10.0)
        efforts = []
        for k in range(4):  # the controller writes on substeps 0 and 2 only
            env.robot.apply_action(command, k)  # hooked: snapshot -> controller -> governor
            efforts.append(float(art.data.joint_effort_target[0, 0]))
            env.robot.post_step()
        # brake = 5 × (1 − 0.5) × 1.0 = 2.5 every substep — never 5.0, 7.5, ... on held substeps
        self.assertEqual(efforts, [7.5, 7.5, 7.5, 7.5])


if __name__ == "__main__":
    unittest.main()


# ----- bulb ---------------------------------------------------------------------------------------
def bulb_env(n=2):
    """The bulb layout: socket (axis up through its origin) at the table origin, bulb lying 13 cm
    away, the hand's angular velocity readable for the turn monitor."""
    cfg = SimpleNamespace(surface_z=0.0, seat_z=0.027, light_start_z=0.032, socket_opening_z=0.0385,
                          align_xy=0.015, align_axis_deg=12.0, bulb_glass_friction=0.3,
                          bulb_init_z=0.024, num_pairs=1)
    bulb = rigid(n, pos=(0.13, 0.0, 0.005))
    bulb.data.root_ang_vel_w = torch.zeros(n, 3)
    socket = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3),
                                                  root_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n)))
    env = prev_env(n, rigid_objects={"bulb_0": bulb}, articulations={"socket_0": socket})
    env.robot.articulation.data.body_ang_vel_w = torch.zeros(n, 2, 3)
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg)
    return env, bulb


def put_bulb(bulb, height, x=0.0):
    """Upright bulb with its origin (cap bottom) `height` above the socket origin."""
    bulb.data.root_pos_w[:] = torch.tensor([x, 0.0, height])
    bulb.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class BulbTests(unittest.TestCase):
    def test_forces_start_minimal_and_seating_fits_under_the_glass_cap(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention, required_twist_grip

        # turning 0.1 N·m through μ 0.3 pads on a 24 mm radius, 1.5x margin: 10.4 N per finger
        self.assertAlmostEqual(required_twist_grip(0.1, 0.3, 0.024), 10.417, places=3)
        env, _ = bulb_env()
        prev = BulbPrevention(env)
        rep = prev.grip_report()
        self.assertAlmostEqual(rep["set_N"], 1.97, places=2)  # just the bulb's weight, 2x margin
        self.assertTrue(rep["can_seat_under_cap"])
        self.assertEqual(rep["glass_cap_N"], 15.0)

    def test_grip_rises_only_on_slip_and_never_past_the_glass_cap(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.031)  # on the thread
        prev.mark_held("bulb_0")  # in the fingers
        screw = prev.screw(0, slip_time=0.01)  # 1 substep of slip per boost
        hand = env.robot.articulation.data.body_ang_vel_w
        hand[:, 1, 2] = 1.0  # the hand turns at 1 rad/s ...
        bulb.data.root_ang_vel_w[:, 2] = 1.0  # ... and the bulb turns with it: no slip
        substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)
        bulb.data.root_ang_vel_w[:, 2] = 0.0  # now the bulb stops: slipping in the fingers
        forces = []
        for _ in range(20):
            substep(env)
            forces.append(prev.grip.force)
        self.assertTrue(all(b >= a for a, b in zip(forces, forces[1:])))  # only ever up ...
        self.assertLessEqual(max(forces), 15.0)  # ... and never past the cap
        self.assertEqual(screw.status(), ["slipping_at_cap"] * 2)
        self.assertEqual(screw.should_stop(), [True, True])

    def test_seated_means_stop_and_no_more_squeezing(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.030)
        screw = prev.screw(0, slip_time=0.01)
        self.assertEqual(screw.status(), ["turning"] * 2)
        put_bulb(bulb, 0.0265)  # home
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0  # hand keeps turning, bulb can't
        for _ in range(5):
            substep(env)
        self.assertEqual(screw.status(), ["seated"] * 2)
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)  # a seated bulb is not "slip"

    def test_a_crooked_bulb_is_not_pressed_in(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.036)
        a = math.radians(10) / 2
        bulb.data.root_quat_w[1] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])  # 10° tilt
        ok, info = prev.aligned(0)
        self.assertEqual(ok.tolist(), [True, False])
        self.assertAlmostEqual(info["tilt_deg"][1], 10.0, places=3)

    def test_binding_is_reported_instead_of_pressing_harder(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.031)
        screw = prev.screw(0, bind_window_s=0.05)
        env.robot.articulation.data.applied_torque[0, 1] = 50.0  # env 0: 57% of 87 N·m, no progress
        for _ in range(8):
            substep(env)
        self.assertEqual(screw.status(), ["binding", "turning"])

    def test_slows_near_the_socket_and_only_lets_go_once_threaded(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        self.assertTrue(prev.lift_test("bulb_0", lift_with(env, bulb), steps=10))
        put_bulb(bulb, 0.20)  # carried high over the socket
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.25, places=6)  # carry speed
        put_bulb(bulb, 0.05)  # just above the bore mouth (38.5 mm)
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.03, places=6)  # approach speed
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        put_bulb(bulb, 0.036)  # resting on the mouth, not yet on the thread
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))  # gate holds
        put_bulb(bulb, 0.031)  # on the thread: it holds the bulb now
        prev.grip.force = 8.0  # (as if the grip had been raised while turning)
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))
        d.joint_pos[:, 7:9] += 0.005  # fingers open: a re-grip, part-way into the thread
        substep(env)
        self.assertFalse(bool(prev.carrying().any()))
        self.assertAlmostEqual(prev.grip.force, 8.0)  # kept: the bulb needed it to turn
        put_bulb(bulb, 0.026)  # seated
        prev.mark_held("bulb_0")
        d.joint_pos[:, 7:9] += 0.005
        substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)  # done: the next grasp starts minimal


# ----- ikea_table ---------------------------------------------------------------------------------
SLAB_TOP = 0.05  # fake: bench surface at z=0, 5 cm slab, tabletop origin on the slab top
STUDS = ((0.25, 0.25), (-0.25, 0.25), (0.25, -0.25), (-0.25, -0.25))
TIP = 0.026  # cfg.leg_start_z: a leg resting on a stud tip


def ikea_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, slots=STUDS, num_legs=4, leg_start_z=TIP, seat_z=0.012,
                          align_xy=0.02, align_axis_deg=10.0, leg_init_z=0.022)
    table = rigid(n, pos=(0.0, 0.0, SLAB_TOP))
    s = math.sqrt(0.5)  # legs start lying on their side (90° about x), as in the scene
    legs = {f"leg_{k}": rigid(n, pos=(0.3 + 0.12 * k, -0.4, 0.022), quat=(s, s, 0.0, 0.0)) for k in range(4)}
    for leg in legs.values():
        leg.data.root_ang_vel_w = torch.zeros(n, 3)
    env = prev_env(n, rigid_objects={"tabletop": table, **legs})
    env.robot.articulation.data.body_ang_vel_w = torch.zeros(n, 2, 3)
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, welded=torch.zeros(n, 4, dtype=torch.bool))
    return env, table, legs


def put_leg(leg, stud=0, height=TIP, dx=0.0, tilt_deg=0.0):
    """Leg upright over STUDS[stud] (table at the origin, unrotated), nut origin `height` above
    the slab, shifted `dx` in x and tilted `tilt_deg` about x."""
    x, y = STUDS[stud]
    leg.data.root_pos_w[:] = torch.tensor([x + dx, y, SLAB_TOP + height])
    a = math.radians(tilt_deg) / 2
    leg.data.root_quat_w[:] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class IkeaTableTests(unittest.TestCase):
    def test_check_passes_only_a_leg_centred_upright_on_a_free_stud_tip(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=2)
        ok, info = prev.check_leg(0)
        self.assertEqual(ok.tolist(), [True, True])
        self.assertEqual(info["stud"], [2, 2])
        put_leg(legs["leg_0"], stud=2, dx=0.006)  # 6 mm off: inside the scene's 20 mm, not ours
        ok, info = prev.check_leg(0)
        self.assertEqual(ok.tolist(), [False, False])
        self.assertIn("6.0 mm off the stud axis", info["problems"][0][0])
        put_leg(legs["leg_0"], stud=2, tilt_deg=5.0)
        ok, info = prev.check_leg(0)
        self.assertFalse(bool(ok.any()))
        self.assertIn("tilted 5.0 deg", info["problems"][0][0])
        put_leg(legs["leg_0"], stud=2, height=TIP + 0.02)  # hovering, not resting on the tip
        self.assertFalse(bool(prev.check_leg(0)[0].any()))

    def test_a_stud_another_leg_is_on_is_the_wrong_place(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_1"], stud=0, height=0.015)  # leg 1 threaded on stud 0
        put_leg(legs["leg_0"], stud=0)  # leg 0 brought to the same stud
        ok, info = prev.check_leg(0)
        self.assertEqual(ok.tolist(), [False, False])
        self.assertIn("stud 0 is already taken", info["problems"][0])
        self.assertEqual(prev.free_studs(for_leg=0)[0].tolist(), [False, True, True, True])
        put_leg(legs["leg_0"], stud=1)  # a free one
        self.assertTrue(bool(prev.check_leg(0)[0].all()))

    def test_a_welded_leg_holds_its_stud_even_off_the_thread_test(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_3"], stud=3, height=0.04)  # nearest stud 3, but above the thread ...
        env.scene.welded[0, 3] = True  # ... yet the scene welded it there in env 0
        self.assertEqual(prev.occupied()[:, 3].tolist(), [True, False])

    def test_the_operator_is_asked_only_about_a_passing_check_and_the_stud_is_recorded(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.ikea_table import APPROVAL, IkeaTablePrevention

        env, _, legs = ikea_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append((n, i["stud"])) or True)
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=1, dx=0.01)
        self.assertFalse(prev.verify_leg(0))
        self.assertEqual(asked, [])
        self.assertEqual(prev.approved[:, 0].tolist(), [-1, -1])
        put_leg(legs["leg_0"], stud=1)
        self.assertTrue(prev.verify_leg(0))
        self.assertEqual(asked, [(APPROVAL, [1, 1])])
        self.assertEqual(prev.approved[:, 0].tolist(), [1, 1])

    def test_an_operator_refusal_leaves_the_leg_unverified(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        install_safety(env, approver=lambda n, i: False)
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=0)
        self.assertFalse(prev.verify_leg(0))
        self.assertEqual(prev.drive(0).status(), ["unverified"] * 2)

    def test_drive_stops_off_the_approved_stud_and_reports_seated(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        leg = legs["leg_0"]
        put_leg(leg, stud=0)
        drive = prev.drive(0)
        self.assertEqual(drive.status(), ["unverified"] * 2)  # never verified: don't drive
        self.assertTrue(prev.verify_leg(0))
        drive = prev.drive(0)
        self.assertEqual(drive.status(), ["driving"] * 2)
        leg.data.root_pos_w[1, 0] += 0.03  # env 1's leg skids 3 mm past the scene tolerance
        self.assertEqual(drive.status(), ["driving", "off_target"])
        self.assertEqual(drive.should_stop(), [False, True])
        put_leg(leg, stud=0, height=0.011)  # threaded home on the approved stud
        self.assertEqual(drive.status(), ["seated"] * 2)
        prev.clear_approval(0)
        put_leg(leg, stud=0, height=0.02)
        self.assertEqual(drive.status(), ["unverified"] * 2)

    def test_drive_stops_when_the_table_is_shoved_or_the_leg_binds(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, table, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=0)
        self.assertTrue(prev.verify_leg(0))
        drive = prev.drive(0, window_s=0.05)
        env.robot.articulation.data.applied_torque[1, 1] = 60.0  # env 1: 69% of 87 N·m, no progress
        for _ in range(8):
            substep(env)
        table.data.root_pos_w[0, 0] += 0.008  # env 0: the tabletop slid 8 mm
        self.assertEqual(drive.status(), ["table_moving", "stalled"])
        table.data.root_pos_w[0, 0] -= 0.008
        a = math.radians(3.0) / 2  # ... or tipped 3°
        table.data.root_quat_w[0] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])
        self.assertTrue(bool(drive.table_moved()[0]))

    def test_studs_follow_a_moved_and_turned_table(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, table, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        s = math.sqrt(0.5)
        table.data.root_pos_w[:] = torch.tensor([-0.45, 0.0, SLAB_TOP])
        table.data.root_quat_w[:] = torch.tensor([s, 0.0, 0.0, s])  # yawed 90°
        tips = prev.stud_positions()
        # stud 0 at (0.25, 0.25) in the slab frame -> (-0.25, 0.25) after the yaw, + the offset
        self.assertTrue(torch.allclose(tips[0, 0], torch.tensor([-0.70, 0.25, SLAB_TOP + TIP]), atol=1e-6))
        legs["leg_0"].data.root_pos_w[:] = tips[:, 0]
        legs["leg_0"].data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        ok, info = prev.check_leg(0)
        self.assertEqual(ok.tolist(), [True, True])
        self.assertEqual(info["stud"], [0, 0])

    def test_carried_leg_slows_over_a_stud_and_is_let_go_only_once_threaded(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        leg = legs["leg_0"]
        prev = IkeaTablePrevention(env)
        self.assertTrue(prev.lift_test("leg_0", lift_with(env, leg), steps=10))
        put_leg(leg, stud=0, height=0.20)  # carried high over the table
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.3, places=6)  # carry speed
        put_leg(leg, stud=0, height=TIP + 0.02)  # just above the stud tip
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.03, places=6)  # approach speed
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        put_leg(leg, stud=0)  # resting on the tip, not yet threaded
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))  # gate holds
        put_leg(leg, stud=0, height=TIP - 0.004)  # 4 mm onto the thread
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))

    def test_grip_comes_from_the_leg_weight(self):
        from robobench.core.prevention import required_grip_force
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, _ = ikea_env()
        prev = IkeaTablePrevention(env)
        self.assertAlmostEqual(prev.grip.force, required_grip_force(0.12, 0.4), places=6)
        self.assertAlmostEqual(prev.grip_report()["required_N_to_hold_leg"], 3.54, places=2)

    # -- regressions from the bug review ----------------------------------------------------------
    def test_a_table_twisted_round_its_middle_counts_as_moving(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, table, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=0)
        self.assertTrue(prev.verify_leg(0))
        drive = prev.drive(0)
        a = math.radians(2.0) / 2  # the turning leg yaws the slab 2° about its centre:
        table.data.root_quat_w[0] = torch.tensor([math.cos(a), 0.0, 0.0, math.sin(a)])
        # the centre has not moved and the slab is level, but the studs swung ~12 mm
        self.assertEqual(drive.table_moved().tolist(), [True, False])

    def test_slow_threading_under_load_is_not_a_stall(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        env.dt = 0.01
        prev = IkeaTablePrevention(env)
        leg = legs["leg_0"]
        put_leg(leg, stud=0)
        self.assertTrue(prev.verify_leg(0))
        drive = prev.drive(0)
        env.robot.articulation.data.applied_torque[:, 1] = 60.0  # pushing hard in both envs ...
        for k in range(150):  # ... env 0 threads down at 0.15 mm/s (a slow ~0.5 rad/s turn); env 1 doesn't move
            leg.data.root_pos_w[0, 2] -= 0.15e-3 * env.dt
            substep(env)
        self.assertEqual(drive.status(), ["driving", "stalled"])

    def test_a_leg_turning_in_the_fingers_raises_the_grip_up_to_the_cap(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        leg = legs["leg_0"]
        put_leg(leg, stud=0)
        self.assertTrue(prev.verify_leg(0))
        drive = prev.drive(0, slip_time=0.01)
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0  # the hand turns, the leg does not
        forces = []
        for _ in range(30):
            substep(env)
            forces.append(prev.grip.force)
        self.assertGreater(forces[0], prev.needed)
        self.assertTrue(all(b >= a for a, b in zip(forces, forces[1:])))
        self.assertLessEqual(max(forces), 40.0)
        self.assertEqual(drive.status(), ["slipping_at_cap"] * 2)

    def test_a_finished_drive_stops_raising_the_grip(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=0)
        self.assertTrue(prev.verify_leg(0))
        prev.drive(0, slip_time=0.01)
        put_leg(legs["leg_1"], stud=1)
        self.assertTrue(prev.verify_leg(1))
        prev.drive(1, slip_time=0.01)  # closes leg 0's drive
        legs["leg_1"].data.root_ang_vel_w[:, 2] = 1.0  # leg 1 turns with the hand: no slip ...
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0  # ... leg 0 (still) would "slip"
        for _ in range(5):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.needed)

    def test_a_leg_standing_on_the_slab_off_a_stud_is_not_let_go(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        leg = legs["leg_0"]
        leg.data.root_pos_w[:] = torch.tensor([0.0, 0.0, SLAB_TOP])  # upright on its nut, mid-slab
        leg.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertFalse(bool(prev.supported("leg_0").any()))
        s = math.sqrt(0.5)
        leg.data.root_pos_w[:] = torch.tensor([0.0, 0.0, SLAB_TOP + 0.022])  # laid down on the slab
        leg.data.root_quat_w[:] = torch.tensor([s, s, 0.0, 0.0])
        self.assertTrue(bool(prev.supported("leg_0").all()))

    def test_two_arms_only_the_hand_holding_the_leg_is_gated(self):
        from test_safety import FakeMulti
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        n = 2
        left, right = FakeArticulation(n), FakeArticulation(n)
        for art, x in ((left, -0.9), (right, 0.3)):
            art.data.body_pos_w = torch.zeros(n, 2, 3)
            art.data.body_pos_w[:, 1] = torch.tensor([x, -0.4, 0.05])
            art.data.body_ang_vel_w = torch.zeros(n, 2, 3)
        env, table, legs = ikea_env(n)
        env.robot = FakeMulti(FrankaRobot(left, "franka"), FrankaRobot(right, "franka"))
        leg = legs["leg_0"]  # at x = 0.3: under the RIGHT hand
        prev = IkeaTablePrevention(env, grip_arms=("right",))

        def step2(action):  # both arms get the same command, as FakeMulti does
            env.robot.apply_action(action, 0)
            env.robot.post_step()

        def lift():
            right.data.body_pos_w[:, 1, 2] += 0.002
            leg.data.root_pos_w[:, 2] += 0.002
            step2(right.data.joint_pos_target.clone())
        self.assertTrue(prev.lift_test("leg_0", lift, steps=10))
        leg.data.root_pos_w[:, 2] = 0.20  # carried in the air
        opening = right.data.joint_pos.clone()
        opening[:, 7:9] = 0.04
        right_closed = right.data.joint_pos[:, 7:9].clone()
        step2(opening)  # both hands are told to open
        self.assertTrue(torch.allclose(right.data.joint_pos_target[:, 7:9], right_closed))  # leg hand held
        self.assertTrue(torch.allclose(left.data.joint_pos_target[:, 7:9], torch.full((n, 2), 0.04)))  # free hand opens
        left.data.joint_pos[:, 7:9] += 0.01  # the free hand opening ...
        step2(opening)
        self.assertTrue(bool(prev.carrying().all()))  # ... does not end the carry
        # the leg grip limit shapes only the right hand: the left (table) hand is not capped
        self.assertEqual(len(prev.grip._arms), 1)
        self.assertIs(prev.grip._arms[0][0], right)


# ----- so101 --------------------------------------------------------------------------------------
UA = (0.0, 0.10, 0.12)  # fake: where the upper_arm link (the servo pocket) is, in the world
SEAT_POS, SEAT_QUAT = (-0.11257, -0.028, 0.0), (0.7071068, 0.0, 0.0, 0.7071068)


def art(n, pos, quat=(1.0, 0.0, 0.0, 0.0), bodies=None, masses=None):
    """A fake floating articulation: root state, optional per-body poses and physx masses."""
    d = SimpleNamespace(root_pos_w=torch.tensor([pos] * n, dtype=torch.float32),
                        root_quat_w=torch.tensor([quat] * n, dtype=torch.float32),
                        root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3))
    if bodies is not None:
        d.body_pos_w = torch.tensor([bodies] * n, dtype=torch.float32)
        d.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * len(bodies)] * n)
    a = SimpleNamespace(data=d)
    if masses is not None:
        a.root_physx_view = SimpleNamespace(get_masses=lambda: torch.tensor([masses] * n))
    return a


def so101_env(n=2, distal_masses=None):
    cfg = SimpleNamespace(surface_z=0.0, detent_pos=0.0035, detent_deg=5.0, seat_detent=True,
                          fork_snap=True, fork_snap_pos=0.006, fork_snap_deg=8.0,
                          elbow_lower_arm_seat_pos=SEAT_POS, elbow_lower_arm_seat_quat=SEAT_QUAT,
                          motor_spawn_z=0.012, distal_spawn_z=0.05, num_elbow_holes=4)
    motor = rigid(n, pos=(0.26, -0.41, 0.012))
    motor.data.root_ang_vel_w = torch.zeros(n, 3)
    proximal = art(n, (0.0, 0.0, 0.0), bodies=[(0.0, 0.0, 0.0), UA])
    # forearm: root (lower_arm) + wrist + gripper links along +x
    distal = art(n, (0.72, 0.14, 0.05), masses=distal_masses,
                 bodies=[(0.72, 0.14, 0.05), (0.82, 0.14, 0.05), (0.88, 0.14, 0.05)])
    env = prev_env(n, rigid_objects={"motor": motor},
                   articulations={"proximal": proximal, "distal": distal})
    d = env.robot.articulation.data
    d.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * n)
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, b_ua=1)
    return env, motor, distal, proximal


def seat_motor(motor, dx=0.0):
    motor.data.root_pos_w[:] = torch.tensor([UA[0] + dx, UA[1], UA[2]])
    motor.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])


def attach_distal(distal, motor):
    """Put the forearm on the servo's horn (motor at identity orientation)."""
    shift = motor.data.root_pos_w + torch.tensor(SEAT_POS) - distal.data.root_pos_w
    distal.data.root_pos_w += shift
    distal.data.body_pos_w += shift[:, None]
    distal.data.root_quat_w[:] = torch.tensor(SEAT_QUAT)


def hand_at(env, pos):
    env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor(pos)


def lift_link(env, distal, link):
    """A step_fn: raise the hand and the forearm together by 2 mm (all links)."""
    hand = env.robot.articulation.data.body_pos_w

    def lift():
        hand[:, 1, 2] += 0.002
        distal.data.root_pos_w[:, 2] += 0.002
        distal.data.body_pos_w[:, :, 2] += 0.002
        substep(env)
    return lift


@unittest.skipIf(torch is None, "needs torch")
class SO101Tests(unittest.TestCase):
    def test_grip_comes_from_each_parts_weight(self):
        from robobench.core.prevention import required_grip_force
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, *_ = so101_env(distal_masses=[0.05, 0.08, 0.06, 0.06, 0.03])  # 0.28 kg over 5 links
        prev = SO101Prevention(env)
        self.assertAlmostEqual(prev.mass("distal"), 0.28, places=5)  # read from the simulator
        self.assertAlmostEqual(prev.mass("proximal"), 0.40)  # no physics view: the fallback
        self.assertAlmostEqual(prev.needed_force("motor"), required_grip_force(0.061, 0.35), places=6)
        self.assertAlmostEqual(prev.grip.force, prev.needed_force("motor"), places=6)  # starts light
        rep = prev.grip_report()["parts"]
        self.assertAlmostEqual(rep["distal"]["needed_N"], 9.45, places=2)
        self.assertFalse(any(p["over_cap"] for p in rep.values()))

    def test_the_whole_robot_needs_the_grip_for_everything_held_to_it(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, _ = so101_env()
        prev = SO101Prevention(env)
        alone = prev.needed_force("proximal")
        seat_motor(motor)
        attach_distal(distal, motor)
        self.assertTrue(bool(prev.motor_in_pocket().all()))
        self.assertTrue(bool(prev.distal_on_horn().all()))
        self.assertAlmostEqual(prev.load_mass("proximal"), 0.40 + 0.061 + 0.30, places=6)
        self.assertGreater(prev.needed_force("proximal"), alone * 1.8)

    def test_a_hold_that_cannot_be_firm_is_refused_before_lifting(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, proximal = so101_env()
        prev = SO101Prevention(env, grip_force_max=12.0)
        seat_motor(motor)
        attach_distal(distal, motor)  # the assembled robot: 0.76 kg, ~25.6 N per finger needed
        calls = []
        self.assertFalse(prev.lift_test("proximal", lambda: calls.append(1)))
        self.assertEqual(calls, [])  # never lifted
        self.assertIn("over the 12.0 N cap", prev.refusal)
        with self.assertRaises(RuntimeError):
            prev.mark_held("proximal")
        self.assertFalse(bool(prev.carrying().any()))

    def test_a_second_grasp_is_refused_while_the_first_part_is_still_in_the_air(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, _ = so101_env()
        prev = SO101Prevention(env)
        self.assertTrue(prev.lift_test("motor", lift_with(env, motor), steps=10))
        motor.data.root_pos_w[:, 2] = 0.20  # carried
        self.assertFalse(prev.lift_test("distal", lambda: None))
        self.assertIn("still carrying motor", prev.refusal)
        self.assertEqual(prev._held_obj, ["motor", "motor"])  # the servo keeps its protection
        motor.data.root_pos_w[:, 2] = 0.012  # put down on the bench
        hand_at(env, (0.72, 0.14, 0.06))
        self.assertTrue(prev.lift_test("distal", lift_link(env, distal, 0), steps=10))

    def test_lift_test_sets_the_grip_for_the_part(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, _, distal, _ = so101_env()
        prev = SO101Prevention(env)
        hand_at(env, (0.72, 0.14, 0.06))
        self.assertTrue(prev.lift_test("distal", lift_link(env, distal, 0), steps=10))
        self.assertAlmostEqual(prev.grip.force, prev.needed_force("distal"), places=6)

    def test_the_link_in_the_fingers_is_watched_not_the_root(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, _, distal, _ = so101_env()
        prev = SO101Prevention(env)
        hand_at(env, (0.88, 0.14, 0.06))  # holding the forearm by its gripper end (link 2)
        self.assertTrue(prev.lift_test("distal", lift_link(env, distal, 2), steps=10))
        f0 = prev.grip.force
        distal.data.root_pos_w[:, 2] -= 0.008  # the free end sags 8 mm on its wrist joints ...
        distal.data.body_pos_w[:, 0, 2] -= 0.008
        substep(env)
        self.assertAlmostEqual(prev.grip.force, f0)  # ... not slip: the gripped link did not move
        distal.data.body_pos_w[:, 2, 2] -= 0.008  # now the gripped link slides in the fingers
        substep(env)
        self.assertAlmostEqual(prev.grip.force, f0 * 1.25, places=5)

    def test_a_forearm_pivoting_in_the_fingers_tightens_the_grip(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, _, distal, _ = so101_env()
        prev = SO101Prevention(env)
        hand_at(env, (0.72, 0.14, 0.06))
        self.assertTrue(prev.lift_test("distal", lift_link(env, distal, 0), steps=10))
        f0 = prev.grip.force
        for deg, expect in ((3.0, f0), (8.0, f0 * 1.25)):  # 3° is within slip_deg; 8° is swinging out
            a = math.radians(deg) / 2
            distal.data.body_quat_w[:, 0] = torch.tensor([math.cos(a), 0.0, math.sin(a), 0.0])
            substep(env)
            self.assertAlmostEqual(prev.grip.force, expect, places=5)

    def test_near_the_pocket_is_not_held_by_the_pocket(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, _, _ = so101_env()
        prev = SO101Prevention(env)
        seat_motor(motor, dx=0.0038)  # inside the screw gate's 4 mm, outside the 3.5 mm click-in
        self.assertFalse(bool(prev.motor_in_pocket().any()))
        seat_motor(motor, dx=0.002)
        self.assertTrue(bool(prev.motor_in_pocket().all()))
        env.scene.cfg.seat_detent = False  # no click-in configured: only screws hold it
        self.assertFalse(bool(prev.motor_in_pocket().any()))

    def test_the_scenes_own_state_decides_when_it_exposes_it(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, _ = so101_env()
        prev = SO101Prevention(env)
        seat_motor(motor)  # right on the seat, but ...
        env.scene._detent_on = [False, True]  # ... only env 1 actually clicked in
        env.scene._fork_snap_on = [False, False]
        env.scene.fastened = torch.tensor([[-1] * 12, [-1] * 12])
        self.assertEqual(prev.motor_in_pocket().tolist(), [False, True])
        env.scene.fastened[0, 2] = 1  # env 0: a tab screw driven into hole 1
        self.assertEqual(prev.motor_in_pocket().tolist(), [True, True])
        self.assertEqual(prev.distal_on_horn().tolist(), [False, False])
        env.scene.fastened[1, 5] = 6  # env 1: an M3 into horn hole 6 (>= the 4 tab holes)
        self.assertEqual(prev.distal_on_horn().tolist(), [False, True])

    def test_the_servo_is_let_go_only_on_the_bench_or_clicked_into_its_pocket(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, _, _ = so101_env()
        prev = SO101Prevention(env)
        self.assertTrue(prev.lift_test("motor", lift_with(env, motor), steps=10))
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        seat_motor(motor, dx=0.02)  # 2 cm short of the pocket
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))  # gate holds
        seat_motor(motor)
        motor.data.root_ang_vel_w[1, 0] = 1.0  # env 1: still rocking in the pocket
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[0, 7:9] > closed[0]).all()))
        self.assertTrue(torch.allclose(d.joint_pos_target[1, 7:9], closed[1]))

    def test_the_forearm_may_be_let_go_once_it_is_on_the_horn(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, _ = so101_env()
        prev = SO101Prevention(env)
        seat_motor(motor)
        distal.data.root_pos_w[:] = torch.tensor([0.3, 0.0, 0.25])  # in the air
        self.assertFalse(bool(prev.supported("distal").any()))
        attach_distal(distal, motor)
        self.assertTrue(bool(prev.supported("distal").all()))
        distal.data.root_pos_w[0, 0] += 0.01  # env 0: 1 cm off the horn, outside the snap capture
        self.assertEqual(prev.supported("distal").tolist(), [False, True])
        distal.data.root_pos_w[:] = torch.tensor([0.72, 0.14, 0.06])  # laid on the bench
        self.assertTrue(bool(prev.supported("distal").all()))

    # -- regressions from the second bug review ---------------------------------------------------
    def test_a_forearm_on_a_loose_servo_is_not_let_go(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, _ = so101_env()
        prev = SO101Prevention(env)
        # the servo held up in the other hand (gripped for 61 g), not in its pocket ...
        motor.data.root_pos_w[:] = torch.tensor([0.40, 0.0, 0.15])
        attach_distal(distal, motor)  # ... and the forearm snapped onto it: let go, it hangs from that grip
        self.assertTrue(bool(prev.distal_on_horn().all()))
        self.assertFalse(bool(prev.supported("distal").any()))  # the pair would tip over
        seat_motor(motor)
        attach_distal(distal, motor)
        self.assertTrue(bool(prev.supported("distal").all()))

    def test_two_hands_share_a_load_one_hand_cannot_hold(self):
        from test_safety import FakeMulti
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        n = 2
        left, right = FakeArticulation(n), FakeArticulation(n)
        for a_, y in ((left, 0.05), (right, -0.05)):
            a_.data.body_pos_w = torch.zeros(n, 2, 3)
            a_.data.body_pos_w[:, 1] = torch.tensor([0.0, y, 0.02])
            a_.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * n)
        env, motor, distal, proximal = so101_env(n)
        env.robot = FakeMulti(FrankaRobot(left, "franka"), FrankaRobot(right, "franka"))
        seat_motor(motor)
        attach_distal(distal, motor)  # assembled: 0.76 kg -> ~25.6 N/finger alone, ~12.8 with two hands
        prev = SO101Prevention(env, grip_force_max=20.0)
        self.assertFalse(prev.lift_test("proximal", lambda: None))
        self.assertIn("use both hands (hands=2)", prev.refusal)

        def step2(action):
            env.robot.apply_action(action, 0)
            env.robot.post_step()

        def lift():
            for a_ in (left, right):
                a_.data.body_pos_w[:, 1, 2] += 0.002
            for part in (proximal, motor, distal):
                part.data.root_pos_w[:, 2] += 0.002
                if hasattr(part.data, "body_pos_w"):
                    part.data.body_pos_w[:, :, 2] += 0.002
            step2(left.data.joint_pos_target.clone())
        self.assertTrue(prev.lift_test("proximal", lift, steps=10, hands=2))
        self.assertAlmostEqual(prev.grip.force, prev.needed_force("proximal", hands=2), places=5)
        self.assertLess(prev.grip.force, 20.0)
        proximal.data.root_pos_w[:, 2] = 0.30  # the robot lifted clear of the bench
        opening = left.data.joint_pos.clone()
        opening[:, 7:9] = 0.04
        lc, rc = left.data.joint_pos[:, 7:9].clone(), right.data.joint_pos[:, 7:9].clone()
        step2(opening)
        self.assertTrue(torch.allclose(left.data.joint_pos_target[:, 7:9], lc))  # BOTH stay closed
        self.assertTrue(torch.allclose(right.data.joint_pos_target[:, 7:9], rc))

    def test_switching_parts_starts_from_the_new_parts_grip(self):
        from robobench.suites.assembly.prevention.so101 import SO101Prevention

        env, motor, distal, _ = so101_env()
        prev = SO101Prevention(env)
        hand_at(env, (0.72, 0.14, 0.06))
        self.assertTrue(prev.lift_test("distal", lift_link(env, distal, 0), steps=10))
        distal.data.root_pos_w[:, 2] = 0.05  # put down on the bench (fingers not yet opened)
        distal.data.body_pos_w[:, :, 2] = 0.05
        hand_at(env, (0.26, -0.41, 0.02))
        self.assertTrue(prev.lift_test("motor", lift_with(env, motor), steps=10))
        self.assertAlmostEqual(prev.grip.force, prev.needed_force("motor"), places=6)  # 1.7 N, not 9.5


# ----- nut_thread ---------------------------------------------------------------------------------
BOLT = (0.5, 0.0, 0.0)  # fake: bolt origin on the table top (z = 0)


def nut_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, num_pairs=1, nut_mass=0.03, nut_friction=0.4, bolt_height=0.025,
                          seat_z=0.012, align_xy=0.015, align_axis_deg=10.0, nut_init_z=0.02)
    nut = rigid(n, pos=(0.38, 0.0, 0.02))
    nut.data.root_ang_vel_w = torch.zeros(n, 3)
    bolt = art(n, BOLT)
    env = prev_env(n, rigid_objects={"nut_0": nut}, articulations={"bolt_0": bolt})
    env.robot.articulation.data.body_ang_vel_w = torch.zeros(n, 2, 3)
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg)
    return env, nut, bolt


def put_nut(nut, height, dx=0.0, tilt_deg=0.0):
    nut.data.root_pos_w[:] = torch.tensor([BOLT[0] + dx, BOLT[1], BOLT[2] + height])
    a = math.radians(tilt_deg) / 2
    nut.data.root_quat_w[:] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class NutThreadTests(unittest.TestCase):
    def test_check_passes_only_a_nut_centred_and_upright_on_the_bolt_top(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        ok, _ = prev.check_nut(0)
        self.assertEqual(ok.tolist(), [True, True])
        put_nut(nut, 0.025, dx=0.003)  # 3 mm off: inside the scene's 15 mm, not ours
        ok, info = prev.check_nut(0)
        self.assertFalse(bool(ok.any()))
        self.assertIn("3.0 mm off the bolt axis", info["problems"][0][0])
        put_nut(nut, 0.025, tilt_deg=6.0)
        ok, info = prev.check_nut(0)
        self.assertIn("tilted 6.0 deg", info["problems"][0][0])
        put_nut(nut, 0.040)  # hovering above the bolt
        self.assertFalse(bool(prev.check_nut(0)[0].any()))

    def test_drive_refuses_until_checked_and_stops_at_the_seat(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025, dx=0.003)
        prev.check_nut(0)  # fails
        drive = prev.drive(0)
        self.assertEqual(drive.status(), ["unchecked"] * 2)
        put_nut(nut, 0.025)
        self.assertTrue(bool(prev.check_nut(0)[0].all()))
        drive = prev.drive(0)
        self.assertEqual(drive.status(), ["driving"] * 2)
        for k in range(14):  # screws down 1 mm per substep (a turn every 2 substeps)
            put_nut(nut, 0.025 - 0.001 * (k + 1))
            substep(env)
        self.assertEqual(drive.status(), ["seated"] * 2)
        self.assertEqual(drive.should_stop(), [True, True])

    def test_turning_the_wrong_way_is_caught(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        prev.check_nut(0)
        drive = prev.drive(0)
        put_nut(nut, 0.020)  # 2.5 turns down ...
        substep(env)
        nut.data.root_pos_w[1, 2] += 0.0015  # ... env 1 now backs it off 1.5 mm (unscrewing)
        substep(env)
        self.assertEqual(drive.status(), ["driving", "wrong_direction"])

    def test_a_tipped_nut_and_a_cross_threaded_nut_both_stop_the_drive(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        prev.check_nut(0)
        drive = prev.drive(0, window_s=0.05)
        env.robot.articulation.data.applied_torque[1, 1] = 50.0  # env 1: 57% of 87 N·m, no progress
        for _ in range(8):
            substep(env)
        a = math.radians(12.0) / 2  # env 0 tips 12° on the thread start
        nut.data.root_quat_w[0] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])
        self.assertEqual(drive.status(), ["off_axis", "cross_threaded"])

    def test_a_nut_spinning_in_the_fingers_raises_the_grip_only_up_to_the_cap(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        self.assertAlmostEqual(prev.grip.force, prev.needed)  # starts at what holds the 30 g nut
        put_nut(nut, 0.025)
        prev.mark_held("nut_0")  # the nut is in the fingers
        prev.check_nut(0)
        drive = prev.drive(0, slip_time=0.01)
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0  # hand turns, nut does not
        forces = []
        for _ in range(30):
            substep(env)
            forces.append(prev.grip.force)
        self.assertTrue(all(b >= a for a, b in zip(forces, forces[1:])))
        self.assertLessEqual(max(forces), 30.0)
        self.assertEqual(drive.status(), ["slipping_at_cap"] * 2)

    def test_a_finished_drive_no_longer_touches_the_grip(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        prev.check_nut(0)
        drive = prev.drive(0, slip_time=0.01)
        drive.close()
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0
        for _ in range(5):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.needed)

    def test_the_nut_is_let_go_only_on_the_table_or_threaded_on(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        self.assertTrue(prev.lift_test("nut_0", lift_with(env, nut), steps=10))
        put_nut(nut, 0.10)
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.25, places=6)  # carry speed
        put_nut(nut, 0.035)
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.03, places=6)  # slow over the bolt top
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        put_nut(nut, 0.025)  # resting on the bolt top: it would slide off
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))
        put_nut(nut, 0.022)  # a turn and a half down: the thread holds it
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))
        put_nut(nut, 0.02, dx=-0.12)  # and lying on the table, clear of the bolt
        self.assertTrue(bool(prev.supported("nut_0").all()))
        put_nut(nut, 0.02, dx=0.016, tilt_deg=60.0)  # low, but tipped up against the bolt's side: not "on the table"
        self.assertFalse(bool(prev.supported("nut_0").any()))

    def test_a_stop_uses_up_the_check(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        prev.check_nut(0)
        drive = prev.drive(0)
        put_nut(nut, 0.022)
        substep(env)
        nut.data.root_pos_w[:, 2] += 0.002  # backs off: wrong direction
        self.assertEqual(drive.status(), ["wrong_direction"] * 2)
        put_nut(nut, 0.022)  # even back down, this drive stays stopped ...
        self.assertEqual(drive.status(), ["wrong_direction"] * 2)
        drive = prev.drive(0)  # ... and a new drive without a new check is refused
        self.assertEqual(drive.status(), ["unchecked"] * 2)
        put_nut(nut, 0.025)
        prev.check_nut(0)
        self.assertEqual(prev.drive(0).status(), ["driving"] * 2)

    # -- regressions from the bug review ----------------------------------------------------------
    def test_regripping_is_not_slip(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.020)  # 2.5 turns on: the thread holds it
        prev.mark_held("nut_0")
        put_nut(nut, 0.025)
        prev.check_nut(0)
        drive = prev.drive(0, slip_time=0.01)
        put_nut(nut, 0.020)
        d = env.robot.articulation.data
        d.joint_pos[:, 7:9] += 0.01  # the wrist ran out: fingers open on the threaded nut ...
        substep(env)
        self.assertFalse(bool(prev.carrying().any()))  # (the carry ends: the thread holds it)
        d.body_ang_vel_w[:, 1, 2] = -2.0  # ... and the wrist swings back while the nut stays put
        for _ in range(30):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.needed)  # not slip: no boost
        self.assertEqual(drive.status(), ["driving"] * 2)  # and the drive carries on
        d.joint_pos[:, 7:9] -= 0.01  # close again ...
        prev.mark_held("nut_0")  # ... and say so: now a spinning nut IS slip
        d.body_ang_vel_w[:, 1, 2] = 1.0
        for _ in range(3):
            substep(env)
        self.assertGreater(prev.grip.force, prev.needed)

    def test_turning_back_to_find_the_thread_start_is_allowed(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        prev.check_nut(0)
        drive = prev.drive(0)
        put_nut(nut, 0.0265)  # turned back on the bolt top: rides 1.5 mm up the thread start ...
        substep(env)
        put_nut(nut, 0.0249)  # ... and clicks down into it
        substep(env)
        self.assertEqual(drive.status(), ["driving"] * 2)
        put_nut(nut, 0.021)  # two turns on
        substep(env)
        put_nut(nut, 0.0225)  # now rising 1.5 mm IS unscrewing
        self.assertEqual(drive.status(), ["wrong_direction"] * 2)

    def test_a_check_goes_stale_if_the_nut_moves_before_the_drive(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, _ = nut_env()
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        self.assertTrue(bool(prev.check_nut(0)[0].all()))
        put_nut(nut, 0.025, tilt_deg=8.0)  # knocked crooked after the check (within the scene's 10°)
        self.assertEqual(prev.drive(0).status(), ["unchecked"] * 2)

    def test_a_check_on_one_bolt_does_not_count_on_another(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, nut, bolt0 = nut_env()
        env.scene.cfg.num_pairs = 2
        env.iscene.articulations["bolt_1"] = art(2, (0.5, 0.2, 0.0))
        nut1 = rigid(2, pos=(0.6, 0.4, 0.02))
        nut1.data.root_ang_vel_w = torch.zeros(2, 3)
        env.iscene.rigid_objects["nut_1"] = nut1
        prev = NutThreadPrevention(env)
        put_nut(nut, 0.025)
        self.assertTrue(bool(prev.check_nut(0)[0].all()))  # checked on bolt 0 ...
        nut.data.root_pos_w[:, 1] = 0.2  # ... then set on bolt 1, same height and pose
        self.assertEqual(prev.nearest_bolt(0).tolist(), [1, 1])
        self.assertEqual(prev.drive(0).status(), ["unchecked"] * 2)

    def test_too_slippery_a_nut_is_reported(self):
        from robobench.suites.assembly.prevention.nut_thread import NutThreadPrevention

        env, _, _ = nut_env()
        env.scene.cfg.nut_friction = 0.01  # the scene default, without the robot configs' 0.4
        rep = NutThreadPrevention(env).grip_report()
        self.assertAlmostEqual(rep["required_N_to_hold_nut"], 35.43, places=2)
        self.assertTrue(rep["capped"])
        env.scene.cfg.nut_friction = 0.4
        self.assertFalse(NutThreadPrevention(env).grip_report()["capped"])

    # -- regressions from the bulb bug review -----------------------------------------------------
    def test_regripping_the_bulb_does_not_squeeze_the_glass(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.031)
        prev.mark_held("bulb_0")
        screw = prev.screw(0, slip_time=0.01)
        d = env.robot.articulation.data
        d.joint_pos[:, 7:9] += 0.01  # wrist out of travel: fingers open on the threaded bulb ...
        substep(env)
        d.body_ang_vel_w[:, 1, 2] = -2.0  # ... and the wrist swings back; the bulb stays put
        for _ in range(30):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)  # no squeeze on the glass
        self.assertEqual(screw.status(), ["turning"] * 2)

    def test_an_old_screw_monitor_stops_when_a_new_one_starts(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.031)
        prev.mark_held("bulb_0")
        first = prev.screw(0, slip_time=0.01)
        first.close()
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0  # hand turns, bulb does not
        for _ in range(5):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)  # a closed monitor does nothing
        prev.screw(0, slip_time=0.01)
        second = prev.screw(0, slip_time=0.01)  # replaces (closes) the one before
        for _ in range(3):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.hold_force * 1.25 ** 3, places=5)  # one boost/step, not two
        self.assertIs(prev._screw, second)

    def test_no_more_squeezing_after_a_stop(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        put_bulb(bulb, 0.031)
        prev.mark_held("bulb_0")
        screw = prev.screw(0, slip_time=0.01, bind_window_s=0.05)
        env.robot.articulation.data.applied_torque[:, 1] = 50.0  # pushing hard, no progress
        for _ in range(8):
            substep(env)
        self.assertEqual(screw.status(), ["binding"] * 2)
        f = prev.grip.force
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = 1.0  # still twisting after the stop
        for _ in range(5):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, f)  # stopped: no more squeeze

    def test_a_bulb_standing_on_its_cap_is_not_let_go(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        bulb.data.root_pos_w[:] = torch.tensor([0.13, 0.0, 0.0])  # upright on its cap, on the table
        bulb.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertFalse(bool(prev.supported("bulb_0").any()))  # top-heavy: it would tip onto the glass
        s = math.sqrt(0.5)
        bulb.data.root_pos_w[:] = torch.tensor([0.13, 0.0, 0.025])  # lying on its side
        bulb.data.root_quat_w[:] = torch.tensor([s, s, 0.0, 0.0])
        self.assertTrue(bool(prev.supported("bulb_0").all()))
