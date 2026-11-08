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
        d.joint_pos[:, 7:9] += 0.005  # fingers open
        substep(env)
        self.assertFalse(bool(prev.carrying().any()))
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)  # next grasp starts minimal
