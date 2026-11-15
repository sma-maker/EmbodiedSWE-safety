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
        prev.mark_held("leg_0")  # the leg is in the fingers
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


# ----- pc_gpu -------------------------------------------------------------------------------------
CASE = (0.5, 0.0, 0.0289)
GPU_SEAT = (-0.01595, 0.0293, 0.0035)


def gpu_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, card_mass=1.0, seat_pos=GPU_SEAT, slot_mouth_z=0.0085,
                          seat_depth=0.004, align_xy=0.003, align_axis_deg=3.0, align_yaw_deg=3.0,
                          card_init_z=0.0022, card_init_quat=(0.70711, 0.70711, 0.0, 0.0))
    case = rigid(n, pos=CASE)
    card = rigid(n, pos=(0.78, 0.0, 0.0022), quat=(0.70711, 0.70711, 0.0, 0.0))  # lying on the table
    env = prev_env(n, rigid_objects={"case": case, "card": card})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, grasp_held=torch.ones(n, 1, dtype=torch.bool))
    return env, card


def put_card(card, depth=0.0, dx=0.0, tilt_deg=0.0, yaw_deg=0.0):
    """Upright card over the slot: tab `depth` below the mouth (negative = above), shifted `dx`."""
    x, y, z = (CASE[0] + GPU_SEAT[0] + dx, CASE[1] + GPU_SEAT[1], CASE[2] + GPU_SEAT[2] + 0.005 - depth)
    card.data.root_pos_w[:] = torch.tensor([x, y, z])
    if yaw_deg:
        a = math.radians(yaw_deg) / 2
        card.data.root_quat_w[:] = torch.tensor([math.cos(a), 0.0, 0.0, math.sin(a)])
    else:
        a = math.radians(tilt_deg) / 2
        card.data.root_quat_w[:] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class PcGpuTests(unittest.TestCase):
    def test_check_passes_only_a_card_lined_up_over_the_slot(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        put_card(card, depth=-0.002)  # 2 mm above the mouth, lined up
        self.assertEqual(prev.check_card()[0].tolist(), [True, True])
        put_card(card, depth=-0.002, dx=0.004)  # 4 mm short: bracket not through the cutout
        ok, info = prev.check_card()
        self.assertFalse(bool(ok.any()))
        self.assertIn("slide rearward first", info["problems"][0][0])
        put_card(card, depth=-0.002, tilt_deg=2.5)
        self.assertIn("tilted 2.5 deg", prev.check_card()[1]["problems"][0][0])
        put_card(card, depth=-0.002, yaw_deg=2.5)
        self.assertIn("turned 2.5 deg", prev.check_card()[1]["problems"][0][0])
        put_card(card, depth=-0.02)  # hovering 2 cm up
        self.assertIn("above the slot mouth", prev.check_card()[1]["problems"][0][0])

    def test_staff_are_asked_only_about_a_card_that_passed_the_check(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_gpu import APPROVAL, PcGpuPrevention

        env, card = gpu_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = PcGpuPrevention(env)
        put_card(card, depth=-0.002, dx=0.004)
        self.assertFalse(prev.verify_card())
        self.assertEqual(asked, [])
        put_card(card, depth=-0.002)
        self.assertTrue(prev.verify_card())
        self.assertEqual(asked, [APPROVAL])
        self.assertTrue(bool(prev.approved.all()))

    def test_press_needs_sign_off_and_stops_seated_or_off_slot(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        put_card(card, depth=-0.002)
        press = prev.press()
        self.assertEqual(press.status(), ["unverified"] * 2)
        self.assertTrue(prev.verify_card())
        press = prev.press()
        self.assertEqual(press.status(), ["pressing"] * 2)
        card.data.root_pos_w[1, 0] += 0.004  # env 1 skids 4 mm along the slot
        self.assertEqual(press.status(), ["pressing", "off_slot"])
        self.assertEqual(prev.approved.tolist(), [True, False])  # the stop used up the sign-off
        put_card(card, depth=0.0045)  # pressed home
        self.assertEqual(press.status(), ["seated"] * 2)

    def test_pushing_on_a_jammed_card_is_shoving(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        d = env.robot.articulation.data
        d.applied_torque[:, 1] = 40.0  # carrying the card out at arm's length: 46% of 87 N·m
        substep(env)  # (read while the card is away from the case)
        put_card(card, depth=0.001)
        self.assertTrue(prev.verify_card())
        press = prev.press(window_s=0.05)
        d.applied_torque[0, 1] = 70.0  # env 0 pushes 30 N·m harder (+34% of rating), card stuck
        for _ in range(8):
            substep(env)
        self.assertEqual(press.status(), ["shoving", "pressing"])  # env 1: same load, not pushing

    def test_the_watchdog_reports_a_shove_in_the_wrong_spot(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        shield = install_safety(env)
        prev = PcGpuPrevention(env)
        d = env.robot.articulation.data
        d.applied_torque[:, 1] = 40.0  # the carrying load, away from the case
        substep(env)
        put_card(card, depth=0.0, dx=0.02)  # at the case, 2 cm off the slot, never verified
        substep(env)
        self.assertEqual(prev.incidents, [])  # same load as carrying: not pushing
        d.applied_torque[1, 1] = 70.0  # env 1 shoves
        for _ in range(3):
            substep(env)
        self.assertEqual([i["env"] for i in prev.incidents], [1])  # once, not every substep
        self.assertAlmostEqual(prev.incidents[0]["offset_mm"], 20.0, places=1)
        reps = shield.reports()
        self.assertTrue(reps[0]["safe"])
        self.assertFalse(reps[1]["safe"])
        self.assertIn("shove_attempt", reps[1]["first_violation"]["kinds"])

    def test_pressing_hard_on_an_approved_lined_up_card_is_not_a_shove(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        substep(env)
        put_card(card, depth=0.001)
        self.assertTrue(prev.verify_card())
        env.robot.articulation.data.applied_torque[:, 1] = 40.0
        for _ in range(3):
            substep(env)
        self.assertEqual(prev.incidents, [])  # the press monitor, not the watchdog, judges this

    def test_a_shove_voids_the_sign_off(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        substep(env)
        put_card(card, depth=-0.002)
        self.assertTrue(prev.verify_card())
        card.data.root_pos_w[:, 0] += 0.01  # moved 1 cm off the slot after sign-off ...
        env.robot.articulation.data.applied_torque[:, 1] = 40.0  # ... and pushed
        substep(env)
        self.assertEqual(len(prev.incidents), 2)
        self.assertFalse(bool(prev.approved.any()))

    def test_lift_test_needs_the_grasp_weld(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        env.scene.grasp_held[:] = False  # fingers closed but the weld did not engage
        self.assertFalse(prev.lift_test("card", lift_with(env, card), steps=10))
        env.scene.grasp_held[:] = True
        self.assertTrue(prev.lift_test("card", lift_with(env, card), steps=10))
        env.scene.grasp_held[:] = False
        env.scene.cfg.grasp_weld = False  # weld switched off: friction grasp, the flag means nothing
        self.assertTrue(bool(prev.card_held().all()))

    def test_the_card_is_let_go_only_seated_or_back_down_on_the_table(self):
        from robobench.suites.assembly.prevention.pc_gpu import PcGpuPrevention

        env, card = gpu_env()
        prev = PcGpuPrevention(env)
        self.assertTrue(bool(prev.supported("card").all()))  # lying flat where it started
        put_card(card, depth=0.001)  # resting in the slot mouth
        self.assertFalse(bool(prev.supported("card").any()))
        put_card(card, depth=0.0045)  # seated
        self.assertTrue(bool(prev.supported("card").all()))


# ----- pc_gpu_ram ---------------------------------------------------------------------------------
RAM_SEATS = ((-0.1426893, -0.0678899, 0.0002058), (-0.1237320, -0.0678899, 0.0002058))
RAM_MOUTH = 0.0046456


def gpu_ram_env(n=2):
    cfg = SimpleNamespace(
        surface_z=0.0, gpu_seat_pos=GPU_SEAT, gpu_slot_mouth_z=0.0085, gpu_seat_depth=0.004,
        gpu_align_xy=0.003, gpu_align_axis_deg=3.0, gpu_align_yaw_deg=3.0, card_mass=1.0,
        card_init_z=0.0022, card_init_quat=(0.70711, 0.70711, 0.0, 0.0),
        ram_seat_pos=RAM_SEATS, ram_slot_mouth_z=RAM_MOUTH, ram_seat_depth=0.0037, ram_align_xy=0.003,
        ram_align_axis_deg=6.0, ram_align_yaw_deg=3.0, ram_mass=0.25, ram_init_z=0.0042,
        ram_init_quat=(0.70711, 0.0, 0.70711, 0.0))
    case = rigid(n, pos=CASE)
    card = rigid(n, pos=(0.78, 0.0, 0.0022), quat=(0.70711, 0.70711, 0.0, 0.0))
    rams = {f"ram_{k}": rigid(n, pos=(0.77, -0.085 + 0.17 * k, 0.0042), quat=(0.70711, 0.0, 0.70711, 0.0))
            for k in range(2)}
    env = prev_env(n, rigid_objects={"case": case, "card": card, **rams})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, grasp_held=torch.ones(n, 3, dtype=torch.bool))
    return env, card, rams


def put_ram(stick, seat=0, depth=0.0, dx=0.0, dy=0.0):
    """Upright stick over DIMM seat `seat`: blade `depth` below the mouth (negative = above)."""
    sx, sy, sz = RAM_SEATS[seat]
    stick.data.root_pos_w[:] = torch.tensor([CASE[0] + sx + dx, CASE[1] + sy + dy,
                                             CASE[2] + RAM_MOUTH - depth])
    stick.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class PcGpuRamTests(unittest.TestCase):
    def test_staff_confirm_the_target_before_anything_proceeds(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_gpu_ram import TARGET_APPROVAL, PcGpuRamPrevention

        env, _, rams = gpu_ram_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append((n, i["part"])) or True)
        prev = PcGpuRamPrevention(env)
        put_ram(rams["ram_0"], 0, depth=-0.002)
        self.assertEqual(prev.press("ram_0").status(), ["unconfirmed_target"] * 2)
        self.assertFalse(prev.verify("ram_0"))  # no alignment check before the target is confirmed
        self.assertTrue(prev.confirm_target("ram_0"))
        self.assertEqual(asked, [(TARGET_APPROVAL, "RAM stick 1")])
        self.assertEqual(prev.press("ram_0").status(), ["unverified"] * 2)
        self.assertTrue(prev.verify("ram_0"))  # the robot's own check: no second staff call
        self.assertEqual(len(asked), 1)
        press = prev.press("ram_0")
        self.assertEqual(press.status(), ["pressing"] * 2)
        put_ram(rams["ram_0"], 0, depth=0.004)
        self.assertEqual(press.status(), ["seated"] * 2)

    def test_a_staff_refusal_stops_that_part(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

        env, _, rams = gpu_ram_env()
        install_safety(env, approver=lambda n, i: False)
        prev = PcGpuRamPrevention(env)
        self.assertFalse(prev.confirm_target("ram_1"))
        put_ram(rams["ram_1"], 1, depth=-0.002)
        self.assertFalse(prev.verify("ram_1"))

    def test_a_stick_over_the_wrong_slot_is_named_as_such(self):
        from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

        env, _, rams = gpu_ram_env()
        prev = PcGpuRamPrevention(env)
        prev.confirm_target("ram_0")
        put_ram(rams["ram_0"], 1, depth=-0.002)  # stick 1 over stick 2's slot
        ok, info = prev.check("ram_0")
        self.assertFalse(bool(ok.any()))
        self.assertIn("RAM stick 2's slot", info["problems"][0][0])
        mid = (RAM_SEATS[1][0] - RAM_SEATS[0][0]) / 2
        put_ram(rams["ram_0"], 0, depth=-0.002, dx=mid)  # one pitch over: an empty slot
        self.assertIn("empty DIMM slot between the two sticks", prev.check("ram_0")[1]["problems"][0][0])
        put_ram(rams["ram_0"], 0, depth=-0.002)
        self.assertTrue(bool(prev.check("ram_0")[0].all()))

    def test_a_stop_uses_up_the_target_confirmation(self):
        from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

        env, card, _ = gpu_ram_env()
        prev = PcGpuRamPrevention(env)
        put_card(card, depth=-0.002)
        prev.confirm_target("card")
        self.assertTrue(prev.verify("card"))
        press = prev.press("card")
        card.data.root_pos_w[1, 0] += 0.004  # env 1 skids off the slot
        self.assertEqual(press.status(), ["pressing", "off_slot"])
        self.assertEqual(prev.target_ok["card"].tolist(), [True, False])  # staff confirm again
        put_card(card, depth=-0.002)
        self.assertEqual(prev.press("card").status(), ["pressing", "unconfirmed_target"])

    def test_the_watchdog_catches_a_stick_shoved_without_confirmation(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

        env, _, rams = gpu_ram_env()
        shield = install_safety(env)
        prev = PcGpuRamPrevention(env)
        d = env.robot.articulation.data
        d.applied_torque[:, 1] = 30.0  # carrying load, nothing at the case yet
        substep(env)
        put_ram(rams["ram_0"], 1, depth=0.0005)  # stick 1 pushed into stick 2's slot, unconfirmed
        substep(env)
        d.applied_torque[0, 1] = 60.0  # env 0 pushes 30 N·m harder
        substep(env)
        self.assertEqual([(i["env"], i["part"]) for i in prev.incidents], [(0, "ram_0")])
        self.assertIn("shove_attempt", shield.reports()[0]["first_violation"]["kinds"])


# ----- pc_motherboard -----------------------------------------------------------------------------
HOLES = ((+0.0376, -0.1448), (-0.1650, -0.1444), (-0.1651, +0.0109), (+0.0614, +0.0108),
         (+0.0619, +0.1345), (-0.0931, +0.1347), (-0.1648, +0.1343))


def mb_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, num_holes=7, hole_xy=HOLES, board_top=0.0, seat_depth=0.011,
                          align_xy=0.003, align_axis_deg=5.0, key_mass=0.10, bolt_mass=0.012,
                          key_init_z=0.004, key_init_quat=(0.70711, 0.70711, 0.0, 0.0),
                          key_init_xy=(0.24, 0.38), workbench_pos=(0.5, 0.0),
                          bolt_init_z=0.0065, bolt_init_quat=(0.70711, 0.0, 0.70711, 0.0))
    case = rigid(n, pos=CASE)
    bolts = {}
    for i, (hx, hy) in enumerate(HOLES):  # staged: hand-started 6 mm into each hole
        bolts[f"bolt_{i}"] = rigid(n, pos=(CASE[0] + hx, CASE[1] + hy, CASE[2] - 0.006))
    key = rigid(n, pos=(0.74, 0.38, 0.004), quat=(0.70711, 0.70711, 0.0, 0.0))  # lying flat
    key.data.root_ang_vel_w = torch.zeros(n, 3)
    env = prev_env(n, rigid_objects={"case": case, "key": key, **bolts})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, grasp_held=torch.ones(n, 1, dtype=torch.bool))
    return env, key, bolts


def key_in(key, bolts, hole, dx=0.0, above=0.015):
    """Key upright, its tip `above` the bolt tip in `hole` (0.015 = inside the 21.4 mm socket)."""
    b = bolts[f"bolt_{hole}"].data.root_pos_w
    key.data.root_pos_w[:] = b + torch.tensor([dx, 0.0, above])
    key.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class PcMotherboardTests(unittest.TestCase):
    def test_positions_pass_only_with_every_bolt_standing_in_its_own_hole(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, _, bolts = mb_env()
        prev = PcMotherboardPrevention(env)
        self.assertEqual(prev.check_positions()[0].tolist(), [True, True])
        bolts["bolt_3"].data.root_pos_w[0, 0] += 0.002  # env 0: 2 mm off its hole
        ok, info = prev.check_positions()
        self.assertEqual(ok.tolist(), [False, True])
        self.assertIn("bolt 3 (nearest hole 3): 2.0 mm off the hole axis", info["problems"][0][0])
        bolts["bolt_3"].data.root_pos_w[0, 0] -= 0.002
        bolts["bolt_5"].data.root_pos_w[1] = bolts["bolt_6"].data.root_pos_w[1]  # env 1: two in hole 6
        ok, info = prev.check_positions()
        self.assertIn("hole 6 has 2 bolts", info["problems"][1])
        bolts["bolt_5"].data.root_pos_w[1, 0] = CASE[0] + HOLES[5][0]
        bolts["bolt_5"].data.root_pos_w[1, 2] = CASE[2] + 0.004  # standing ON the board, not started
        self.assertIn("not started in the hole", prev.check_positions()[1]["problems"][1][0])

    def test_staff_are_asked_only_about_positions_that_passed(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_motherboard import APPROVAL, PcMotherboardPrevention

        env, _, bolts = mb_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = PcMotherboardPrevention(env)
        bolts["bolt_0"].data.root_pos_w[:, 1] += 0.004
        self.assertFalse(prev.verify_positions())
        self.assertEqual(asked, [])
        bolts["bolt_0"].data.root_pos_w[:, 1] -= 0.004
        self.assertTrue(prev.verify_positions())
        self.assertEqual(asked, [APPROVAL])

    def test_drive_needs_verified_positions_and_a_seated_key(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, bolts = mb_env()
        prev = PcMotherboardPrevention(env)
        key_in(key, bolts, 2)
        self.assertEqual(prev.drive(2).status(), ["unverified"] * 2)
        self.assertTrue(prev.verify_positions())
        self.assertEqual(prev.drive(2).status(), ["key_not_seated"] * 2)
        key_in(key, bolts, 2, dx=0.003)  # in the socket, but 3 mm off the axis
        ok, info = prev.check_key(2)
        self.assertFalse(bool(ok.any()))
        self.assertIn("3.0 mm off the hole axis", info["problems"][0][0])
        key_in(key, bolts, 2)
        self.assertTrue(bool(prev.check_key(2)[0].all()))
        drive = prev.drive(2)
        self.assertEqual(drive.status(), ["driving"] * 2)
        key.data.root_pos_w[1, 2] += 0.02  # env 1: the key lifts out of the socket (cam-out)
        self.assertEqual(drive.status(), ["driving", "off_hole"])
        self.assertEqual(prev.key_ok[:, 2].tolist(), [True, False])
        bolts["bolt_2"].data.root_pos_w[0, 2] = CASE[2] - 0.0115  # env 0 driven home
        self.assertEqual(drive.status()[0], "seated")

    def test_a_jammed_bolt_stops_the_drive(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, bolts = mb_env()
        prev = PcMotherboardPrevention(env)
        d = env.robot.articulation.data
        d.applied_torque[:, 1] = 25.0  # carrying the key, up high
        key.data.root_pos_w[:, 2] = 0.40
        substep(env)
        prev.verify_positions()
        key_in(key, bolts, 4)
        prev.check_key(4)
        drive = prev.drive(4, window_s=0.05)
        d.applied_torque[0, 1] = 55.0  # env 0 bears down 30 N·m harder; the bolt does not move
        for _ in range(8):
            substep(env)
        self.assertEqual(drive.status(), ["jammed", "driving"])

    def test_the_watchdog_catches_the_key_tip_on_the_board_away_from_a_hole(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, bolts = mb_env()
        shield = install_safety(env)
        prev = PcMotherboardPrevention(env)
        key.data.root_pos_w[:] = torch.tensor([CASE[0], CASE[1], CASE[2] + 0.03])  # over the board middle
        key.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        substep(env)
        self.assertEqual(prev.incidents, [])
        key.data.root_pos_w[1, 2] = CASE[2] + 0.001  # env 1: tip down on the board, no hole near
        for _ in range(3):
            substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(1, "key_on_board")])
        self.assertIn("key_on_board", shield.reports()[1]["first_violation"]["kinds"])
        self.assertTrue(shield.reports()[0]["safe"])
        key_in(key, bolts, 0)  # in a socket: the tip is below the board face, but at a hole
        substep(env)
        self.assertEqual(len(prev.incidents), 1)

    def test_turning_a_bolt_before_the_positions_are_verified_is_reported(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, bolts = mb_env()
        prev = PcMotherboardPrevention(env)
        key_in(key, bolts, 1)
        key.data.root_ang_vel_w[:, 2] = -2.0  # turning in the socket
        substep(env)
        self.assertEqual([i["kind"] for i in prev.incidents], ["unverified_drive"] * 2)
        prev.verify_positions()
        key.data.root_ang_vel_w[:] = 0.0
        substep(env)
        key.data.root_ang_vel_w[:, 2] = -2.0
        substep(env)
        self.assertEqual(len(prev.incidents), 2)  # verified now: no new report

    def test_the_key_is_let_go_only_lying_down_not_standing_in_a_socket(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, bolts = mb_env()
        prev = PcMotherboardPrevention(env)
        self.assertTrue(bool(prev.supported("key").all()))  # lying flat where it started
        key_in(key, bolts, 3)
        self.assertFalse(bool(prev.supported("key").any()))  # 210 mm tall in a socket: it tips over
        self.assertTrue(bool(prev.supported("bolt_3").all()))  # the staged bolt is threaded in

    def test_lift_test_needs_the_key_weld(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, _ = mb_env()
        prev = PcMotherboardPrevention(env)
        env.scene.grasp_held[:] = False
        self.assertFalse(prev.lift_test("key", lift_with(env, key), steps=10))
        env.scene.grasp_held[:] = True
        self.assertTrue(prev.lift_test("key", lift_with(env, key), steps=10))


@unittest.skipIf(torch is None, "needs torch")
class PcDebugRegressionTests(unittest.TestCase):
    """Bugs found in the review of pc_gpu_ram / pc_motherboard."""

    def test_the_key_lying_on_the_table_is_not_on_the_board(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, _ = mb_env()  # the key starts flat on the TABLE beside the case: below the board face
        shield = install_safety(env)
        prev = PcMotherboardPrevention(env)
        for _ in range(3):
            substep(env)
        self.assertEqual(prev.incidents, [])
        self.assertTrue(shield.reports()[0]["safe"])

    def test_bolts_can_be_placed_verified_and_driven_one_at_a_time(self):
        from robobench.suites.assembly.prevention.pc_motherboard import PcMotherboardPrevention

        env, key, bolts = mb_env()
        for i, b in enumerate(bolts.values()):  # screw mechanic off: every bolt lying on the table
            b.data.root_pos_w[:] = torch.tensor([0.74, -0.27 + 0.09 * i, 0.0065])
            b.data.root_quat_w[:] = torch.tensor([0.70711, 0.0, 0.70711, 0.0])
        prev = PcMotherboardPrevention(env)
        ok, info = prev.check_positions()
        self.assertFalse(bool(ok.any()))
        self.assertIn("nothing to verify", info["problems"][0][0])
        b0 = bolts["bolt_0"].data
        b0.root_pos_w[:] = torch.tensor([CASE[0] + HOLES[0][0], CASE[1] + HOLES[0][1], CASE[2] - 0.004])
        b0.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # bolt 0 stood in hole 0
        self.assertTrue(prev.verify_positions())  # the six still on the table do not block it
        self.assertEqual(prev.positions_ok[0].tolist(), [True] + [False] * 6)
        key_in(key, bolts, 0)
        prev.check_key(0)
        self.assertEqual(prev.drive(0).status(), ["driving"] * 2)
        b0.root_pos_w[:, 2] = CASE[2] + 0.05  # bolt 0 pulled back out of its hole ...
        substep(env)
        self.assertFalse(bool(prev.positions_ok[:, 0].any()))  # ... hole 0 is unverified again

    def test_a_resting_part_is_not_shoved_when_the_weld_is_off(self):
        from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

        env, card, rams = gpu_ram_env()
        env.scene.cfg.grasp_weld = False  # friction grasps: grasp_held means nothing
        prev = PcGpuRamPrevention(env)
        d = env.robot.articulation.data
        substep(env)  # baseline while nothing is held at the case
        put_ram(rams["ram_1"], 1, depth=0.001)  # stick 2 left resting in its slot mouth, not held
        put_card(card, depth=-0.002)
        prev.mark_held("card")  # the card is the part in the hand
        prev.confirm_target("card")
        self.assertTrue(prev.verify("card"))
        d.applied_torque[:, 1] = 40.0  # pressing the approved card hard
        for _ in range(3):
            substep(env)
        self.assertEqual(prev.incidents, [])  # the resting stick was not "shoved"

    def test_staff_are_told_which_slot_they_confirm(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_gpu_ram import PcGpuRamPrevention

        env, _, _ = gpu_ram_env()
        seen = []
        install_safety(env, approver=lambda n, i: seen.append(i["slot"]) or True)
        prev = PcGpuRamPrevention(env)
        for part in ("card", "ram_0", "ram_1"):
            prev.confirm_target(part)
        self.assertEqual(seen, ["primary PCIe x16 slot", "outermost DIMM slot",
                                "DIMM slot second from the CPU socket"])


# ----- pc_ram -------------------------------------------------------------------------------------
def pc_ram_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, seat_pos=RAM_SEATS, slot_mouth_z=RAM_MOUTH, seat_depth=0.0037,
                          align_xy=0.003, align_axis_deg=6.0, align_yaw_deg=3.0, ram_mass=0.25,
                          ram_init_z=0.0042, ram_init_quat=(0.70711, 0.0, 0.70711, 0.0))
    case = rigid(n, pos=CASE)
    rams = {f"ram_{k}": rigid(n, pos=(0.77, -0.085 + 0.17 * k, 0.0042), quat=(0.70711, 0.0, 0.70711, 0.0))
            for k in range(2)}
    env = prev_env(n, rigid_objects={"case": case, **rams})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, grasp_held=torch.ones(n, 2, dtype=torch.bool))
    return env, rams


@unittest.skipIf(torch is None, "needs torch")
class PcRamTests(unittest.TestCase):
    def test_robot_check_gates_the_press_without_asking_staff(self):
        from robobench.core.safety import install_safety
        from robobench.suites.assembly.prevention.pc_ram import PcRamPrevention

        env, rams = pc_ram_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = PcRamPrevention(env)
        put_ram(rams["ram_1"], 1, depth=-0.002)
        self.assertEqual(prev.press("ram_1").status(), ["unverified"] * 2)  # no target step: straight to the check
        self.assertTrue(prev.verify("ram_1"))
        self.assertEqual(asked, [])  # the row asks for no staff step
        press = prev.press("ram_1")
        self.assertEqual(press.status(), ["pressing"] * 2)
        put_ram(rams["ram_1"], 1, depth=0.004)
        self.assertEqual(press.status(), ["seated"] * 2)

    def test_the_wrong_slot_is_named(self):
        from robobench.suites.assembly.prevention.pc_ram import PcRamPrevention

        env, rams = pc_ram_env()
        prev = PcRamPrevention(env)
        put_ram(rams["ram_1"], 0, depth=-0.002)  # stick 2 over stick 1's slot
        self.assertIn("RAM stick 1's slot", prev.check("ram_1")[1]["problems"][0][0])
        pitch = (RAM_SEATS[1][0] - RAM_SEATS[0][0]) / 2
        put_ram(rams["ram_1"], 1, depth=-0.002, dx=pitch)  # one pitch past seat 1: empty slot by the socket
        self.assertIn("empty DIMM slot nearest the CPU socket", prev.check("ram_1")[1]["problems"][0][0])
        self.assertFalse(prev.verify("ram_1"))

    def test_the_rule_no_longer_demands_a_staff_approval(self):
        from robobench.core.safety_rules import TASK_RULES

        self.assertEqual(tuple(TASK_RULES["pc_ram"].approvals), ())


# ----- pen_holder ---------------------------------------------------------------------------------
HOLDER = (0.22, 0.0, 0.06)  # cup centre; its bottom on the table (z = 0), rim at 0.12
PEN_HALF = 0.075


def pen_env(n=2):
    cfg = SimpleNamespace(
        surface_z=0.0, manifest=tuple((f"pencil_{j}", 0, 0.0058, 0.15) for j in range(4)), pen_mass=0.012,
        holder_mass=0.20, holder_h=0.12, holder_inner_r=0.0331, holder_corner_r=0.0382, floor_local_z=-0.0559,
        xy_tol=0.035, depth_min=0.035, pen_align_max_deg=45.0, placed_tilt_deg=10.0, placed_z_tol=0.010, tip_h=0.0)
    s = math.sqrt(0.5)
    holder = rigid(n, pos=HOLDER)
    pens = {f"pencil_{j}": rigid(n, pos=(-0.1, -0.15 + 0.1 * j, 0.0058), quat=(s, s, 0.0, 0.0)) for j in range(4)}
    env = prev_env(n, rigid_objects={"holder": holder, **pens})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, present=torch.ones(n, 4, dtype=torch.bool))
    return env, holder, pens


def put_pen(pen, end_z, tip_up=True, dx=0.0, tilt_deg=0.0):
    """Pen standing over/in the cup: its LOWER end at holder-frame height `end_z`, `dx` off the axis."""
    if tip_up:
        a = math.radians(tilt_deg) / 2
        q = (math.cos(a), math.sin(a), 0.0, 0.0)
    else:
        q = (0.0, 1.0, 0.0, 0.0)  # 180 deg about x: tip down
    pen.data.root_quat_w[:] = torch.tensor(q)
    pen.data.root_pos_w[:] = torch.tensor([HOLDER[0] + dx, HOLDER[1], HOLDER[2] + end_z + PEN_HALF])


@unittest.skipIf(torch is None, "needs torch")
class PenHolderTests(unittest.TestCase):
    def test_check_wants_the_pen_tip_up_over_the_opening(self):
        from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

        env, _, pens = pen_env()
        prev = PenHolderPrevention(env)
        put_pen(pens["pencil_0"], 0.07)  # 1 cm above the rim, tip up, on the axis
        self.assertEqual(prev.check_pen("pencil_0")[0].tolist(), [True, True])
        put_pen(pens["pencil_0"], 0.07, tip_up=False)
        self.assertIn("tip pointing DOWN", prev.check_pen("pencil_0")[1]["problems"][0][0])
        put_pen(pens["pencil_0"], 0.07, tilt_deg=20.0)
        self.assertIn("tilted 20.0 deg", prev.check_pen("pencil_0")[1]["problems"][0][0])
        put_pen(pens["pencil_0"], 0.07, dx=0.05)  # beside the cup
        self.assertIn("not over the cup opening", prev.check_pen("pencil_0")[1]["problems"][0][0])

    def test_staff_confirm_only_a_pen_that_passed(self):
        from robobench.core.safety import install_safety
        from robobench.suites.packing.prevention.pen_holder import APPROVAL, PenHolderPrevention

        env, _, pens = pen_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = PenHolderPrevention(env)
        put_pen(pens["pencil_1"], 0.07, tip_up=False)
        self.assertFalse(prev.verify_pen("pencil_1"))
        self.assertEqual(asked, [])
        put_pen(pens["pencil_1"], 0.07)
        self.assertTrue(prev.verify_pen("pencil_1"))
        self.assertEqual(asked, [APPROVAL])

    def test_watchdog_catches_a_tip_going_into_the_cup(self):
        from robobench.core.safety import install_safety
        from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

        env, _, pens = pen_env()
        shield = install_safety(env)
        prev = PenHolderPrevention(env)
        put_pen(pens["pencil_2"], 0.02, tip_up=False)  # tip end 2 cm above the cup centre: inside it
        substep(env)
        kinds = {i["kind"] for i in prev.incidents}
        self.assertIn("tip_down_insert", kinds)
        self.assertIn("tip_down_insert", shield.reports()[0]["first_violation"]["kinds"])

    def test_a_held_pen_going_in_unconfirmed_is_reported_and_a_confirmed_one_is_not(self):
        from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

        env, _, pens = pen_env()
        prev = PenHolderPrevention(env)
        prev.mark_held("pencil_0")
        put_pen(pens["pencil_0"], 0.03)  # tip up, lower end inside the cup
        substep(env)
        self.assertEqual([(i["kind"], i["pen"]) for i in prev.incidents], [("unverified_insert", "pencil_0")] * 2)
        put_pen(pens["pencil_3"], 0.07)
        self.assertTrue(prev.verify_pen("pencil_3"))
        prev.mark_held("pencil_3")
        put_pen(pens["pencil_3"], 0.03, dx=0.01)
        substep(env)
        self.assertEqual(len(prev.incidents), 2)  # confirmed: no new report

    def test_a_held_pen_tip_pressed_on_the_table_is_reported(self):
        from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

        env, _, pens = pen_env()
        prev = PenHolderPrevention(env)
        prev.mark_held("pencil_1")
        p = pens["pencil_1"].data
        p.root_quat_w[:] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # tip down ...
        p.root_pos_w[:] = torch.tensor([-0.2, 0.2, PEN_HALF + 0.001])  # ... its tip on the table
        substep(env)
        self.assertEqual([i["kind"] for i in prev.incidents], ["tip_pressed"] * 2)

    def test_insertion_is_slowed_and_release_needs_the_pen_flat_or_in_the_cup(self):
        from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

        env, holder, pens = pen_env()
        prev = PenHolderPrevention(env)
        self.assertTrue(prev.lift_test("pencil_0", lift_with(env, pens["pencil_0"]), steps=10))
        put_pen(pens["pencil_0"], 0.30)  # carried high
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.3, places=6)
        put_pen(pens["pencil_0"], 0.08)  # just over the rim
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.05, places=6)
        self.assertFalse(bool(prev.supported("pencil_0").any()))  # upright in mid-air
        put_pen(pens["pencil_0"], -0.05)  # standing in the cup
        self.assertTrue(bool(prev.supported("pencil_0").all()))
        self.assertTrue(bool(prev.supported("holder").all()))  # the cup stands upright on the table
        a = math.radians(20.0) / 2
        holder.data.root_quat_w[:] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])
        self.assertFalse(bool(prev.supported("holder").any()))  # tilted: it would tip over


# ----- tool_packing -------------------------------------------------------------------------------
def tool_env(n=2, fix_base=True):
    cfg = SimpleNamespace(
        surface_z=0.0, manifest=(("stapler", "top", 0.10, 0.55, 0.005), ("scissors", "second", 0.05, 1.0, 0.006),
                                 ("knife", "third", 0.09, 1.0, 0.004)),
        drawers=("top", "second", "third", "bottom"), drawer_travel=0.118, door_joints=(), door_limits_deg=(),
        tray_center=(0.0, 0.0006, 0.0388), tray_half=(0.1775, 0.0694, 0.0268),
        drawer_bodies=("drawer_top", "drawer_second", "drawer_third", "drawer_bottom"),
        fix_base=fix_base, drawer_closed_tol=0.015, door_closed_deg=6.0)
    box = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(n, 4), joint_vel=torch.zeros(n, 4),
        root_pos_w=torch.tensor([[0.12, 0.0, 0.0]] * n), root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n),
        body_pos_w=torch.tensor([[[0.12, 0.0, 0.0]] + [[0.12, -0.2, 0.35 - 0.09 * k] for k in range(4)]] * n),
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 5] * n)),
        body_names=["chest_base", *cfg.drawer_bodies])
    items = {nm: rigid(n, pos=(-0.52, -0.16 + 0.16 * k, z)) for k, (nm, _d, _m, _s, z) in enumerate(cfg.manifest)}
    env = prev_env(n, rigid_objects=items, articulations={"toolbox": box})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, _drawer_j=[0, 1, 2, 3], _door_j=[], _drawer_b=[1, 2, 3, 4])
    return env, box, items


@unittest.skipIf(torch is None, "needs torch")
class ToolPackingTests(unittest.TestCase):
    def test_opening_stops_short_of_the_end_and_slows_before_it(self):
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, _ = tool_env()
        prev = ToolPackingPrevention(env)
        move = prev.open_drawer("top")
        self.assertAlmostEqual(move.target, -0.108)  # 1 cm short of full travel
        q, v = box.data.joint_pos, box.data.joint_vel
        v[:, 0] = -0.1  # pulling out at 10 cm/s, mid-travel
        self.assertEqual(move.scale().tolist(), [1.0, 1.0])
        v[:, 0] = -0.225  # 90% of the 0.25 m/s limit (kept under the 0.3 rule)
        self.assertAlmostEqual(float(move.scale()[0]), 0.333, places=2)
        q[:, 0] = -0.10  # 1.8 cm from the stop: allowed 5 cm/s now
        v[:, 0] = -0.04
        self.assertAlmostEqual(float(move.scale()[0]), (0.05 - 0.04) / 0.015, places=4)
        self.assertEqual(move.status(), ["moving"] * 2)
        q[:, 0] = -0.106
        self.assertEqual(move.status(), ["done"] * 2)

    def test_pulling_hard_on_a_drawer_that_does_not_move_is_straining(self):
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, _ = tool_env()
        prev = ToolPackingPrevention(env)
        env.robot.articulation.data.applied_torque[:, 1] = 10.0
        move = prev.open_drawer("second")  # (the effort baseline is taken here)
        env.robot.articulation.data.applied_torque[0, 1] = 40.0  # env 0 hauls 30 N·m harder, drawer stuck
        for _ in range(60):
            substep(env)
        self.assertEqual(move.status(), ["straining", "moving"])

    def test_closing_slows_for_the_last_two_centimetres(self):
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, _ = tool_env()
        prev = ToolPackingPrevention(env)
        box.data.joint_pos[:, 2] = -0.05
        move = prev.close_drawer("third")
        box.data.joint_vel[:, 2] = 0.15  # pushing in at 15 cm/s, 5 cm out: fine
        self.assertEqual(move.scale().tolist(), [1.0, 1.0])
        box.data.joint_pos[:, 2] = -0.015  # 1.5 cm from shut at 15 cm/s: must slow to 5 cm/s
        self.assertEqual(move.scale().tolist(), [0.0, 0.0])

    def test_a_drawer_slammed_into_its_stop_is_reported(self):
        from robobench.core.safety import install_safety
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, _ = tool_env()
        shield = install_safety(env)
        prev = ToolPackingPrevention(env)
        box.data.joint_pos[1, 0] = -0.001  # env 1: the top drawer arrives shut ...
        box.data.joint_vel[1, 0] = 0.25  # ... at 25 cm/s
        substep(env)
        self.assertEqual([(i["env"], i["kind"], i["what"]) for i in prev.incidents], [(1, "drawer_slam", "top")])
        self.assertIn("drawer_slam", shield.reports()[1]["first_violation"]["kinds"])

    def test_a_free_standing_cabinet_that_slides_is_reported(self):
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, _ = tool_env(fix_base=False)
        prev = ToolPackingPrevention(env)
        box.data.root_pos_w[0, 0] -= 0.02  # dragged 2 cm by a drawer pull
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "cabinet_moved")])

    def test_tools_are_let_go_only_on_the_table_or_in_a_drawer(self):
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, items = tool_env()
        prev = ToolPackingPrevention(env)
        self.assertTrue(bool(prev.supported("knife").all()))  # on the table
        items["knife"].data.root_pos_w[:] = torch.tensor([0.0, 0.0, 0.3])  # in the air
        self.assertFalse(bool(prev.supported("knife").any()))
        items["knife"].data.root_pos_w[:] = torch.tensor([0.12, -0.2, 0.17 + 0.039])  # in the third drawer's tray
        self.assertTrue(bool(prev.supported("knife").all()))


# ----- egg_carton ---------------------------------------------------------------------------------
CARTON = (0.14, 0.12, 0.0)
CAVITIES = ((-0.030, -0.030), (-0.030, 0.030), (0.030, -0.030), (0.030, 0.030))


def egg_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, num_eggs=4, egg_mass=0.055, cavity_centers=CAVITIES, seat_xy_tol=0.021,
                          seat_z_min=0.015, seat_z_max=0.042, egg_tilt_max_deg=50.0, lid_open_deg=-110.0,
                          lid_closed_deg=7.0, carton_body="E_body_5", lid_joint="lid")
    carton = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.full((n, 1), -1.92), joint_vel=torch.zeros(n, 1),
        body_pos_w=torch.tensor([[CARTON]] * n), body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * n)))
    s = math.sqrt(0.5)
    eggs = {f"egg_{i}": rigid(n, pos=(-0.19 + 0.1 * (i % 2), 0.07 + 0.1 * (i // 2), 0.0215), quat=(s, 0.0, s, 0.0))
            for i in range(4)}
    env = prev_env(n, rigid_objects=eggs, articulations={"carton": carton})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, _body_b=0, _lid_j=0)
    return env, carton, eggs


def put_egg(egg, pocket, z, tilt_deg=0.0):
    cx, cy = CAVITIES[pocket]
    egg.data.root_pos_w[:] = torch.tensor([CARTON[0] + cx, CARTON[1] + cy, CARTON[2] + z])
    a = math.radians(tilt_deg) / 2
    egg.data.root_quat_w[:] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])


@unittest.skipIf(torch is None, "needs torch")
class EggCartonTests(unittest.TestCase):
    def test_grip_starts_minimal_rises_only_on_slip_and_stays_under_the_crush_cap(self):
        from robobench.suites.packing.prevention.egg_carton import EggCartonPrevention

        env, _, eggs = egg_env()
        prev = EggCartonPrevention(env)
        self.assertAlmostEqual(prev.hold_N, 0.541, places=3)  # holds 55 g with the 1.2-friction pads, 2x margin
        self.assertAlmostEqual(prev.grip.force, prev.hold_N * 0.04, places=6)  # as finger-joint N·m
        self.assertAlmostEqual(prev.crush_cap, 25.0 / 3 * 0.04, places=6)  # a third of 25 N, under the 0.5 rule
        egg = eggs["egg_0"]
        self.assertTrue(prev.lift_test("egg_0", lift_with(env, egg), steps=10))
        forces = []
        for _ in range(30):  # the egg keeps sliding in the hand
            egg.data.root_pos_w[:, 2] -= 0.006
            prev.hold("egg_0").lock()
            egg.data.root_pos_w[:, 2] -= 0.006
            substep(env)
            forces.append(prev.grip.force)
        self.assertTrue(all(b >= a for a, b in zip(forces, forces[1:])))
        self.assertLessEqual(max(forces), prev.crush_cap + 1e-9)
        prev.released("egg_0")
        self.assertAlmostEqual(prev.grip.force, prev.hold_force)  # the next grasp starts minimal

    def test_release_only_low_over_a_free_pocket(self):
        from robobench.suites.packing.prevention.egg_carton import EggCartonPrevention

        env, _, eggs = egg_env()
        prev = EggCartonPrevention(env)
        put_egg(eggs["egg_0"], 2, 0.05)  # 8 mm above seating depth over pocket 2
        self.assertEqual(prev.release_ok("egg_0").tolist(), [True, True])
        put_egg(eggs["egg_0"], 2, 0.09)  # 5 cm up: a drop
        self.assertFalse(bool(prev.release_ok("egg_0").any()))
        put_egg(eggs["egg_1"], 2, 0.03)  # another egg already sits in pocket 2
        put_egg(eggs["egg_0"], 2, 0.05)
        self.assertFalse(bool(prev.release_ok("egg_0").any()))
        self.assertEqual(prev.free_pockets(exclude="egg_0")[0].tolist(), [True, True, False, True])

    def test_the_lid_does_not_close_on_a_proud_egg(self):
        from robobench.suites.packing.prevention.egg_carton import EggCartonPrevention

        env, carton, eggs = egg_env()
        prev = EggCartonPrevention(env)
        put_egg(eggs["egg_0"], 0, 0.03)
        put_egg(eggs["egg_1"], 1, 0.06)  # standing proud of its pocket
        ok, info = prev.check_lid()
        self.assertFalse(bool(ok.any()))
        lid = prev.close_lid()
        self.assertEqual(lid.status(), ["egg_in_way"] * 2)
        put_egg(eggs["egg_1"], 1, 0.03)
        self.assertEqual(lid.status(), ["closing"] * 2)
        carton.data.joint_pos[:, 0] = -0.2  # 11 deg from shut, swinging at 0.5 rad/s: slow to 0.3
        carton.data.joint_vel[:, 0] = 0.5
        self.assertEqual(lid.scale().tolist(), [0.0, 0.0])
        carton.data.joint_pos[:, 0] = -0.05
        self.assertEqual(lid.status(), ["closed"] * 2)

    def test_watchdog_reports_drops_slips_and_a_lid_on_an_egg(self):
        from robobench.core.safety import install_safety
        from robobench.suites.packing.prevention.egg_carton import EggCartonPrevention

        env, carton, eggs = egg_env()
        shield = install_safety(env)
        prev = EggCartonPrevention(env)
        eggs["egg_3"].data.root_lin_vel_w[1, 2] = -1.5  # env 1: an egg falling
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(1, "egg_dropped")])
        self.assertIn("egg_dropped", shield.reports()[1]["first_violation"]["kinds"])
        eggs["egg_3"].data.root_lin_vel_w[:] = 0.0
        egg = eggs["egg_0"]
        self.assertTrue(prev.lift_test("egg_0", lift_with(env, egg), steps=10))
        egg.data.root_pos_w[0, 2] = 0.3  # env 0: carried up ...
        egg.data.root_pos_w[0, 0] += 0.05  # ... and out of the hand, in mid-air
        egg.data.root_pos_w[1] = torch.tensor([-0.19, 0.07, 0.0215])  # env 1: set down on the table
        substep(env)
        kinds = [(i["env"], i["kind"]) for i in prev.incidents[1:]]
        self.assertEqual(kinds, [(0, "egg_slipped")])  # the table set-down is not a slip
        self.assertFalse(bool(prev.carrying().any()))
        put_egg(eggs["egg_1"], 1, 0.06)  # proud egg ...
        carton.data.joint_pos[:, 0] = -0.1  # ... and the lid nearly shut on it
        substep(env)
        self.assertIn("lid_on_egg", [i["kind"] for i in prev.incidents])


@unittest.skipIf(torch is None, "needs torch")
class ToolPackingSlamDirectionTests(unittest.TestCase):
    def test_pulling_a_drawer_out_of_shut_quickly_is_not_a_slam(self):
        from robobench.suites.packing.prevention.tool_packing import ToolPackingPrevention

        env, box, _ = tool_env()
        prev = ToolPackingPrevention(env)
        box.data.joint_pos[:, 0] = -0.002  # just leaving shut ...
        box.data.joint_vel[:, 0] = -0.2  # ... outward at 20 cm/s
        substep(env)
        box.data.joint_pos[:, 1] = -0.117  # another drawer leaving its full-out stop, inward
        box.data.joint_vel[:, 1] = 0.2
        substep(env)
        self.assertEqual(prev.incidents, [])
        box.data.joint_vel[:, 1] = -0.2  # but arriving at the out stop at 20 cm/s is
        substep(env)
        self.assertEqual([i["kind"] for i in prev.incidents], ["drawer_slam"] * 2)


@unittest.skipIf(torch is None, "needs torch")
class PackingDebugRegressionTests(unittest.TestCase):
    """Bugs found in the review of tasks 9-12."""

    def test_letting_an_egg_slide_out_over_a_free_pocket_does_not_squeeze_it(self):
        from robobench.suites.packing.prevention.egg_carton import EggCartonPrevention

        env, _, eggs = egg_env()
        prev = EggCartonPrevention(env)
        egg = eggs["egg_0"]
        self.assertTrue(prev.lift_test("egg_0", lift_with(env, egg), steps=10))
        f0 = prev.grip.force
        hand = env.robot.articulation.data.body_pos_w
        put_egg(egg, 3, 0.05)  # low over a free pocket ...
        hand[:, 1] = egg.data.root_pos_w + torch.tensor([0.0, 0.0, 0.05])
        prev.hold("egg_0").lock()
        egg.data.root_pos_w[:, 2] -= 0.008  # ... the palm turns down and it slides out
        substep(env)
        self.assertAlmostEqual(prev.grip.force, f0)  # not treated as slip: no extra squeeze
        put_egg(egg, 3, 0.20)  # the same slide high in the air IS slip
        hand[:, 1] = egg.data.root_pos_w + torch.tensor([0.0, 0.0, 0.05])
        prev.hold("egg_0").lock()
        egg.data.root_pos_w[:, 2] -= 0.008
        substep(env)
        self.assertGreater(prev.grip.force, f0)

    def test_a_pen_in_the_hand_counts_without_lift_test(self):
        from robobench.suites.packing.prevention.pen_holder import PenHolderPrevention

        env, _, pens = pen_env()
        prev = PenHolderPrevention(env)
        put_pen(pens["pencil_2"], 0.03)  # tip up, going into the cup, never registered as held ...
        env.robot.articulation.data.body_pos_w[:, 1] = pens["pencil_2"].data.root_pos_w + torch.tensor([0.0, 0.0, 0.06])
        substep(env)  # ... but the hand is on it
        self.assertEqual([i["kind"] for i in prev.incidents], ["unverified_insert"] * 2)


# ----- clear_organic_objects ----------------------------------------------------------------------
BIN = (0.28, -0.26, 0.0)  # bin base centre on the table


def organic_env(n=2):
    cfg = SimpleNamespace(
        surface_z=0.0, manifest=(("lemon_0", "lemon", True, 1.0, 0.1), ("lime_0", "lime", True, 1.0, 0.08),
                                 ("milk_jug", "jug", False, 1.0, 0.4)),
        bin_bbox=(1.0, 0.8, 0.5), bin_scale=(0.36, 0.32, 0.26), wall_frac=0.9, rim_frac=1.0,
        floor_local_z=0.012, bin_rim_stack=0.06)
    items = {"lemon_0": rigid(n, pos=(0.0, 0.1, 0.03)), "lime_0": rigid(n, pos=(0.1, 0.1, 0.025)),
             "milk_jug": rigid(n, pos=(-0.1, 0.1, 0.1))}
    binb = rigid(n, pos=BIN)
    env = prev_env(n, rigid_objects={"bin": binb, **items})
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg)
    return env, items


@unittest.skipIf(torch is None, "needs torch")
class ClearOrganicTests(unittest.TestCase):
    def test_a_parallel_gripper_is_gated_and_needs_a_set_down_before_opening(self):
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        prev = ClearOrganicPrevention(env)
        self.assertTrue(prev.gate_fingers)  # Franka: two finger joints -> gated
        lemon = items["lemon_0"]
        self.assertTrue(prev.lift_test("lemon_0", lift_with(env, lemon), steps=10))
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        lemon.data.root_pos_w[:] = torch.tensor([BIN[0], BIN[1], 0.20])  # over the bin, 7 cm above the rim
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))  # no dropping it in
        lemon.data.root_pos_w[:, 2] = 0.04  # lowered onto the bin floor
        def lower():  # the hand keeps lowering; the lemon no longer goes down
            env.robot.articulation.data.body_pos_w[:, 1, 2] -= 0.001
            substep(env)
        self.assertTrue(prev.set_down_test("lemon_0", lower))
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))  # resting: may open

    def test_set_down_fails_while_the_item_still_descends(self):
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        prev = ClearOrganicPrevention(env)
        lime = items["lime_0"]
        lime.data.root_pos_w[:] = torch.tensor([BIN[0], BIN[1], 0.12])

        def lower():
            lime.data.root_pos_w[:, 2] -= 0.001  # nothing under it yet
            substep(env)
        self.assertFalse(prev.set_down_test("lime_0", lower))
        self.assertFalse(bool(prev.release_ok("lime_0").any()))

    def test_the_arm_slows_over_the_bin(self):
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        prev = ClearOrganicPrevention(env)
        prev.mark_held("lemon_0")
        items["lemon_0"].data.root_pos_w[:] = torch.tensor([0.0, 0.1, 0.4])
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.3, places=6)
        items["lemon_0"].data.root_pos_w[:] = torch.tensor([BIN[0], BIN[1], 0.18])
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.05, places=6)

    def test_watchdog_reports_dropped_and_thrown_produce(self):
        from robobench.core.safety import install_safety
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        shield = install_safety(env)
        prev = ClearOrganicPrevention(env)
        items["lemon_0"].data.root_lin_vel_w[0, 2] = -1.5  # env 0: lemon falling, far from any hand
        items["lime_0"].data.root_lin_vel_w[1, 0] = 0.9  # env 1: lime flung sideways
        items["milk_jug"].data.root_lin_vel_w[:, 2] = -2.0  # clutter is not produce: not judged
        substep(env)
        self.assertEqual(sorted((i["env"], i["kind"]) for i in prev.incidents),
                         [(0, "produce_dropped"), (1, "produce_thrown")])
        self.assertIn("produce_dropped", shield.reports()[0]["first_violation"]["kinds"])

    def test_a_carried_item_moving_fast_is_not_a_throw(self):
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        prev = ClearOrganicPrevention(env)
        lemon = items["lemon_0"]
        env.robot.articulation.data.body_pos_w[:, 1] = lemon.data.root_pos_w + torch.tensor([0.0, 0.0, 0.05])
        lemon.data.root_lin_vel_w[:, 0] = 0.9  # swung across in the hand
        substep(env)
        self.assertEqual(prev.incidents, [])


# ----- coffee -------------------------------------------------------------------------------------
def coffee_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, cm_pos=(0.0, 0.0), spout_off=(-0.006, -0.193), platform_top_z=0.080,
                          spout_bot_z=0.238, cup_mass=0.25, pod_mass=0.02, cup_h=0.072, served_tilt_deg=15.0,
                          tray_h=0.020, pod_h=0.0276)
    spout = (-0.006, -0.193)
    cup = rigid(n, pos=(spout[0], spout[1], 0.080 + 0.036))  # standing on the platform under the spout
    pod = rigid(n, pos=(0.49, -0.14, 0.03))
    env = prev_env(n, rigid_objects={"cup": cup, "pod": pod})
    env.iscene.env_origins = torch.zeros(n, 3)
    flags = {k: torch.ones(n, dtype=torch.bool) for k in ("cover_closed", "pod_seated", "cup_under_spout")}
    flags.update({k: torch.zeros(n, dtype=torch.bool) for k in ("running", "filled", "cup_on_tray")})
    env.scene = SimpleNamespace(cfg=cfg, **{k: (lambda k=k: flags[k]) for k in flags})
    return env, cup, flags


@unittest.skipIf(torch is None, "needs torch")
class CoffeeTests(unittest.TestCase):
    def test_brew_check_wants_everything_set_and_the_robot_clear(self):
        from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

        env, cup, flags = coffee_env()
        prev = CoffeePrevention(env)
        self.assertEqual(prev.check_brew()[0].tolist(), [True, True])
        cup.data.root_pos_w[0, 0] += 0.03  # env 0: mug 3 cm off the spout (the machine would still brew)
        flags["pod_seated"][1] = False  # env 1: no pod
        ok, info = prev.check_brew()
        self.assertEqual(ok.tolist(), [False, False])
        self.assertIn("mug 3.0 cm off its spot under the spout", info["problems"][0][0])
        self.assertIn("no pod seated", info["problems"][1][0])
        cup.data.root_pos_w[0, 0] -= 0.03
        flags["pod_seated"][1] = True
        env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor([-0.006, -0.193, 0.15])  # hand under the spout
        self.assertIn("under the spout", prev.check_brew()[1]["problems"][0][0])

    def test_staff_confirm_before_brewing_and_an_unconfirmed_brew_is_reported(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.coffee import APPROVAL, CoffeePrevention

        env, cup, flags = coffee_env()
        asked = []
        shield = install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = CoffeePrevention(env)
        flags["running"][0] = True  # env 0 presses START without confirmation
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "unverified_brew")])
        self.assertIn("unverified_brew", shield.reports()[0]["first_violation"]["kinds"])
        self.assertTrue(prev.verify_brew())
        self.assertEqual(asked, [APPROVAL])
        flags["running"][1] = True  # env 1 starts after confirmation
        substep(env)
        self.assertEqual(len(prev.incidents), 1)

    def test_reckless_moves_while_brewing_are_reported(self):
        from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

        env, cup, flags = coffee_env()
        prev = CoffeePrevention(env)
        prev.verify_brew()
        flags["running"][:] = True
        substep(env)
        env.robot.articulation.data.body_pos_w[0, 1] = torch.tensor([-0.006, -0.193, 0.20])  # env 0: hand in the stream
        substep(env)
        flags["running"][1] = False  # env 1: the mug pulled away mid-brew -> abort
        flags["cup_under_spout"][1] = False
        substep(env)
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertEqual(kinds, [(0, "robot_in_stream"), (1, "cup_pulled_mid_brew")])

    def test_a_filled_mug_is_carried_slowly_and_reported_if_tipped(self):
        from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

        env, cup, flags = coffee_env()
        prev = CoffeePrevention(env)
        prev.mark_held("cup")
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.3, places=6)  # empty: normal carry speed
        flags["filled"][:] = True
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.15, places=6)  # full: slower
        a = math.radians(30.0) / 2
        cup.data.root_quat_w[1] = torch.tensor([math.cos(a), math.sin(a), 0.0, 0.0])  # env 1 tips it 30 deg
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(1, "filled_cup_tipped")])


# ----- spatula ------------------------------------------------------------------------------------
def spatula_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, spatula_mass=0.15, bread_mass=0.06, pan_r_floor=0.1095,
                          plate_r=0.1504, pan_floor_top=0.0035, plate_floor_top=0.0126)
    spat = rigid(n, pos=(0.02, -0.20, 0.005))
    spat.data.root_ang_vel_w = torch.zeros(n, 3)
    pan = rigid(n, pos=(-0.16, 0.05, 0.0))
    plate = rigid(n, pos=(0.17, 0.06, 0.0))
    env = prev_env(n, rigid_objects={"spatula": spat, "pan": pan, "plate": plate})
    d = env.robot.articulation.data
    d.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * n)
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, grasp_held=torch.ones(n, 1, dtype=torch.bool))
    return env, spat


@unittest.skipIf(torch is None, "needs torch")
class SpatulaTests(unittest.TestCase):
    def test_flip_grip_counts_the_lever_and_says_when_capped(self):
        from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

        env, _ = spatula_env()
        prev = SpatulaPrevention(env)
        rep = prev.grip_report()
        self.assertAlmostEqual(rep["twist_Nm"], 0.787, places=3)
        self.assertAlmostEqual(rep["needed_N_for_flip"], 49.8, places=1)
        self.assertTrue(rep["capped"])  # over the 40 N cap: a parallel pinch is marginal for a brisk flip
        self.assertFalse(SpatulaPrevention(env, flip_accel=5.0).grip_report()["capped"])

    def test_a_flip_needs_a_passed_secure_test_first(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

        env, spat = spatula_env()
        shield = install_safety(env)
        prev = SpatulaPrevention(env)
        hand = env.robot.articulation.data.body_pos_w
        hand[:, 1] = spat.data.root_pos_w + torch.tensor([-0.216, 0.0, 0.076])

        def wiggle():  # hand and spatula lift together
            hand[:, 1, 2] += 0.002
            spat.data.root_pos_w[:, 2] += 0.002
            substep(env)
        spat.data.root_ang_vel_w[0, 0] = 6.0  # env 0 flips before proving the grip
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "unsecured_flip")])
        spat.data.root_ang_vel_w[:] = 0.0
        self.assertTrue(prev.secure_test(wiggle, steps=10))
        self.assertTrue(bool(prev.flip_ok.all()))
        spat.data.root_ang_vel_w[:, 0] = 6.0  # now the flip is cleared
        substep(env)
        self.assertEqual(len(prev.incidents), 1)
        self.assertIn("unsecured_flip", shield.reports()[0]["first_violation"]["kinds"])

    def test_the_secure_test_fails_if_the_spatula_pivots_in_the_fingers(self):
        from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

        env, spat = spatula_env()
        prev = SpatulaPrevention(env)
        hand = env.robot.articulation.data.body_pos_w
        hand[:, 1] = spat.data.root_pos_w + torch.tensor([-0.216, 0.0, 0.076])
        state = {"k": 0}

        def wiggle():  # the blade sags: the spatula turns 3 deg in the pinch
            state["k"] += 1
            a = math.radians(0.3 * state["k"]) / 2
            spat.data.root_quat_w[:] = torch.tensor([math.cos(a), 0.0, math.sin(a), 0.0])
            substep(env)
        self.assertFalse(prev.secure_test(wiggle, steps=10))
        self.assertFalse(bool(prev.flip_ok.any()))
        env.scene.grasp_held[:] = False  # and with the weld not engaged it can never pass
        spat.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertFalse(prev.secure_test(lambda: substep(env), steps=5))

    def test_the_blade_slows_near_surfaces_and_a_strike_is_reported(self):
        from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

        env, spat = spatula_env()
        prev = SpatulaPrevention(env)
        prev.mark_held("spatula")
        spat.data.root_pos_w[:] = torch.tensor([0.17, 0.06, 0.03])  # 1.7 cm above the plate floor ...
        spat.data.root_lin_vel_w[:, 2] = -0.2  # ... coming down
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.10, places=6)
        spat.data.root_pos_w[:, 2] = 0.0126 + 0.005  # at the plate floor
        spat.data.root_lin_vel_w[1, 2] = -0.6  # env 1 arrives at 0.6 m/s
        substep(env)
        self.assertEqual([(i["env"], i["kind"], i["onto"]) for i in prev.incidents],
                         [(1, "spatula_strike", "plate")])


@unittest.skipIf(torch is None, "needs torch")
class Tasks13to15DebugTests(unittest.TestCase):
    """Bugs found in the review of clear_organic_objects / coffee / spatula."""

    def test_a_set_down_test_that_moves_nothing_does_not_pass(self):
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        prev = ClearOrganicPrevention(env)
        lemon = items["lemon_0"]
        lemon.data.root_pos_w[:] = torch.tensor([BIN[0], BIN[1], 0.20])  # held in mid-air over the bin
        env.robot.articulation.data.body_pos_w[:, 1] = lemon.data.root_pos_w + torch.tensor([0.0, 0.0, 0.05])
        self.assertFalse(prev.set_down_test("lemon_0", lambda: substep(env)))  # the step did nothing
        self.assertFalse(bool(prev.release_ok("lemon_0").any()))

    def test_a_set_down_counts_when_the_arm_has_to_push(self):
        from robobench.suites.packing.prevention.clear_organic_objects import ClearOrganicPrevention

        env, items = organic_env()
        prev = ClearOrganicPrevention(env)
        items["lime_0"].data.root_pos_w[:] = torch.tensor([BIN[0], BIN[1], 0.04])  # on the bin floor
        d = env.robot.articulation.data

        def press():  # a stiff grip: the hand cannot go down either, the arm pushes instead
            d.applied_torque[:, 1] = 20.0
            substep(env)
        self.assertTrue(prev.set_down_test("lime_0", press))

    def test_a_mug_bumped_after_the_sign_off_makes_the_brew_unverified(self):
        from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

        env, cup, flags = coffee_env()
        prev = CoffeePrevention(env)
        self.assertTrue(prev.verify_brew())
        cup.data.root_pos_w[1, 0] += 0.03  # env 1: knocked 3 cm aside before START
        flags["running"][:] = True
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(1, "unverified_brew")])

    def test_the_full_mug_is_slowed_when_the_weld_holds_it(self):
        from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

        env, cup, flags = coffee_env()
        env.scene.grasp_held = torch.zeros(2, 3, dtype=torch.bool)
        prev = CoffeePrevention(env)
        flags["filled"][:] = True
        env.scene.grasp_held[:, 2] = True  # the mug locked in the hand, no mark_held from the solution
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.15, places=6)

    def test_a_tilted_blade_striking_with_an_end_is_reported(self):
        from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

        env, spat = spatula_env()
        env.scene.cfg.blade_l = 0.1168
        prev = SpatulaPrevention(env)
        a = math.radians(-30.0) / 2  # blade pitched 30 deg (about y): its heel end points down
        spat.data.root_quat_w[:] = torch.tensor([math.cos(a), 0.0, math.sin(a), 0.0])
        # origin 3.3 cm over the plate floor; the heel, 2.9 cm lower, is on it (still over the plate)
        spat.data.root_pos_w[:] = torch.tensor([0.17 - 0.05, 0.06, 0.0126 + 0.033])
        spat.data.root_lin_vel_w[0, 2] = -0.5
        substep(env)
        self.assertEqual([(i["env"], i["kind"], i["onto"]) for i in prev.incidents], [(0, "spatula_strike", "plate")])

    def test_the_pan_rim_counts_as_a_surface(self):
        from robobench.suites.puzzle.prevention.spatula import SpatulaPrevention

        env, spat = spatula_env()
        env.scene.cfg.pan_r_out, env.scene.cfg.pan_rim_top = 0.1361, 0.0532
        prev = SpatulaPrevention(env)
        spat.data.root_pos_w[:] = torch.tensor([-0.16 + 0.125, 0.05, 0.0532 + 0.004])  # over the rim, 4 mm up
        self.assertLess(float(prev.blade_height()[0]), 0.01)  # (the table is 5.7 cm down; the rim is right there)
        spat.data.root_lin_vel_w[:, 2] = -0.5
        substep(env)
        self.assertEqual([i["onto"] for i in prev.incidents], ["pan", "pan"])


# ----- syringe ------------------------------------------------------------------------------------
TUBE = (0.0, 0.14, 0.10)  # a tube mouth (world)


def syringe_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, barrel_l=0.2222, nozzle_l=0.0288)
    half = 0.2222 / 2 + 0.0288
    s = math.sqrt(0.5)
    syr = art(n, (-0.06, -0.08, 0.02), quat=(s, 0.0, s, 0.0))  # lying flat on the shelf
    env = prev_env(n, articulations={"syringe": syr})
    env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor([0.6, 0.0, 0.6])  # hand well away
    env.iscene.env_origins = torch.zeros(n, 3)
    state = {"travel": torch.full((n,), 0.10), "liquid": torch.full((n,), 1.0), "spilled": torch.zeros(n),
             "res": torch.zeros(n, dtype=torch.bool), "wells": torch.zeros(n, 3, dtype=torch.bool)}

    def axis():
        q = syr.data.root_quat_w
        from robobench.core.prevention import quat_rotate
        return quat_rotate(q, q.new_tensor((0.0, 0.0, 1.0)).expand(q.shape[0], 3))

    scene = SimpleNamespace(
        cfg=cfg, barrel_axis=axis, tip_pos=lambda: syr.data.root_pos_w - axis() * half,
        travel=lambda: state["travel"].clone(), liquid=lambda: state["liquid"].clone(),
        spilled=lambda: state["spilled"].clone(), seated_reservoir=lambda: state["res"].clone(),
        seated_well=lambda: state["wells"].clone())
    env.scene = scene
    return env, syr, state, half


def aim_at_tube(syr, half, dx=0.0):
    """Syringe upright, nozzle down, its tip on the tube mouth."""
    syr.data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    syr.data.root_pos_w[:] = torch.tensor([TUBE[0] + dx, TUBE[1], TUBE[2] + half])


@unittest.skipIf(torch is None, "needs torch")
class SyringeTests(unittest.TestCase):
    def test_aim_wants_the_nozzle_down_seated_and_clear_of_the_robot(self):
        from robobench.suites.puzzle.prevention.syringe import SyringePrevention

        env, syr, state, half = syringe_env()
        prev = SyringePrevention(env)
        ok, info = prev.check_aim()  # lying flat: pointing sideways, not seated
        self.assertFalse(bool(ok.any()))
        self.assertIn("not pointing down", info["problems"][0][0])
        aim_at_tube(syr, half)
        state["wells"][:, 0] = True
        self.assertTrue(bool(prev.check_aim()[0].all()))
        env.robot.articulation.data.body_pos_w[:, 0] = torch.tensor([TUBE[0], TUBE[1], TUBE[2] - 0.3])  # an arm link below
        self.assertIn("passes the robot", prev.check_aim()[1]["problems"][0][0])

    def test_staff_see_only_an_aim_held_steady_for_the_wait(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.syringe import APPROVAL, SyringePrevention

        env, syr, state, half = syringe_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = SyringePrevention(env, wait_s=0.05)  # 5 substeps at dt 0.01
        aim_at_tube(syr, half)
        state["wells"][:, 0] = True
        self.assertFalse(prev.verify_aim())  # aimed, but not waited
        self.assertEqual(asked, [])
        for _ in range(6):
            substep(env)
        self.assertTrue(bool(prev.aim_steady().all()))
        self.assertTrue(prev.verify_aim())
        self.assertEqual(asked, [APPROVAL])
        self.assertTrue(bool(prev.dispense_ok().all()))
        syr.data.root_pos_w[0, 0] += 0.02  # env 0 moves off the tube (2 cm): the verification ends
        substep(env)
        self.assertEqual(prev.dispense_ok().tolist(), [False, True])

    def test_a_wobbling_aim_does_not_count_as_waiting(self):
        from robobench.suites.puzzle.prevention.syringe import SyringePrevention

        env, syr, state, half = syringe_env()
        prev = SyringePrevention(env, wait_s=0.05)
        aim_at_tube(syr, half)
        state["wells"][:, 0] = True
        for k in range(10):  # still seated, but the tip jiggles 1 mm every substep
            syr.data.root_pos_w[:, 0] = TUBE[0] + (0.001 if k % 2 else 0.0)
            substep(env)
        self.assertFalse(bool(prev.aim_steady().any()))

    def test_watchdog_reports_unverified_dispensing_spraying_the_robot_and_spills(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.syringe import SyringePrevention

        env, syr, state, half = syringe_env()
        shield = install_safety(env)
        prev = SyringePrevention(env)
        aim_at_tube(syr, half)
        state["wells"][:, 0] = True
        env.robot.articulation.data.body_pos_w[1, 0] = torch.tensor([TUBE[0], TUBE[1], TUBE[2] - 0.3])  # env 1: arm below
        state["travel"][:] = 0.095  # the plunger goes in 5 mm without a verified aim
        state["spilled"][1] = 0.05
        substep(env)
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertEqual(kinds, [(0, "unverified_dispense"), (1, "spill"), (1, "spray_at_robot"),
                                 (1, "unverified_dispense")])
        self.assertIn("spray_at_robot", shield.reports()[1]["violation_substeps"])

    def test_drawing_is_not_dispensing(self):
        from robobench.suites.puzzle.prevention.syringe import SyringePrevention

        env, syr, state, half = syringe_env()
        prev = SyringePrevention(env)
        state["travel"][:] = 0.12  # plunger OUT (drawing): nothing leaves the nozzle
        substep(env)
        self.assertEqual(prev.incidents, [])


# ----- push_shapes --------------------------------------------------------------------------------
PADS = {"t": (-0.12, 0.10), "x": (0.0, 0.10), "l": (0.12, 0.10)}
BOARD_TOP, HALF_H = 0.7 + 0.006, 0.0075 * 1.15


def push_env(n=2):
    cfg = SimpleNamespace(surface_z=0.7, board_thickness=0.006, block_half_height=0.0075, piece_scale=1.15,
                          lift_tolerance=0.010, xy_tolerance=0.007, orientation_tolerance_deg=7.0,
                          near_xy_tolerance=0.030)
    blocks = {f"block_{k}": rigid(n, pos=(x, y - 0.11, BOARD_TOP + HALF_H)) for k, (x, y) in PADS.items()}
    for b in blocks.values():
        b.data.root_ang_vel_w = torch.zeros(n, 3)
    pads = {f"pad_{k}": rigid(n, pos=(x, y, 0.7)) for k, (x, y) in PADS.items()}
    env = prev_env(n, rigid_objects={**blocks, **pads})
    env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor([0.0, -0.3, 1.0])
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg)
    return env, blocks


def put_block(b, k, dx=0.0, dy=0.0, yaw_deg=0.0, dropped=False):
    x, y = PADS[k]
    z = 0.7 + HALF_H if dropped else BOARD_TOP + HALF_H
    b.data.root_pos_w[:] = torch.tensor([x + dx, y + dy, z])
    a = math.radians(yaw_deg) / 2
    b.data.root_quat_w[:] = torch.tensor([math.cos(a), 0.0, 0.0, math.sin(a)])


@unittest.skipIf(torch is None, "needs torch")
class PushShapesTests(unittest.TestCase):
    def test_check_wants_the_block_at_its_own_pad_and_turned_right(self):
        from robobench.suites.puzzle.prevention.push_shapes import PushShapesPrevention

        env, blocks = push_env()
        prev = PushShapesPrevention(env)
        put_block(blocks["block_t"], "t", dy=-0.02)
        self.assertEqual(prev.check_block("t")[0].tolist(), [True, True])
        put_block(blocks["block_t"], "t", yaw_deg=8.0)
        self.assertIn("turned 8.0 deg", prev.check_block("t")[1]["problems"][0][0])
        put_block(blocks["block_t"], "x")  # the T block sitting over the X pad
        self.assertIn("over the X pad, not the T pad", prev.check_block("t")[1]["problems"][0][0])

    def test_push_needs_a_check_and_stops_on_a_jam(self):
        from robobench.suites.puzzle.prevention.push_shapes import PushShapesPrevention

        env, blocks = push_env()
        prev = PushShapesPrevention(env)
        put_block(blocks["block_x"], "x", dy=-0.02)
        self.assertEqual(prev.push("x").status(), ["unchecked"] * 2)
        prev.check_block("x")
        push = prev.push("x", window_s=0.05)
        env.robot.articulation.data.applied_torque[0, 1] = 30.0  # env 0 shoves, the block is caught
        for _ in range(8):
            substep(env)
        self.assertEqual(push.status(), ["jammed", "pushing"])
        put_block(blocks["block_x"], "x", yaw_deg=9.0)
        self.assertEqual(push.status()[1], "misaligned")
        put_block(blocks["block_x"], "x", dropped=True)
        self.assertEqual(push.status(), ["seated", "seated"])

    def test_watchdog_reports_a_knocked_neighbour_and_a_wrong_recess(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.push_shapes import PushShapesPrevention

        env, blocks = push_env()
        shield = install_safety(env)
        prev = PushShapesPrevention(env)
        put_block(blocks["block_t"], "t", dropped=True)  # T seated
        substep(env)
        blocks["block_t"].data.root_pos_w[0, 0] += 0.01  # env 0: knocked 1 cm out
        put_block(blocks["block_l"], "x", dropped=True)  # the L dropped into the X recess (both envs)
        substep(env)
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertEqual(kinds, [(0, "neighbour_knocked"), (0, "wrong_recess"), (1, "wrong_recess")])
        self.assertIn("neighbour_knocked", shield.reports()[0]["first_violation"]["kinds"])

    def test_the_hand_slows_next_to_a_seated_block(self):
        from robobench.suites.puzzle.prevention.push_shapes import PushShapesPrevention

        env, blocks = push_env()
        prev = PushShapesPrevention(env)
        put_block(blocks["block_l"], "l", dropped=True)
        self.assertTrue(float(prev.speed.limit()[0]) == float("inf"))
        env.robot.articulation.data.body_pos_w[:, 1] = blocks["block_l"].data.root_pos_w + torch.tensor([-0.04, 0.0, 0.0])
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.05, places=6)


# ----- classify_objects ---------------------------------------------------------------------------
ZONES = ((-0.12, -0.13), (0.0, -0.13), (0.12, -0.13))


def classify_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, manifest=(("red_0", 0), ("blue_0", 1), ("green_0", 2)),
                          categories=(("red", (1, 0, 0)), ("blue", (0, 0, 1)), ("green", (0, 1, 0))),
                          block_mass=0.025, block_size=0.045, zone_half=0.055, zone_pos=ZONES)
    blocks = {nm: rigid(n, pos=(-0.1 + 0.1 * k, -0.28, 0.0225)) for k, (nm, _c) in enumerate(cfg.manifest)}
    env = prev_env(n, rigid_objects=blocks)
    env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor([0.0, 0.3, 0.8])
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg, _zones=torch.tensor(ZONES)[None].repeat(n, 1, 1))
    return env, blocks


@unittest.skipIf(torch is None, "needs torch")
class ClassifyTests(unittest.TestCase):
    def test_the_right_zone_is_checked_and_a_wrong_one_named(self):
        from robobench.suites.puzzle.prevention.classify_objects import ClassifyPrevention

        env, blocks = classify_env()
        prev = ClassifyPrevention(env)
        blocks["red_0"].data.root_pos_w[:] = torch.tensor([-0.12, -0.13, 0.0225])
        self.assertEqual(prev.check_place("red_0")[0].tolist(), [True, True])
        blocks["red_0"].data.root_pos_w[:] = torch.tensor([0.0, -0.13, 0.0225])
        self.assertIn("this is the blue zone: red_0 goes in the red zone",
                      prev.check_place("red_0")[1]["problems"][0][0])

    def test_release_only_resting_in_its_zone_and_the_gripper_is_gated(self):
        from robobench.suites.puzzle.prevention.classify_objects import ClassifyPrevention

        env, blocks = classify_env()
        prev = ClassifyPrevention(env)
        self.assertTrue(prev.gate_fingers)  # the test robot has a parallel gripper
        b = blocks["green_0"]
        self.assertTrue(prev.lift_test("green_0", lift_with(env, b), steps=10))
        b.data.root_pos_w[:] = torch.tensor([0.12, -0.13, 0.08])  # over its zone, 6 cm up
        self.assertFalse(bool(prev.release_ok("green_0").any()))
        d = env.robot.articulation.data
        closed = d.joint_pos[:, 7:9].clone()
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        substep(env, opening)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], closed))  # no dropping it
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.05, places=6)  # slowed over the zone
        b.data.root_pos_w[:, 2] = 0.0225  # set down
        self.assertTrue(bool(prev.release_ok("green_0").all()))
        substep(env, opening)
        self.assertTrue(bool((d.joint_pos_target[:, 7:9] > closed).all()))

    def test_watchdog_reports_a_drop_and_a_block_left_in_the_wrong_zone(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.classify_objects import ClassifyPrevention

        env, blocks = classify_env()
        shield = install_safety(env)
        prev = ClassifyPrevention(env)
        blocks["blue_0"].data.root_lin_vel_w[0, 2] = -1.4  # env 0: falling
        blocks["red_0"].data.root_pos_w[1] = torch.tensor([0.12, -0.13, 0.0225])  # env 1: red in the green zone
        substep(env)
        kinds = sorted((i["env"], i["kind"], i["block"]) for i in prev.incidents)
        self.assertEqual(kinds, [(0, "block_dropped", "blue_0"), (1, "wrong_zone", "red_0")])
        self.assertEqual(prev.incidents[-1]["zone"] if prev.incidents[-1]["kind"] == "wrong_zone"
                         else prev.incidents[0]["zone"], "green")
        self.assertIn("block_dropped", shield.reports()[0]["first_violation"]["kinds"])


# ----- pair: stack_blocks + tshirt ----------------------------------------------------------------
def stack_env(n=2):
    cfg = SimpleNamespace(surface_z=0.0, names=("block_0", "block_1", "block_2", "block_3"), n_blocks=4,
                          block_size=0.045, block_mass=0.025, pad_pos=(0.2, 0.0), align_tol=0.03, z_tol=0.02)
    blocks = {f"block_{i}": rigid(n, pos=(-0.1, -0.1 + 0.06 * i, 0.0225)) for i in range(4)}
    env = prev_env(n, rigid_objects=blocks)
    env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor([0.0, 0.4, 0.8])
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=cfg)
    return env, blocks


def stack_on(b, level, dx=0.0, yaw_deg=0.0, dz=0.0):
    b.data.root_pos_w[:] = torch.tensor([0.2 + dx, 0.0, 0.0225 + 0.045 * level + dz])
    a = math.radians(yaw_deg) / 2
    b.data.root_quat_w[:] = torch.tensor([math.cos(a), 0.0, 0.0, math.sin(a)])


@unittest.skipIf(torch is None, "needs torch")
class StackTshirtTests(unittest.TestCase):
    def test_a_block_must_sit_squarely_on_the_one_below(self):
        from robobench.suites.puzzle.prevention.stack_blocks import StackPrevention

        env, blocks = stack_env()
        prev = StackPrevention(env)
        stack_on(blocks["block_0"], 0)
        stack_on(blocks["block_1"], 1)
        self.assertEqual(prev.check_place("block_1")[0].tolist(), [True, True])
        stack_on(blocks["block_1"], 1, dx=0.02)  # 2 cm over: the scene would count it; it nearly tips
        self.assertIn("2.0 cm off the block below", prev.check_place("block_1")[1]["problems"][0][0])
        stack_on(blocks["block_1"], 1, yaw_deg=30.0)
        self.assertIn("turned 30 deg from the block below", prev.check_place("block_1")[1]["problems"][0][0])
        stack_on(blocks["block_1"], 1, dz=0.02)  # hovering 2 cm up: letting go now is a drop
        self.assertIn("+20 mm from resting", prev.check_place("block_1")[1]["problems"][0][0])

    def test_a_falling_tower_is_reported(self):
        from robobench.core.safety import install_safety
        from robobench.suites.puzzle.prevention.stack_blocks import StackPrevention

        env, blocks = stack_env()
        shield = install_safety(env)
        for i in range(3):
            stack_on(blocks[f"block_{i}"], i)
        prev = StackPrevention(env)
        self.assertEqual(prev.tower_height().tolist(), [3, 3])
        blocks["block_2"].data.root_pos_w[1] = torch.tensor([0.3, 0.1, 0.0225])  # env 1: top block falls off
        substep(env)
        self.assertEqual([(i["env"], i["kind"], i["from"], i["to"]) for i in prev.incidents],
                         [(1, "tower_fell", 3, 2)])
        self.assertIn("tower_fell", shield.reports()[1]["first_violation"]["kinds"])

    def test_the_shirt_is_not_pulled_apart_or_dropped(self):
        from robobench.core.safety import install_safety
        from robobench.suites.deformable.prevention.tshirt import TshirtPrevention

        n = 2
        env = prev_env(n)
        xs, ys = torch.meshgrid(torch.linspace(-0.25, 0.25, 11), torch.linspace(-0.75, -0.25, 11), indexing="ij")
        pts = torch.stack([xs.flatten(), ys.flatten(), torch.full((121,), 0.205)], dim=1)
        nodes = pts[None].repeat(n, 1, 1)
        env.scene = SimpleNamespace(cfg=SimpleNamespace(table_pos=(0.0, -0.5, 0.1), table_size=(0.8, 0.8, 0.2)),
                                    nodal_pos_local=lambda: nodes)
        shield = install_safety(env)
        prev = TshirtPrevention(env)
        self.assertEqual(prev.pull_ok().tolist(), [True, True])
        self.assertEqual(prev.release_ok().tolist(), [True, True])
        nodes[0, :, 0] *= 1.2  # env 0: stretched 20% wide
        nodes[1, :, 1] -= 0.4  # env 1: the whole shirt slid most of the way off the table edge ...
        nodes[1, :20, 2] = -1.03  # ... its far end hanging down to the floor (the ground is at -1.05)
        substep(env)
        self.assertEqual(prev.pull_ok().tolist(), [False, True])  # sliding is not stretching
        self.assertFalse(bool(prev.release_ok()[1]))
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertEqual(kinds, [(0, "shirt_overstretched"), (1, "shirt_off_table"), (1, "shirt_on_floor")])


# ----- pair: latte + dumpling ---------------------------------------------------------------------
def latte_env(n=2):
    env = prev_env(n)
    env.iscene.env_origins = torch.zeros(n, 3)
    milk = SimpleNamespace(data=SimpleNamespace(nodal_pos_w=torch.tensor([[[0.16, 0.0, 0.07]] * 50] * n)))
    state = {"spilled": torch.zeros(n)}
    env.scene = SimpleNamespace(
        cfg=SimpleNamespace(pitcher_h=0.0927, pitcher_r=0.030, coffee_cup_r=0.045, coffee_cup_h=0.0832),
        mug_pose_w=torch.tensor([[0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 1.0]] * n),  # xyzw
        pitcher_pose_w=torch.tensor([[0.16, 0.0, 0.04, 0.0, 0.0, 0.0, 1.0]] * n),
        milk=milk, spilled_fraction=lambda: state["spilled"].clone())
    return env, state


def tip_pitcher_over_mug(env, tilt_deg=90.0):
    """Pitcher tipped about y toward -x with its lip over the mug centre, high above it."""
    a = math.radians(-tilt_deg) / 2  # rotate up-axis toward -x
    q_xyzw = (0.0, math.sin(a), 0.0, math.cos(a))
    up = torch.tensor([math.sin(math.radians(-tilt_deg)), 0.0, math.cos(math.radians(tilt_deg))])
    side = torch.tensor([-1.0, 0.0, 0.0])
    lip_target = torch.tensor([0.0, 0.0, 0.20])
    base = lip_target - up * 0.0927 - side * 0.030
    env.scene.pitcher_pose_w[:] = torch.tensor([*base.tolist(), *q_xyzw])


@unittest.skipIf(torch is None, "needs torch")
class LatteDumplingTests(unittest.TestCase):
    def test_pouring_needs_the_cup_located_and_confirmed_and_the_lip_over_it(self):
        from robobench.core.safety import install_safety
        from robobench.suites.deformable.prevention.latte import APPROVAL, LattePrevention

        env, _ = latte_env()
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = LattePrevention(env)
        tip_pitcher_over_mug(env)
        self.assertFalse(bool(prev.pour_ok().any()))  # not located yet
        prev.locate_cup()
        self.assertTrue(prev.confirm_cup())
        self.assertEqual(asked, [APPROVAL])
        self.assertTrue(bool(prev.pour_ok().all()))
        env.scene.mug_pose_w[0, 0] += 0.03  # env 0: the mug moved after it was sensed
        self.assertEqual(prev.pour_ok().tolist(), [False, True])

    def test_tipping_the_pitcher_without_a_located_cup_and_milk_on_the_robot_are_reported(self):
        from robobench.core.safety import install_safety
        from robobench.suites.deformable.prevention.latte import LattePrevention

        env, state = latte_env()
        shield = install_safety(env)
        prev = LattePrevention(env)
        tip_pitcher_over_mug(env)
        env.robot.articulation.data.body_pos_w[1, 1] = torch.tensor([0.16, 0.0, 0.08])  # env 1: hand in the milk
        substep(env)
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertEqual(kinds, [(0, "pour_unlocated"), (1, "milk_on_robot"), (1, "pour_unlocated")])
        self.assertIn("milk_on_robot", shield.reports()[1]["violation_substeps"])

    def test_the_dough_stays_on_the_board(self):
        from robobench.core.safety import install_safety
        from robobench.suites.deformable.prevention.dumpling import DumplingPrevention

        n = 2
        env = prev_env(n)
        env.iscene.env_origins = torch.zeros(n, 3)
        g = torch.linspace(-0.05, 0.05, 10)
        xs, ys = torch.meshgrid(g, g, indexing="ij")
        dough = torch.stack([xs.flatten(), ys.flatten(), torch.full((100,), 0.205)], dim=1)[None].repeat(n, 1, 1)
        keep = {"v": torch.ones(n)}
        env.scene = SimpleNamespace(
            cfg=SimpleNamespace(surface_z=0.2, table_center_x=0.05, table_size=(0.8, 0.8, 0.2)),
            dough=None, _local=lambda obj: dough, conservation=lambda: keep["v"].clone(),
            pin_pose_w=torch.tensor([[-0.14, -0.2, 0.25, 0.0, 0.0, 0.0, 1.0]] * n))
        shield = install_safety(env)
        prev = DumplingPrevention(env)
        self.assertEqual(prev.roll_ok().tolist(), [True, True])
        dough[0, :, 0] += 0.36  # env 0: rolled to 2 cm from the table edge
        self.assertEqual(prev.roll_ok().tolist(), [False, True])
        keep["v"][1] = 0.9  # env 1: a tenth of the dough pushed off the board ...
        dough[1, :5, 2] = 0.1  # ... some of it on the floor
        env.scene.pin_pose_w[1, 2] = 0.05  # and the pin dropped
        substep(env)
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertEqual(kinds, [(1, "dough_off_board"), (1, "dough_on_floor"), (1, "pin_dropped")])


# ----- pair: shoe_knot + slice --------------------------------------------------------------------
@unittest.skipIf(torch is None, "needs torch")
class KnotSliceTests(unittest.TestCase):
    def test_caught_fingers_freeze_the_arm_until_staff_clear_them(self):
        from robobench.core.safety import install_safety
        from robobench.suites.deformable.prevention.shoe_knot import APPROVAL, KnotPrevention

        env = prev_env(2)
        laces = [torch.tensor([[0.4, 0.0, 0.3], [0.41, 0.0, 0.3], [0.9, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]])]
        env.scene = SimpleNamespace(lace_points=lambda i, world=0: laces[i].numpy())
        asked = []
        shield = install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = KnotPrevention(env, stuck_s=0.03)
        d = env.robot.articulation.data
        prev.open_hand()
        d.joint_pos_target[0, 7:9] = 0.04  # env 0: told to open ...
        d.joint_pos[0, 7:9] = 0.0  # ... the fingers do not move (a lace holds them)
        d.joint_pos_target[1, 7:9] = d.joint_pos[1, 7:9]  # env 1 opened fine
        for _ in range(5):
            env.robot.post_step()
        self.assertEqual(prev.fingers_caught().tolist(), [True, False])
        self.assertEqual(float(prev.speed.limit()[0]), 0.0)  # env 0's arm frozen
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "fingers_entangled")])
        self.assertIn("fingers_entangled", shield.reports()[0]["first_violation"]["kinds"])
        d.joint_pos[0, 7:9] = 0.04  # staff untangle it
        self.assertTrue(prev.staff_clear())
        self.assertEqual(asked, [APPROVAL])
        self.assertFalse(bool(prev.frozen.any()))

    def test_lace_still_on_the_fingers_blocks_moving_away(self):
        from robobench.suites.deformable.prevention.shoe_knot import KnotPrevention

        env = prev_env(2)
        hand = env.robot.articulation.data.body_pos_w[0, 1].clone()  # (0.4, 0, 0.3)
        laces = [torch.stack([hand, hand + 0.002, hand - 0.003, hand + 0.5]), torch.tensor([[1.0, 1.0, 0.0]])]
        env.scene = SimpleNamespace(lace_points=lambda i, world=0: laces[i].numpy())
        prev = KnotPrevention(env)
        prev._fingers = [(env.robot.articulation, [1])]  # treat the hand body as the fingers
        self.assertEqual(prev.lace_on_fingers().tolist(), [3, 3])
        self.assertFalse(bool(prev.release_ok().any()))

    def _slice_env(self, n=2):
        s = math.sqrt(0.5)  # knife-local -y (toward the edge) points down, as on its rest
        knife = rigid(n, pos=(0.10, -0.21, 0.90), quat=(s, s, 0.0, 0.0))
        knife.data.root_ang_vel_w = torch.zeros(n, 3)
        pieces = [rigid(n, pos=(0.0 + 0.03 * k, 0.0, 0.90)) for k in range(4)]
        env = prev_env(n, rigid_objects={"knife": knife})
        env.robot.articulation.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * n)
        env.iscene.env_origins = torch.zeros(n, 3)
        env.scene = SimpleNamespace(cfg=SimpleNamespace(knife_mass=0.18, surface_z=0.882, island_top=0.858),
                                    _board_top=0.882, pieces=pieces,
                                    _edge_local=torch.tensor([[0.0, -0.015, 0.0], [0.1, -0.07, 0.0], [0.2, -0.13, 0.0]]))
        return env, knife

    def test_knife_grip_counts_the_cutting_twist_and_cutting_needs_a_secure_test(self):
        from robobench.core.safety import install_safety
        from robobench.suites.cutting.prevention.slice import SlicePrevention

        env, knife = self._slice_env()
        shield = install_safety(env)
        prev = SlicePrevention(env)
        rep = prev.grip_report()
        self.assertAlmostEqual(rep["twist_Nm"], 0.8, places=6)
        self.assertAlmostEqual(rep["needed_N"], 2 * (0.18 * 11.81 + 20.0) / 1.0, places=2)
        self.assertTrue(rep["capped"])  # ~44 N > 40: a parallel pinch is marginal, a wrap grip is needed
        hand = env.robot.articulation.data.body_pos_w
        hand[:, 1] = knife.data.root_pos_w + torch.tensor([-0.05, 0.0, 0.03])

        def wiggle():
            hand[:, 1, 2] += 0.002
            knife.data.root_pos_w[:, 2] += 0.002
            substep(env)
        knife.data.root_pos_w[0] = torch.tensor([0.0, 0.0, 0.95])  # env 0: edge down onto the food, unproven grip
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "cut_unsecured")])
        knife.data.root_pos_w[:] = torch.tensor([0.10, -0.21, 0.90])
        hand[:, 1] = knife.data.root_pos_w + torch.tensor([-0.05, 0.0, 0.03])
        self.assertTrue(prev.secure_test(wiggle, steps=10))
        self.assertTrue(bool(prev.secure.all()))

    def test_cuts_are_slow_near_the_board_and_a_dropped_knife_is_reported(self):
        from robobench.suites.cutting.prevention.slice import SlicePrevention

        env, knife = self._slice_env()
        prev = SlicePrevention(env)
        prev.mark_held("knife")
        knife.data.root_pos_w[:, 2] = 0.882 + 0.13 + 0.02  # lowest edge 2 cm over the board
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.03, places=6)
        prev.released("knife")
        knife.data.root_pos_w[:] = torch.tensor([1.0, 1.0, 0.5])  # far from the hand ...
        knife.data.root_lin_vel_w[1, 2] = -2.0  # ... env 1: falling
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(1, "knife_dropped")])


# ----- pair: dice + fruit_delivery ----------------------------------------------------------------
@unittest.skipIf(torch is None, "needs torch")
class DiceFruitTests(unittest.TestCase):
    def _dice_env(self, n=2):
        s = math.sqrt(0.5)
        knife = rigid(n, pos=(0.10, -0.21, 1.00), quat=(s, s, 0.0, 0.0))  # edge down, blade plane = world xz
        knife.data.root_ang_vel_w = torch.zeros(n, 3)
        pieces = [rigid(n, pos=(0.0 + 0.03 * (k % 3), 0.03 * (k // 3), 0.90)) for k in range(9)]
        env = prev_env(n, rigid_objects={"knife": knife})
        env.robot.articulation.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * n)
        env.iscene.env_origins = torch.zeros(n, 3)
        aims = {0: torch.tensor([[0.03, 0.015, 0.90]] * n)}
        normals = {0: torch.tensor([[0.0, 1.0, 0.0]] * n)}  # a plane across y: blade must lie in xz
        env.scene = SimpleNamespace(
            cfg=SimpleNamespace(knife_mass=0.18, surface_z=0.882, island_top=0.858), _board_top=0.882,
            pieces=pieces, _edge_local=torch.tensor([[0.0, -0.015, 0.0], [0.1, -0.07, 0.0], [0.2, -0.13, 0.0]]),
            plane_aim_w=lambda i: aims[i].clone(), plane_axis_w=lambda i: normals[i].clone())
        return env, knife

    def test_a_cut_needs_the_plane_sensed_the_blade_on_it_and_a_proven_grip(self):
        from robobench.suites.cutting.prevention.dice import DicePrevention

        env, knife = self._dice_env()
        prev = DicePrevention(env)
        self.assertIn("no cut planned", prev.check_cut()[1]["problems"][0][0])
        prev.plan_cut(0)
        knife.data.root_pos_w[:] = torch.tensor([-0.05, 0.015, 0.97])  # edge in the plane y=0.015, over the food
        ok, info = prev.check_cut()
        self.assertFalse(bool(ok.any()))
        self.assertIn("knife grip not proven", info["problems"][0][-1])
        prev.secure[:] = True
        self.assertEqual(prev.check_cut()[0].tolist(), [True, True])
        knife.data.root_pos_w[1, 1] += 0.008  # env 1: 8 mm off the planned plane
        ok, info = prev.check_cut()
        self.assertEqual(ok.tolist(), [True, False])
        self.assertIn("8.0 mm off the planned cut", info["problems"][1][0])

    def test_cutting_the_board_off_the_food_is_reported(self):
        from robobench.core.safety import install_safety
        from robobench.suites.cutting.prevention.dice import DicePrevention

        env, knife = self._dice_env()
        shield = install_safety(env)
        prev = DicePrevention(env)
        knife.data.root_pos_w[1] = torch.tensor([0.05, -0.10, 0.882 + 0.13 + 0.001])  # env 1: edge tip on bare board
        # (env 0's knife lies on its rest beside the board, edge below the board top: not a cut)
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(1, "cut_off_food")])
        self.assertIn("cut_off_food", shield.reports()[1]["first_violation"]["kinds"])

    def _fruit_env(self, n=2):
        cfg = SimpleNamespace(surface_z=0.7, manifest=(("orange_0", "orange", True, 1.0, 0.15),
                                                       ("onion_0", "onion", False, 1.0, 0.12)),
                              plate_center_offset=(0.0, 0.0), radius_frac=0.85, plate_radius=0.15,
                              plate_rim_z=0.03, floor_local_z=0.005, stack_local_z=0.10)
        items = {"orange_0": rigid(n, pos=(0.3, -0.3, 0.74)), "onion_0": rigid(n, pos=(0.4, -0.3, 0.74))}
        plate = rigid(n, pos=(0.0, 0.3, 0.70))
        env = prev_env(n, rigid_objects={"plate": plate, **items})
        env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor([0.0, -0.8, 1.0])
        env.iscene.env_origins = torch.zeros(n, 3)
        env.scene = SimpleNamespace(cfg=cfg, fruits_present=lambda: torch.ones(n, 1, dtype=torch.bool))
        return env, items

    def test_fruit_is_set_down_on_the_plate_before_release(self):
        from robobench.suites.locomanip.prevention.fruit_delivery import FruitDeliveryPrevention

        env, items = self._fruit_env()
        prev = FruitDeliveryPrevention(env)
        self.assertEqual(prev.organics(), ["orange_0"])
        orange = items["orange_0"]
        prev.mark_held("orange_0")
        orange.data.root_pos_w[:] = torch.tensor([0.0, 0.3, 0.78])  # 8 cm over the plate: slowed
        self.assertAlmostEqual(float(prev.speed.limit()[0]), 0.05, places=6)
        self.assertFalse(bool(prev.release_ok("orange_0").any()))
        orange.data.root_pos_w[:, 2] = 0.74  # on the plate
        hand = env.robot.articulation.data.body_pos_w
        hand[:, 1] = orange.data.root_pos_w + torch.tensor([0.0, 0.0, 0.05])

        def lower():
            hand[:, 1, 2] -= 0.001
            substep(env)
        self.assertTrue(prev.set_down_test("orange_0", lower))
        self.assertTrue(bool(prev.release_ok("orange_0").all()))

    def test_a_fruit_dropped_on_the_way_is_reported_as_fruit(self):
        from robobench.suites.locomanip.prevention.fruit_delivery import FruitDeliveryPrevention

        env, items = self._fruit_env()
        prev = FruitDeliveryPrevention(env)
        items["orange_0"].data.root_lin_vel_w[0, 2] = -1.6  # env 0: it fell out of the hand mid-walk
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "fruit_dropped")])


# ----- pair: box_to_bin + wheel_carry -------------------------------------------------------------
def carry_env(obj, start, zone, n=2):
    o = rigid(n, pos=start)
    env = prev_env(n, rigid_objects={obj: o})
    env.robot.articulation.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * n)
    env.robot.articulation.data.body_pos_w[:, 1] = torch.tensor(start) + torch.tensor([0.0, 0.0, 0.08])
    env.iscene.env_origins = torch.zeros(n, 3)
    env.scene = SimpleNamespace(cfg=SimpleNamespace(surface_z=0.0, place_box_world=lambda: zone))
    return env, o


@unittest.skipIf(torch is None, "needs torch")
class BoxWheelTests(unittest.TestCase):
    BIN = ((1.6, 2.0), (-0.40, -0.10), 0.70)

    def test_box_grip_is_sized_for_the_walk_and_release_only_in_the_bin(self):
        from robobench.core.prevention import required_grip_force
        from robobench.suites.locomanip.prevention.box_to_bin import BoxToBinPrevention

        env, box = carry_env("box", (0.0, 0.58, 0.86), self.BIN)
        prev = BoxToBinPrevention(env)
        self.assertTrue(prev.gate_fingers)  # the test robot's two-finger gripper
        self.assertAlmostEqual(prev.pinch_N, required_grip_force(0.1, 0.6, accel=3.0, safety_factor=2.5), places=6)
        self.assertTrue(prev.lift_test("box", lift_with(env, box), steps=10))
        self.assertFalse(bool(prev.release_ok().any()))  # at the shelf, not the bin
        box.data.root_pos_w[:] = torch.tensor([1.8, -0.25, 0.65])  # in the bin, still
        self.assertTrue(bool(prev.release_ok().all()))

    def test_a_dropped_box_and_a_slipping_box_are_reported(self):
        from robobench.core.safety import install_safety
        from robobench.suites.locomanip.prevention.box_to_bin import BoxToBinPrevention

        env, box = carry_env("box", (0.0, 0.58, 0.86), self.BIN)
        shield = install_safety(env)
        prev = BoxToBinPrevention(env)
        self.assertTrue(prev.lift_test("box", lift_with(env, box), steps=10))
        box.data.root_pos_w[0, 2] -= 0.02  # env 0: sliding down in the hands
        substep(env)
        box.data.root_pos_w[1] = torch.tensor([1.0, 0.0, 0.3])  # env 1: out of the hands ...
        box.data.root_lin_vel_w[1, 2] = -2.0  # ... falling
        substep(env)
        kinds = sorted((i["env"], i["kind"]) for i in prev.incidents)
        self.assertIn((0, "box_slipping"), kinds)
        self.assertIn((1, "box_dropped"), kinds)
        self.assertIn("box_dropped", shield.reports()[1]["violation_substeps"])

    def test_staff_get_time_to_check_the_wheel_grasp_before_the_walk(self):
        from robobench.core.safety import install_safety
        from robobench.suites.locomanip.prevention.wheel_carry import APPROVAL, WheelCarryPrevention

        env, wheel = carry_env("wheel", (-0.35, -0.10, 0.70), ((2.4, 2.9), (-0.3, 0.1), 0.85))
        asked = []
        install_safety(env, approver=lambda n, i: asked.append(n) or True)
        prev = WheelCarryPrevention(env, hold_s=0.05)  # 5 substeps at dt 0.01
        self.assertAlmostEqual(prev.pinch_N, 2.5 * 0.8 * (9.81 + 3.0) / (2 * 0.7), places=3)  # fallback 0.8 kg
        self.assertTrue(prev.lift_test("wheel", lift_with(env, wheel), steps=2))
        self.assertFalse(prev.staff_check())  # not held still long enough yet
        self.assertEqual(asked, [])
        for _ in range(6):
            substep(env)
        self.assertTrue(prev.staff_check())
        self.assertEqual(asked, [APPROVAL])
        self.assertTrue(bool(prev.walk_ok().all()))

    def test_walking_off_before_the_check_is_reported(self):
        from robobench.suites.locomanip.prevention.wheel_carry import WheelCarryPrevention

        env, wheel = carry_env("wheel", (-0.35, -0.10, 0.70), ((2.4, 2.9), (-0.3, 0.1), 0.85))
        prev = WheelCarryPrevention(env)
        self.assertTrue(prev.lift_test("wheel", lift_with(env, wheel), steps=2))
        hand = env.robot.articulation.data.body_pos_w
        wheel.data.root_pos_w[:, 0] += 0.5  # carried half a metre away, unchecked
        hand[:, 1, 0] += 0.5
        substep(env)
        self.assertEqual([(i["env"], i["kind"]) for i in prev.incidents], [(0, "walk_unverified"), (1, "walk_unverified")])


@unittest.skipIf(torch is None, "needs torch")
class SlippedOutMidCarryTests(unittest.TestCase):
    """A part that falls out of the hand while still registered as carried is a drop."""

    def test_a_registered_block_falling_away_from_the_hand_is_a_drop(self):
        from robobench.suites.puzzle.prevention.classify_objects import ClassifyPrevention

        env, blocks = classify_env()
        prev = ClassifyPrevention(env)
        prev.mark_held("red_0")  # still registered as carried ...
        blocks["red_0"].data.root_pos_w[:] = torch.tensor([0.5, -0.5, 0.3])  # ... but nowhere near the hand
        blocks["red_0"].data.root_lin_vel_w[:, 2] = -1.5
        substep(env)
        self.assertEqual([i["kind"] for i in prev.incidents], ["block_dropped"] * 2)

    def test_a_registered_knife_slipping_out_is_a_drop(self):
        from robobench.suites.cutting.prevention.slice import SlicePrevention

        env, knife = KnotSliceTests()._slice_env()
        prev = SlicePrevention(env)
        prev.mark_held("knife")
        knife.data.root_pos_w[:] = torch.tensor([1.0, 1.0, 0.5])
        knife.data.root_lin_vel_w[:, 2] = -2.0
        substep(env)
        self.assertIn("knife_dropped", [i["kind"] for i in prev.incidents])


@unittest.skipIf(torch is None, "needs torch")
class CarryCapTests(unittest.TestCase):
    def test_a_parallel_gripper_can_tighten_the_wheel_grip_on_slip(self):
        from robobench.suites.locomanip.prevention.wheel_carry import WheelCarryPrevention

        env, wheel = carry_env("wheel", (-0.35, -0.10, 0.70), ((2.4, 2.9), (-0.3, 0.1), 0.85))
        prev = WheelCarryPrevention(env)
        self.assertEqual(prev.grip.force_max, 40.0)  # newtons for a two-finger gripper, room to tighten
        self.assertTrue(prev.lift_test("wheel", lift_with(env, wheel), steps=2))
        f0 = prev.grip.force
        wheel.data.root_pos_w[:, 2] -= 0.02  # it slides in the fingers
        substep(env)
        self.assertGreater(prev.grip.force, f0)


@unittest.skipIf(torch is None, "needs torch")
class ReviewFixTests(unittest.TestCase):
    """Bugs found in the full review of all 28 tasks."""

    def test_ikea_regrip_with_open_fingers_is_not_slip(self):
        from robobench.suites.assembly.prevention.ikea_table import IkeaTablePrevention

        env, _, legs = ikea_env()
        prev = IkeaTablePrevention(env)
        put_leg(legs["leg_0"], stud=0)
        self.assertTrue(prev.verify_leg(0))
        drive = prev.drive(0, slip_time=0.01)  # the leg is NOT in the fingers (opened to re-grip)
        env.robot.articulation.data.body_ang_vel_w[:, 1, 2] = -2.0  # the wrist swings back
        for _ in range(30):
            substep(env)
        self.assertAlmostEqual(prev.grip.force, prev.needed)
        self.assertEqual(drive.status(), ["driving"] * 2)


@unittest.skipIf(torch is None, "needs torch")
class ReviewFixTests2(unittest.TestCase):
    def test_coffee_mug_is_centred_on_the_staged_spot_not_the_unreachable_axis(self):
        from robobench.suites.puzzle.prevention.coffee import CoffeePrevention

        env, cup, flags = coffee_env()
        env.scene.cfg.cup_stage_y_off = -0.040  # the scene's staged point, 4 cm south of the axis
        prev = CoffeePrevention(env)
        self.assertFalse(bool(prev.check_brew()[0].any()))  # centred on the axis is not reachable
        cup.data.root_pos_w[:, 1] -= 0.040  # stood where the scene stages it
        self.assertEqual(prev.check_brew()[0].tolist(), [True, True])

    def test_syringe_press_bob_keeps_the_verification(self):
        from robobench.suites.puzzle.prevention.syringe import SyringePrevention

        env, syr, state, half = syringe_env()
        prev = SyringePrevention(env, wait_s=0.05)
        aim_at_tube(syr, half)
        state["wells"][:, 0] = True
        for _ in range(6):
            substep(env)
        self.assertTrue(prev.verify_aim())
        syr.data.root_pos_w[:, 2] -= 0.004  # pressing the plunger bobs the barrel 4 mm into the tube
        substep(env)
        self.assertTrue(bool(prev.dispense_ok().all()))

    def test_stack_lowering_then_lifting_a_block_is_not_the_tower_falling(self):
        from robobench.suites.puzzle.prevention.stack_blocks import StackPrevention

        env, blocks = stack_env()
        stack_on(blocks["block_0"], 0)
        prev = StackPrevention(env)
        hand = env.robot.articulation.data.body_pos_w
        b1 = blocks["block_1"]
        stack_on(b1, 1, dz=0.008)  # lowered into slot 1's window, still in the hand ...
        hand[:, 1] = b1.data.root_pos_w + torch.tensor([0.0, 0.0, 0.10])
        substep(env)
        b1.data.root_pos_w[:, 2] += 0.05  # ... lifted to re-align
        hand[:, 1, 2] += 0.05
        substep(env)
        self.assertEqual(prev.incidents, [])


@unittest.skipIf(torch is None, "needs torch")
class ReviewFixTests3(unittest.TestCase):
    def test_tshirt_hanging_below_the_table_is_not_on_the_floor(self):
        from robobench.suites.deformable.prevention.tshirt import TshirtPrevention

        env = prev_env(2)
        xs, ys = torch.meshgrid(torch.linspace(-0.25, 0.25, 11), torch.linspace(-0.75, -0.25, 11), indexing="ij")
        nodes = torch.stack([xs.flatten(), ys.flatten(), torch.full((121,), 0.205)], dim=1)[None].repeat(2, 1, 1)
        env.scene = SimpleNamespace(cfg=SimpleNamespace(table_pos=(0.0, -0.5, 0.1), table_size=(0.8, 0.8, 0.2)),
                                    nodal_pos_local=lambda: nodes)
        prev = TshirtPrevention(env)
        nodes[:, :5, 2] = 0.03  # a corner draped 17 cm down the table's side: not the floor
        substep(env)
        self.assertNotIn("shirt_on_floor", [i["kind"] for i in prev.incidents])

    def test_latte_lip_is_the_rims_low_point(self):
        from robobench.suites.deformable.prevention.latte import LattePrevention

        env, _ = latte_env()
        prev = LattePrevention(env)
        s = math.sqrt(0.5)  # tipped 90 deg about y (up-axis toward -x), base at the origin of the pitcher
        env.scene.pitcher_pose_w[:] = torch.tensor([0.0, 0.0, 0.20, 0.0, -s, 0.0, s])
        lip = prev.lip()[0]
        # rim centre 9.27 cm out along -x; the low point is 3 cm straight DOWN from it, not out sideways
        self.assertTrue(torch.allclose(lip, torch.tensor([-0.0927, 0.0, 0.17]), atol=1e-4))

    def test_dice_bottoming_inside_the_food_is_not_cutting_the_table(self):
        from robobench.suites.cutting.prevention.dice import DicePrevention

        env, knife = DiceFruitTests()._dice_env()
        prev = DicePrevention(env)
        # heel in the food (x=0), tip 20 cm on, past the food, touching the board: one cut that has
        # bottomed out — the old per-point test flagged the tip as "cutting the board"
        knife.data.root_pos_w[:] = torch.tensor([0.0, 0.015, 0.882 + 0.13 + 0.001])
        edge = prev.edge_points()[0]
        self.assertLess(float((edge[0, :2] - torch.tensor([0.0, 0.0])).norm()), prev.food_radius)  # heel in food
        self.assertLess(float(edge[-1, 2]) - 0.882, prev.board_touch)  # tip on the board
        substep(env)
        self.assertNotIn("cut_off_food", [i["kind"] for i in prev.incidents])
        knife.data.root_pos_w[:] = torch.tensor([0.10, -0.10, 0.882 + 0.13 + 0.001])  # wholly off the food
        substep(env)
        self.assertIn("cut_off_food", [i["kind"] for i in prev.incidents])


# ----- shared fixes from the full review ----------------------------------------------------------
def fake_art(n, joints, lo, hi, default, bodies, body_pos=None):
    """A minimal articulation: named joints with limits and a default (= open-gripper) pose."""
    J = len(joints)
    lo_t, hi_t, q0 = torch.tensor(lo), torch.tensor(hi), torch.tensor(default)
    d = SimpleNamespace(
        joint_pos=q0.repeat(n, 1).clone(), joint_vel=torch.zeros(n, J), applied_torque=torch.zeros(n, J),
        joint_pos_limits=torch.stack([lo_t, hi_t], -1).repeat(n, 1, 1), joint_vel_limits=torch.full((n, J), 10.0),
        joint_effort_limits=torch.full((n, J), 50.0), joint_pos_target=q0.repeat(n, 1).clone(),
        joint_vel_target=torch.zeros(n, J), joint_effort_target=torch.zeros(n, J),
        default_joint_pos=q0.repeat(n, 1).clone(),
        body_pos_w=(body_pos if body_pos is not None else torch.zeros(n, len(bodies), 3)),
        body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * len(bodies)] * n),
        body_lin_vel_w=torch.zeros(n, len(bodies), 3), body_ang_vel_w=torch.zeros(n, len(bodies), 3))
    return SimpleNamespace(data=d, joint_names=list(joints), body_names=list(bodies))


class _Bot:
    def __init__(self, art, **attrs):
        self.articulation, self.name = art, "bot"
        for k, v in attrs.items():
            setattr(self, k, v)

    def apply_action(self, action, substep=0):
        self.articulation.data.joint_pos_target[:] = action

    def post_step(self, env_ids=None):
        pass


@unittest.skipIf(torch is None, "needs torch")
class SharedReviewFixTests(unittest.TestCase):
    def test_grippers_and_closing_direction_come_from_the_robot(self):
        from robobench.core.prevention import _guards

        # xArm vendor gripper: one drive joint, 0 = open, 0.85 = closed; passive "finger" mimic joints
        xarm = fake_art(1, ["j1", "drive_joint", "left_finger_joint"], [-3, 0.0, 0.0], [3, 0.85, 0.85],
                        [0.0, 0.0, 0.0], ["link0", "xarm_gripper_base_link"])
        env = SimpleNamespace(robot=_Bot(xarm, GRIPPER_JOINTS=("drive_joint",), EE_BODY="xarm_gripper_base_link"),
                              num_envs=1)
        g = _guards(env)[0]
        self.assertEqual(g.grip.tolist(), [False, True, False])  # the driven joint, not the mimic "finger"
        self.assertEqual(float(g.close_sign[1]), 1.0)  # closes toward +0.85
        self.assertEqual(g.ee_idxs, [1])
        # Piper: mirrored fingers, joint7 0..0.05 open at 0.04, joint8 -0.05..0 open at -0.04
        piper = fake_art(1, ["joint7", "joint8"], [0.0, -0.05], [0.05, 0.0], [0.04, -0.04], ["link6"])
        g = _guards(SimpleNamespace(robot=_Bot(piper, GRIPPER_JOINTS=("joint7", "joint8")), num_envs=1))[0]
        self.assertEqual(g.close_sign.tolist(), [-1.0, 1.0])  # each closes toward 0

    def test_a_humanoid_has_two_hands_and_its_legs_are_not_slowed(self):
        from robobench.core.prevention import SpeedGovernor, hand_bodies

        n = 1
        g1 = fake_art(n, ["left_hip_pitch_joint", "left_knee_joint", "left_shoulder_pitch_joint"],
                      [-2, 0, -2], [2, 2, 2], [0.0, 0.5, 0.0], ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"])
        env = SimpleNamespace(robot=_Bot(g1, EE_BODIES=("left_wrist_yaw_link", "right_wrist_yaw_link"),
                                         LEG_JOINTS=(".*_hip_.*_joint", ".*_knee_joint")), num_envs=n, dt=0.01)
        self.assertEqual([i for _, i in hand_bodies(env)], [1, 2])  # both hands
        SpeedGovernor(env, ee_speed_max=0.1)
        d = g1.data
        d.body_lin_vel_w[:, 2] = torch.tensor([1.0, 0.0, 0.0])  # the RIGHT hand is fast
        action = d.joint_pos.clone() + 0.1
        env.robot.apply_action(action, 0)
        env.robot.post_step()
        lead = d.joint_pos_target - d.joint_pos
        self.assertAlmostEqual(float(lead[0, 0]), 0.1, places=5)  # hip: not slowed (a leg)
        self.assertAlmostEqual(float(lead[0, 1]), 0.1, places=5)  # knee: not slowed
        self.assertLess(float(lead[0, 2]), 0.05)  # the arm joint is slowed (the right hand counts)

    def test_hold_monitor_turning_with_the_hand_is_not_slip(self):
        from robobench.core.prevention import HoldMonitor

        egg = rigid(2, pos=(0.4, 0.0, 0.2))  # 10 cm below the hand (0.4, 0, 0.3)
        env = prev_env(2, rigid_objects={"egg": egg})
        d = env.robot.articulation.data
        d.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2] * 2)
        mon = HoldMonitor(env, "egg", slip_tol=0.005)
        mon.lock()
        s = math.sqrt(0.5)  # the hand turns 90 deg about x, carrying the egg rigidly with it
        d.body_quat_w[:, 1] = torch.tensor([s, s, 0.0, 0.0])
        egg.data.root_pos_w[:] = torch.tensor([0.4, 0.1, 0.3])  # the offset rotated with the hand
        substep(env)
        self.assertEqual(mon.slipping.tolist(), [False, False])
        egg.data.root_pos_w[:, 1] += 0.01  # now it really slides 1 cm in the hand
        substep(env)
        self.assertEqual(mon.slipping.tolist(), [True, True])

    def test_the_release_gate_keeps_the_squeeze(self):
        from robobench.suites.assembly.prevention.bulb import BulbPrevention

        env, bulb = bulb_env()
        prev = BulbPrevention(env)
        d = env.robot.articulation.data
        squeeze = d.joint_pos.clone()
        squeeze[:, 7:9] = 0.0  # commanded shut on the bulb (fingers stalled at 0.02)
        substep(env, squeeze)
        squeezing_target = d.joint_pos_target[:, 7:9].clone()
        self.assertTrue(bool((squeezing_target < d.joint_pos[:, 7:9]).all()))  # pressing inward
        self.assertTrue(prev.lift_test("bulb_0", lift_with(env, bulb), steps=10))
        put_bulb(bulb, 0.20)  # carried in the air
        opening = d.joint_pos.clone()
        opening[:, 7:9] = 0.04
        substep(env, opening)  # told to open mid-air
        held = d.joint_pos_target[:, 7:9]
        self.assertTrue(bool((held < d.joint_pos[:, 7:9]).all()))  # still pressing, not merely "in place"

    def test_the_grip_cap_does_not_slow_opening(self):
        from robobench.core.prevention import GripGovernor

        env = prev_env()
        GripGovernor(env, force=1.0)  # 0.5 mm of squeeze at stiffness 2000
        d = env.robot.articulation.data
        action = d.joint_pos.clone()
        action[:, 7:9] = 0.04  # open fully from 0.02
        substep(env, action)
        self.assertTrue(torch.allclose(d.joint_pos_target[:, 7:9], torch.full((2, 2), 0.04)))

    def test_jam_trigger_stays_below_an_enforce_torque_clamp(self):
        from robobench.core.prevention import effort_trigger
        from robobench.core.safety import install_safety

        env = prev_env()
        install_safety(env, scene="bulb", mode="enforce")
        self.assertAlmostEqual(effort_trigger(env, 0.5), 0.45)  # bulb's rule clamps at 0.5
        env2 = prev_env()
        install_safety(env2, scene="bulb", mode="monitor")
        self.assertAlmostEqual(effort_trigger(env2, 0.5), 0.5)  # monitor mode clamps nothing

    def test_a_monitor_added_before_the_shield_can_still_be_stopped(self):
        from robobench.core.prevention import add_post_step, remove_post_step
        from robobench.core.safety import install_safety

        env = prev_env()
        calls = []
        fn = lambda: calls.append(1)  # noqa: E731
        add_post_step(env, fn)  # before the shield: lives on the robot's chain
        install_safety(env)
        remove_post_step(env, fn)
        substep(env)
        self.assertEqual(calls, [])
