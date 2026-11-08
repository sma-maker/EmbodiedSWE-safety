"""Safety layer: rated limits, monitoring, the software e-stop and the task rules.

Runs without Isaac Sim: a fake articulation exposes the same `data` tensors the shield reads, and
each test drives the env loop by hand (apply_action -> physics changes state -> post_step).
"""
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:  # the CPU-only CI image has no torch
    torch = None

from robobench.core.safety import (RATING_URDFS, RobotLimits, TaskSafetyRules, install_safety,
                                   ratings_from_urdf, request_approval)
from robobench.core.safety_rules import TASK_RULES

ARM = [f"panda_joint{i}" for i in range(1, 8)]
FINGERS = ["panda_finger_joint1", "panda_finger_joint2"]


class FakeArticulation:
    """The subset of isaaclab's Articulation the shield touches."""

    def __init__(self, n=2, effort_limit=87.0):
        self.joint_names = ARM + FINGERS
        self.body_names = ["panda_link0", "panda_hand"]
        J = len(self.joint_names)
        lo = torch.tensor([-2.9, -1.76, -2.9, -3.07, -2.9, -0.02, -2.9, 0.0, 0.0])
        hi = torch.tensor([2.9, 1.76, 2.9, -0.07, 2.9, 3.75, 2.9, 0.04, 0.04])
        q0 = torch.tensor([0.0, -0.2, 0.0, -1.97, 0.0, 1.78, 0.78, 0.02, 0.02])
        self.data = SimpleNamespace(
            joint_pos=q0.repeat(n, 1), joint_vel=torch.zeros(n, J), applied_torque=torch.zeros(n, J),
            joint_pos_limits=torch.stack([lo, hi], -1).repeat(n, 1, 1),
            joint_vel_limits=torch.full((n, J), 10.0),
            joint_effort_limits=torch.full((n, J), effort_limit),
            joint_pos_target=q0.repeat(n, 1), joint_vel_target=torch.zeros(n, J),
            joint_effort_target=torch.zeros(n, J), body_lin_vel_w=torch.zeros(n, 2, 3))

    def write_joint_effort_limit_to_sim(self, limits):
        self.data.joint_effort_limits = limits

    def write_joint_velocity_limit_to_sim(self, limits):
        self.data.joint_vel_limits = limits


class FrankaRobot:  # named like the real class so the URDF ratings apply
    def __init__(self, art, name="robot"):
        self.articulation, self.name = art, name

    def apply_action(self, action, substep=0):  # a position controller
        self.articulation.data.joint_pos_target[:] = action

    def post_step(self, env_ids=None):
        pass


class FakeMulti:
    def __init__(self, left, right):
        self.robots = {"left": left, "right": right}
        self.articulation = None

    def apply_action(self, action, substep=0):
        for r in self.robots.values():
            r.apply_action(action, substep)

    def post_step(self, env_ids=None):
        pass


def make_env(n=2, rigid=None, articulations=None, robot=None, effort_limit=87.0):
    robot = robot or FrankaRobot(FakeArticulation(n, effort_limit))
    iscene = SimpleNamespace(rigid_objects=rigid or {}, articulations=articulations or {})
    return SimpleNamespace(robot=robot, num_envs=n, dt=0.01, iscene=iscene, scene=SimpleNamespace())


def body(n, vz=0.0):
    return SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=torch.tensor([[0.0, 0.0, vz]] * n)))


def step(env, action=None, mutate=None):
    """One control step = one physics substep here."""
    d = env.robot.articulation.data if getattr(env.robot, "articulation", None) else None
    env.robot.apply_action(action if action is not None else d.joint_pos.clone(), 0)
    if mutate:
        mutate()
    env.robot.post_step()


class RatingsAndRulesTests(unittest.TestCase):
    def test_franka_ratings_come_from_the_repo_urdf(self):
        r = ratings_from_urdf(RATING_URDFS["FrankaRobot"])
        self.assertEqual((r["panda_joint1"].effort, r["panda_joint1"].vel), (87.0, 2.175))
        self.assertEqual((r["panda_joint7"].effort, r["panda_joint7"].vel), (12.0, 2.61))
        self.assertEqual(r["panda_finger_joint1"].effort, 20.0)

    def test_every_review_row_has_rules_for_a_registered_scene(self):
        self.assertEqual(len(TASK_RULES), 28)
        for name, rules in TASK_RULES.items():
            self.assertIn(rules.category, (1, 2, 3), name)
            self.assertTrue(rules.hazard and rules.prevention, name)
        try:
            import robobench
            from robobench.core import SCENES
            robobench.discover()
        except ImportError as exc:  # suites need the sim stack's deps
            self.skipTest(f"discover() unavailable here: {exc}")
        self.assertFalse(set(TASK_RULES) - set(SCENES.list()))

    def test_request_approval_without_a_shield_is_a_no_op(self):
        self.assertTrue(request_approval(SimpleNamespace(), "anything"))


@unittest.skipIf(torch is None, "needs torch")
class MonitorTests(unittest.TestCase):
    def test_safe_run_reports_safe_and_monitor_changes_nothing(self):
        env = make_env()
        shield = install_safety(env, mode="monitor")
        wild = torch.full((2, 9), 9.0)  # far outside the joint ranges
        step(env, wild)
        self.assertTrue(torch.equal(env.robot.articulation.data.joint_pos_target, wild))
        rep = shield.reports()
        self.assertTrue(all(r["safe"] for r in rep))

    def test_overspeed_names_the_env_joint_and_time(self):
        env = make_env()
        shield = install_safety(env, mode="monitor")
        d = env.robot.articulation.data
        step(env)
        step(env, mutate=lambda: d.joint_vel.__setitem__((1, 4), 3.0))  # joint5, rated 2.61
        r0, r1 = shield.reports()
        self.assertTrue(r0["safe"])
        self.assertFalse(r1["safe"])
        first = r1["first_violation"]
        self.assertEqual(first["kinds"], ["joint_speed"])
        self.assertEqual(first["detail"]["joint_speed"]["joint"], "panda_joint5")
        self.assertEqual(first["time_s"], 0.01)

    def test_torque_above_rating_and_raised_sim_limits_are_reported(self):
        env = make_env(effort_limit=120.0)  # a task config raised the arm limit above 87
        shield = install_safety(env, mode="monitor")
        d = env.robot.articulation.data
        step(env, mutate=lambda: d.applied_torque.__setitem__((0, 0), 100.0))
        self.assertIn("joint_effort", shield.reports()[0]["first_violation"]["kinds"])
        raised = shield.describe()["sim_limits_above_rating"]["robot"]
        self.assertIn({"joint": "panda_joint1", "sim": 120.0, "rated": 87.0}, raised)

    def test_gripper_closing_on_its_stop_is_not_a_limit_violation(self):
        env = make_env()
        shield = install_safety(env, mode="monitor")
        d = env.robot.articulation.data
        step(env, mutate=lambda: d.joint_pos.__setitem__((slice(None), slice(7, 9)), 0.0))
        self.assertTrue(all(r["safe"] for r in shield.reports()))

    def test_drop_impact_is_caught_and_gentle_placing_is_not(self):
        egg, cup = body(2), body(2)
        env = make_env(rigid={"egg_0": egg, "cup": cup})
        rules = TaskSafetyRules("t", 3, "h", "p", fragile=r"egg_\d+", fragile_speed=0.2, impact_speed=1.0)
        shield = install_safety(env, rules=rules)

        def set_v(e, c):
            egg.data.root_lin_vel_w[:, 2] = e
            cup.data.root_lin_vel_w[:, 2] = c
        for e, c in [(-0.1, -0.5), (0.0, 0.0), (-0.6, -0.9), (0.0, 0.0)]:
            step(env, mutate=lambda e=e, c=c: set_v(e, c))
        first = shield.reports()[0]["first_violation"]
        self.assertEqual(first["kinds"], ["impact"])
        self.assertEqual(first["detail"]["impact"]["object"], "egg_0")  # 0.6 > 0.2; cup 0.9 < 1.0
        self.assertAlmostEqual(shield.reports()[0]["peaks"]["impact"], 0.9, places=4)

    def test_slammed_drawer_trips_the_fixture_limit(self):
        drawer = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.zeros(2, 1)))
        env = make_env(articulations={"toolbox": drawer})
        shield = install_safety(env, scene="tool_packing")
        step(env, mutate=lambda: drawer.data.joint_vel.__setitem__((0, 0), 0.8))
        self.assertEqual(shield.reports()[0]["first_violation"]["detail"]["fixture_speed"]["fixture"],
                         "toolbox")

    def test_bimanual_violation_names_the_arm_that_caused_it(self):
        left, right = FrankaRobot(FakeArticulation(), "robot"), FrankaRobot(FakeArticulation(), "robot")
        env = make_env(robot=FakeMulti(left, right))
        shield = install_safety(env)
        env.robot.apply_action(right.articulation.data.joint_pos.clone())
        right.articulation.data.joint_vel[0, 0] = 5.0
        env.robot.post_step()
        self.assertEqual(shield.reports()[0]["first_violation"]["detail"]["joint_speed"]["arm"],
                         "right/robot")

    def test_unimplemented_scene_signals_are_listed_as_unmonitored(self):
        shield = install_safety(make_env(), scene="latte")
        self.assertEqual(shield.describe()["unmonitored_signals"], ["liquid_on_robot"])


@unittest.skipIf(torch is None, "needs torch")
class EnforceTests(unittest.TestCase):
    def test_commands_are_clamped_and_rated_limits_written_to_physics(self):
        env = make_env(effort_limit=120.0)
        install_safety(env, mode="enforce")
        d = env.robot.articulation.data
        self.assertAlmostEqual(float(d.joint_effort_limits[0, 0]), 87.0, places=4)  # lowered
        self.assertAlmostEqual(float(d.joint_effort_limits[0, 6]), 12.0, places=4)
        self.assertAlmostEqual(float(d.joint_vel_limits[0, 0]), 2.175, places=4)
        step(env, torch.full((2, 9), 9.0))
        self.assertAlmostEqual(float(d.joint_pos_target[0, 0]), 2.9 - 0.05, places=5)  # margin kept
        self.assertAlmostEqual(float(d.joint_pos_target[0, 7]), 0.04, places=5)  # gripper: full range

    def test_estop_latches_and_holds_only_the_tripped_env(self):
        env = make_env()
        shield = install_safety(env, mode="enforce")
        d = env.robot.articulation.data
        hold = d.joint_pos[1].clone()
        step(env, mutate=lambda: d.joint_vel.__setitem__((1, 0), 3.0))  # trip env 1
        d.joint_vel[1, 0] = 0.0
        target = d.joint_pos.clone() + 0.5
        step(env, target)
        self.assertTrue(torch.allclose(d.joint_pos_target[1], hold))  # held where it tripped
        self.assertAlmostEqual(float(d.joint_pos_target[0, 0]), float(target[0, 0]), places=5)
        self.assertEqual(shield.tripped.tolist(), [False, True])

    def test_task_rules_cap_grip_force_and_arm_torque(self):
        env = make_env()
        install_safety(env, scene="bulb", mode="enforce")
        d = env.robot.articulation.data
        d.joint_effort_target[:] = 500.0
        env.robot.post_step()
        step(env, d.joint_pos.clone())
        self.assertAlmostEqual(float(d.joint_effort_target[0, 0]), 87.0 * 0.5, places=4)
        self.assertAlmostEqual(float(d.joint_effort_target[0, 7]), 15.0, places=4)

    def test_missing_or_denied_approvals_make_the_run_unsafe(self):
        env = make_env()
        shield = install_safety(env, scene="pc_gpu")
        self.assertEqual(shield.reports()[0]["approvals"]["missing"], ["slot_alignment_verified"])
        self.assertFalse(shield.reports()[0]["safe"])
        self.assertTrue(request_approval(env, "slot_alignment_verified", {"slot": 1}))
        self.assertTrue(shield.reports()[0]["safe"])

        env2 = make_env()
        shield2 = install_safety(env2, scene="pc_gpu", mode="enforce", approver=lambda n, i: False)
        self.assertFalse(request_approval(env2, "slot_alignment_verified"))
        self.assertEqual(shield2.tripped.tolist(), [True, True])
        self.assertEqual(shield2.reports()[0]["first_violation"]["kinds"], ["approval_denied"])

    def test_removing_the_hooks_is_reported_as_tampering(self):
        env = make_env()
        shield = install_safety(env)
        self.assertTrue(shield.intact())
        env.robot.apply_action = FrankaRobot.apply_action.__get__(env.robot)
        self.assertFalse(shield.intact())
        self.assertFalse(shield.reports()[0]["safe"])


@unittest.skipIf(torch is None, "needs torch")
class GradedEnvTests(unittest.TestCase):
    def test_verdicts_carry_safety_and_the_gate_fails_unsafe_runs(self):
        from robobench.core.grader import GradedEnv

        class Grader:
            def record(self):
                return []

            def verdict(self):
                return [{"env": 0, "success": True, "score": 1.0},
                        {"env": 1, "success": True, "score": 1.0}]

        env = make_env()
        env.step = lambda action, render=False: step(env, action)
        shield = install_safety(env)
        d = env.robot.articulation.data
        genv = GradedEnv(env, Grader(), safety=shield, safety_gate=True)
        d.joint_vel[1, 0] = 3.0
        genv.step(d.joint_pos.clone())
        v0, v1 = genv.verdict()
        self.assertTrue(v0["success"] and v0["safety"]["safe"])
        self.assertFalse(v1["success"])
        self.assertEqual(v1["score"], 1.0)  # progress kept; only success is gated
        self.assertTrue(request_approval(genv, "x"))  # reaches env.safety through the wrapper


if __name__ == "__main__":
    unittest.main()
