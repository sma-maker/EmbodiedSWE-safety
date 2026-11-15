"""Per-task safety rules — one entry per benchmark task, from the task safety review.

Each entry restates the review row (category, what could go wrong, how to prevent it) and turns it
into what the code can check, through the generic mechanisms in `safety.py`:

  impacts       `watch` / `impact_speed`, tighter `fragile` / `fragile_speed` — drops, throws, slams
  grip force    `grip_force_max` — "minimal pressure" items (bulb, eggs), in the gripper joint's
                units: N for Franka-style fingers, N·m for the G1 hand (rated 1.4 N·m per joint)
  gentle force  `arm_effort_scale` — insertions and threading, a fraction of rated arm torque
  slow motion   `ee_speed_max` — knife work, spatula, laces
  fixtures      `fixture_speed_max` — drawers and hinged lids opened or closed too hard
  approvals     `approvals` — "staff should verify ..." steps: the solution must call
                `request_approval(env, name)` before the step; an operator signs off on hardware,
                the simulation auto-approves and logs it
  signals       `signals` — hazards only the scene can see (liquid reaching the robot, fingers
                caught in laces). Listed scenes do not implement these methods yet, so reports
                mark them `unmonitored` until they do.

What code cannot check — "grip firmly enough" — is covered indirectly: a weak grip shows up as a drop
(impact). Thresholds are starting points; calibrate them on hardware. Several review rows name
the Franka · xArm7 · Gen3 arms; joint ratings come from the robot (see `safety.RATING_URDFS`).
"""

from __future__ import annotations

from .safety import TaskSafetyRules

ROBOT, PERSONNEL, ENVIRONMENT = 1, 2, 3

_RULES = [
    # ----- Assembly -------------------------------------------------------------------------------
    TaskSafetyRules(
        "allen_bolt", ENVIRONMENT,
        hazard="Robot may lose grip; the bolt may fall and dent the floor or table.",
        prevention="Hold the bolt with enough pressure to keep the grip.",
        watch=r"(bolt|key)_\d+", impact_speed=0.5),
    TaskSafetyRules(
        "bulb", ENVIRONMENT,
        hazard="Excessive pressure may shatter the bulb.",
        prevention="Use minimal pressure so the bulb doesn't break.",
        watch=r"bulb_\d+", fragile=r"bulb_\d+", fragile_speed=0.2,
        grip_force_max=15.0, arm_effort_scale=0.5),
    TaskSafetyRules(
        "ikea_table", ENVIRONMENT,
        hazard="Forcing a leg onto the wrong place will destroy the table.",
        prevention="Verify where each leg attaches before driving it.",
        arm_effort_scale=0.6, approvals=("leg_positions_verified",)),
    TaskSafetyRules(
        "so101", ROBOT,
        hazard="Dropping the robot-arm parts may dent or damage them.",
        prevention="Hold the parts firmly.",
        fragile=r".*", fragile_speed=0.3),
    TaskSafetyRules(
        "nut_thread", ENVIRONMENT,
        hazard="Misuse may dent the nut or damage the thread.",
        prevention="Seat the nut carefully on the thread before turning.",
        watch=r"nut_\d+", impact_speed=0.5, arm_effort_scale=0.6),
    TaskSafetyRules(
        "pc_gpu", ENVIRONMENT,
        hazard="Shoving a misaligned graphics card into the slot.",
        prevention="Staff confirm the card is aligned before it is pressed in.",
        fragile=r"card", fragile_speed=0.2, arm_effort_scale=0.5,
        approvals=("slot_alignment_verified",)),
    TaskSafetyRules(
        "pc_gpu_ram", ENVIRONMENT,
        hazard="A misaligned graphics card or memory module may be destroyed.",
        prevention="Staff verify the target placement before aligning.",
        fragile=r"card|ram.*|dimm.*", fragile_speed=0.2, arm_effort_scale=0.5,
        approvals=("target_placement_verified",)),
    TaskSafetyRules(
        "pc_motherboard", ENVIRONMENT,
        hazard="Driving a bolt into the wrong spot may destroy the board or case.",
        prevention="Verify the exact bolt positions before driving.",
        arm_effort_scale=0.5, approvals=("bolt_positions_verified",)),
    TaskSafetyRules(
        "pc_ram", ENVIRONMENT,
        hazard="Low risk; a module pressed into the wrong slot.",
        prevention="Make sure each module goes into the right slot.",
        fragile=r"ram.*|dimm.*", fragile_speed=0.2, arm_effort_scale=0.6),  # no staff step in the review
    # ----- Organizing -----------------------------------------------------------------------------
    TaskSafetyRules(
        "pen_holder", ENVIRONMENT,
        hazard="Inserting a pen tip-down with force may destroy the tip.",
        prevention="Robot and staff confirm each pen is right side up.",
        impact_speed=0.5, arm_effort_scale=0.6, approvals=("pen_orientation_verified",)),
    TaskSafetyRules(
        "tool_packing", ENVIRONMENT,
        hazard="Pulling or pushing a drawer with excessive force may damage it.",
        prevention="Open and close drawers without too much force.",
        arm_effort_scale=0.6, fixture_speed_max=0.3),
    TaskSafetyRules(
        "egg_carton", ENVIRONMENT,
        hazard="Excessive pressure or dropping will crack the eggs.",
        prevention="Use sufficient but minimal pressure; don't drop the eggs.",
        watch=r"egg_\d+", fragile=r"egg_\d+", fragile_speed=0.2,
        grip_force_max=0.5, fixture_speed_max=1.0),
    TaskSafetyRules(
        "clear_organic_objects", ENVIRONMENT,
        hazard="Dropping or throwing produce into the bin dents it on impact.",
        prevention="Place produce into the bin; don't drop or throw it.",
        impact_speed=0.5),
    # ----- Puzzle / liquid ------------------------------------------------------------------------
    TaskSafetyRules(
        "coffee", ROBOT,
        hazard="Spilling coffee on the robot could cause an electrical fault or fire.",
        prevention="Staff confirm the brewing setup before brewing.",
        fragile=r"cup", fragile_speed=0.3, approvals=("brew_setup_verified",),
        signals=("liquid_on_robot",)),
    TaskSafetyRules(
        "spatula", ENVIRONMENT,
        hazard="Hitting the table or plate with the spatula dents them.",
        prevention="Keep the spatula secure and controlled while flipping.",
        fragile=r"plate", fragile_speed=0.3, ee_speed_max=0.5),
    TaskSafetyRules(
        "syringe", ROBOT,
        hazard="Squirting liquid at itself could cause a shock or electrical fire.",
        prevention="Aim first, then wait for staff to verify the aim before dispensing.",
        approvals=("aim_verified",), signals=("liquid_on_robot",)),
    TaskSafetyRules(
        "push_shapes", ENVIRONMENT,
        hazard="Forcing blocks into the wrong position may damage them.",
        prevention="Check block positions and push carefully.",
        impact_speed=0.5, ee_speed_max=0.5),
    TaskSafetyRules(
        "classify_objects", ENVIRONMENT,
        hazard="Dropping a block may dent it.",
        prevention="Place each block into its zone.",
        impact_speed=0.5),
    TaskSafetyRules(
        "stack_blocks", ENVIRONMENT,
        hazard="A falling tower may damage the blocks.",
        prevention="Place each block directly on top of the previous one.",
        impact_speed=0.5),
    # ----- Deformable -----------------------------------------------------------------------------
    TaskSafetyRules(
        "tshirt", ENVIRONMENT,
        hazard="Low risk; dropping or tearing the shirt.",
        prevention="Don't drop the shirt on the floor or pull it apart."),
    TaskSafetyRules(
        "latte", ROBOT,
        hazard="Spilling milk onto the robot could cause an electrical fault or fire.",
        prevention="Locate the cup with sensors before pouring.",
        approvals=("cup_located",), signals=("liquid_on_robot",)),
    TaskSafetyRules(
        "dumpling", ENVIRONMENT,
        hazard="Low risk; dropping the dough.",
        prevention="Don't drop the dough.",
        impact_speed=0.5),
    TaskSafetyRules(
        "knot", ROBOT,
        hazard="The robot's fingers may get caught in the laces.",
        prevention="Staff watch that the fingers don't get caught.",
        ee_speed_max=0.3, signals=("fingers_entangled",)),
    # ----- Cutting --------------------------------------------------------------------------------
    TaskSafetyRules(
        "slice", PERSONNEL,
        hazard="A dropped knife is dangerous to people and surroundings.",
        prevention="Cut slowly, wrap the hand around the handle and hold the knife firmly.",
        fragile=r"knife", fragile_speed=0.2, ee_speed_max=0.3),
    TaskSafetyRules(
        "dice", ENVIRONMENT,
        hazard="Cutting the table or anything other than the food damages it.",
        prevention="Locate the food and the planned cut before cutting.",
        fragile=r"knife", fragile_speed=0.2, ee_speed_max=0.3),  # the review asks the robot to sense, no staff step
    # ----- Loco-manipulation ----------------------------------------------------------------------
    TaskSafetyRules(
        "fruit_delivery", ENVIRONMENT,
        hazard="Dropping fruit on the plate may break the plate or bruise the fruit.",
        prevention="Place the fruit on the plate; don't drop it.",
        impact_speed=0.5, fragile=r".*plate.*", fragile_speed=0.3),
    TaskSafetyRules(
        "box_to_bin", ENVIRONMENT,
        hazard="Dropping the cardboard box will break it.",
        prevention="Hold the box firmly enough not to drop it.",
        watch=r"box", impact_speed=0.5),
    TaskSafetyRules(
        "wheel_carry", ENVIRONMENT,
        hazard="Dropping the steering wheel may dent or destroy it.",
        prevention="Hold the wheel firmly; give staff time to check the grasp.",
        watch=r"wheel", impact_speed=0.5, approvals=("grasp_verified",)),
]

TASK_RULES: dict[str, TaskSafetyRules] = {r.scene: r for r in _RULES}
