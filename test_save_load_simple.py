"""
Simple test script for OmniGibson save/load functionality.
Creates environment, teleoperates, saves state, loads state, teleoperates more.
"""

import omnigibson as og
from omnigibson.utils.ui_utils import KeyboardRobotController

SAVE_PATH = "test_save_state.json"

def teleop(env, n_steps):
    robot = env.robots[0]
    teleop = KeyboardRobotController(robot)
    for i in range(n_steps):
        env.step(teleop.get_teleop_action())
    # No teleop.clear() method; just let the object be GC-ed
    print(f"  finished {n_steps} steps")

def make_cfg(scene_file=None):
    scene_cfg = {"type": "Scene"}
    if scene_file is not None:
        scene_cfg["scene_file"] = scene_file         # ← key point!
    cfg = {
        "scene": scene_cfg,
        "robots": [{
            "type": "Tiago",
            "obs_modalities": ["rgb"],
            "action_type": "continuous",
            "action_normalize": True,
            "position": [0., 0.65, 0.0],
            "orientation": [0, 0, -1, 1],
            "grasping_mode": "assisted",
        }]
    }
    return cfg

def main():
    # ---------------- First run ----------------
    env = og.Environment(configs=make_cfg())
    env.reset()
    teleop(env, 100)

    print("saving …")
    og.sim.save([SAVE_PATH])
    og.clear()                               # destroys env, sim, stage

    # ---------------- Load from file ---------
    print("reloading from disk …")
    env = og.Environment(configs=make_cfg(scene_file=SAVE_PATH))
    env.reset()                              # state is already correct
    teleop(env, 100)

    og.shutdown()

if __name__ == "__main__":
    main() 