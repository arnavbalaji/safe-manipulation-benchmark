import os
import json
import h5py
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm
import omnigibson as og
from pathlib import Path

from safety_benchmark.utils.misc_utils import json_default
# from omnigibson.joylo.gello.robots.sim_robot.og_sim import DISABLED_TRANSITION_RULES


gm.ENABLE_OBJECT_STATES = True
# gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

# for rule in DISABLED_TRANSITION_RULES:
#     rule.ENABLED = False


# behavior_tasks = ["task-0035", "task-0037", "task-0040", "task-0034", "task-0030"]
behavior_tasks = ["task-0040", "task-0034", "task-0030"]
num_episodes = 25
for task in behavior_tasks:
    dir_path = Path(f"/home/arpit/behavior_dataset/{task}")
    files = sorted(
        (
            f for f in dir_path.iterdir()
            if f.is_file() and "replayed" not in f.name
        ),
        key=lambda f: f.stat().st_size
    )
    output_dir = f"/home/arpit/behavior_dataset/{task}/replayed"
    os.makedirs(output_dir, exist_ok=True)

    for episode_num, f in enumerate(files):
        print(f.name, f.stat().st_size)

        collect_hdf5_path = f"/home/arpit/behavior_dataset/{task}/{f.name}"
        output_hdf5_path = f"{output_dir}/{f.name}_replayed.hdf5"

        # Read viewer dimensions from HDF5 config to ensure consistency
        viewer_width = None
        viewer_height = None
        try:
            with h5py.File(collect_hdf5_path, "r") as hdf5_file:
                config = json.loads(hdf5_file["data"].attrs["config"])
                if "render" in config:
                    viewer_width = config["render"].get("viewer_width")
                    viewer_height = config["render"].get("viewer_height")
        except Exception as e:
            print(f"Warning: Could not read viewer dimensions from HDF5: {e}")

        # Create a playback env and playback the data, collecting obs along the way
        env = DataPlaybackWrapper.create_from_hdf5(
            input_path=collect_hdf5_path,
            output_path=output_hdf5_path,
            robot_obs_modalities=[],
            n_render_iterations=1,
            only_successes=False,
        )
        env.playback_dataset(record_data=True)
        env.save_data()
        env.close()
        
        # Clear with viewer dimensions to match next environment's expectations
        if viewer_width is not None and viewer_height is not None:
            og.clear(viewer_width=viewer_width, viewer_height=viewer_height)
        else:
            og.clear()

        if episode_num == num_episodes - 1:
            break

og.shutdown()
