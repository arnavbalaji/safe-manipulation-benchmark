from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm

from safety_benchmark.utils.misc_utils import json_default
# from omnigibson.joylo.gello.robots.sim_robot.og_sim import DISABLED_TRANSITION_RULES


gm.ENABLE_OBJECT_STATES = True
# gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False

# for rule in DISABLED_TRANSITION_RULES:
#     rule.ENABLED = False

collect_hdf5_path = "/home/arpit/behavior_dataset/task-0000/episode_00000010.hdf5"
output_hdf5_path = "/home/arpit/behavior_dataset/task-0000/episode_00000010_replayed.hdf5"
# output_hdf5_path = "temp.hdf5"

# Create a playback env and playback the data, collecting obs along the way
env = DataPlaybackWrapper.create_from_hdf5(
    input_path=collect_hdf5_path,
    output_path=output_hdf5_path,
    robot_obs_modalities=[],
    # robot_sensor_config=robot_sensor_config,
    # external_sensors_config=external_sensors_config,
    n_render_iterations=1,
    only_successes=False,
)
env.playback_dataset(record_data=True)
env.save_data()
breakpoint()