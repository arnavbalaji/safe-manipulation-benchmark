import torch as th
import omnigibson as og
from omnigibson.macros import gm

from safety_benchmark.damageable_env import (
    DamageableDataPlaybackWrapper,
)

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

image_h, image_w = 256, 256

robot_sensor_config = {
    "VisionSensor": {
        "modalities": ["rgb", "seg_instance"],
        "sensor_kwargs": {
            "image_height": image_h,
            "image_width": image_w,
        },
    },
}

# Allow task-specific playback wrapper (e.g. firewood overrides playback_episode)
wrapper_cls = DamageableDataPlaybackWrapper

env = wrapper_cls.create_from_hdf5(
    input_path="/mnt/ssd/safe-manipulation-benchmark/resources/teleop_data/firewood/firewood_trial_1.hdf5",
    output_path="temp.hdf5",
    robot_obs_modalities=["proprio", "rgb", "seg_instance"],
    robot_sensor_config=robot_sensor_config,
    external_sensors_config=[],
    n_render_iterations=1,
    only_successes=False,
)

breakpoint()