import h5py
import pandas as pd
import os

data_path = "/home/arpit/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/sim_behavior_r1_pro.task-0000_turning_on_radio/data/chunk-000/episode_00000000.parquet"
df = pd.read_parquet(data_path, engine="pyarrow")

f1 = h5py.File("/home/arpit/behavior_dataset/task-0000/episode_00000010_replayed.hdf5", "r")
breakpoint()