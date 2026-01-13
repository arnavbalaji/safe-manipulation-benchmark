import h5py
import json
import pandas as pd
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import os
import matplotlib.pyplot as plt

# data_path = "/home/arpit/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/sim_behavior_r1_pro.task-0000_turning_on_radio/data/chunk-000/episode_00000000.parquet"
# df = pd.read_parquet(data_path, engine="pyarrow")

# f1 = h5py.File("/home/arpit/behavior_dataset/task-0034/episode_00340020_replayed.hdf5", "r")
# f2 = h5py.File("/home/arpit/behavior_dataset/task-0034/episode_00340020.hdf5", "r")

# f1 = h5py.File("/home/arpit/behavior_dataset/task-0000/episode_00000010_replayed.hdf5", "r")
# f2 = h5py.File("/home/arpit/behavior_dataset/task-0000/episode_00000010.hdf5", "r")

# f1 = h5py.File("/home/arpit/behavior_dataset/task-0040/episode_00402500_replayed.hdf5", "r")
# f2 = h5py.File("/home/arpit/behavior_dataset/task-0040/episode_00402500.hdf5", "r")

# f4 = h5py.File("/home/arpit/test_projects/groot/Isaac-GR00T/gr00t/eval/sim/BEHAVIOR/rollouts/make_microwave_popcorn/rollout_0000_00402500.hdf5", "r")
# f3 = h5py.File("/home/arpit/test_projects/groot/Isaac-GR00T/gr00t/eval/sim/BEHAVIOR/rollouts/clean_a_trumpet/rollout_0000_00372720_playback.hdf5", "r")

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# axs[0, 0].imshow(f3["data/demo_0/obs/external::external_sensor1::rgb"][2, :, :, :3])
# axs[0, 1].imshow(f3["data/demo_1/obs/external::external_sensor1::rgb"][2, :, :, :3])
# axs[1, 0].imshow(f3["data/demo_2/obs/external::external_sensor1::rgb"][2, :, :, :3])
# axs[1, 1].imshow(f3["data/demo_3/obs/external::external_sensor1::rgb"][2, :, :, :3])
# plt.show()


# f = h5py.File("/home/arpit/test_projects/safe-manipulation-benchmark/resources/playback_data/robot_tiago_motion_1_playback.hdf5", "r")
# for i in range(len(f["data/demo_0/info/damage_info"])):
#     damage_info = json.loads(f["data/demo_0/info/damage_info"][i].decode("utf-8"))
#     print(damage_info["tiago0"]["gripper_right_left_finger_link"]["mechanical"]["contacts"])

# breakpoint()

# # ========== debug health of robot =========
# f = h5py.File("/home/arpit/projects/Isaac-GR00T/gr00t/eval/sim/BEHAVIOR/rollouts/attach_a_camera_to_a_tripod/rollout_0021_00351240_playback.hdf5", "r")
# for demo_idx in range(len(f["data"])):
#     print("episode: ", demo_idx)
#     all_obj_healths = np.array(f[f"data/demo_{demo_idx}/obs/health"])
#     health_list_link_names = f[f"data/demo_{demo_idx}"].attrs["health_list_link_names"]
#     health = dict()
#     for obj_name in health_list_link_names:
#         health[obj_name] = all_obj_healths[:, np.where(health_list_link_names == obj_name)[0][0]]
#         health[obj_name] = health[obj_name][1:]
#         # for debugging
#         if np.any(health[obj_name] < 100.0):
#             # breakpoint()
#             print(f"{obj_name} has health less than 100.0")

#     target_objects_health = ["robot_r1"]
#     # Obtain health information for the entire target objects 
#     for obj_name in target_objects_health:
#         arrays = [v for k, v in health.items() if k.startswith(f"{obj_name}@")]
#         # Compute element-wise min
#         if arrays:
#             health[obj_name] = np.minimum.reduce(arrays)
#         else:
#             health[obj_name] = None
#     print("robot_r1_health: ", health["robot_r1"][-1])
#     print(" ========================================== ")
#     # breakpoint()

# # ===========================================

# =========== Debugging action vs delta right eef position z ===========
# action_list = []
# delta_right_eef_pos_z_list = []
# for i in range(len(f["data/demo_0/action"])-1):
#     action = f["data/demo_0/action"][i+1][15]
#     delta_right_eef_pos_z = f["data/demo_0/obs/right_eef_pos"][i+1][2] - f["data/demo_0/obs/right_eef_pos"][i][2]
#     action_list.append(action)
#     delta_right_eef_pos_z_list.append(delta_right_eef_pos_z)

# plt.scatter(range(len(delta_right_eef_pos_z_list)), delta_right_eef_pos_z_list, color='b', s=10, label=None, alpha=0.5)
# plt.scatter(range(len(action_list)), action_list, color='orange', s=10, label=None, alpha=0.5)

# plt.plot(delta_right_eef_pos_z_list, label="Delta Right EEF Position Z")
# plt.plot(action_list, label="Action")
# plt.legend()
# plt.title("Delta Right EEF Position Z vs Action")
# plt.xlabel("Time")
# plt.show()
# plt.savefig("/home/arpit/Downloads/plot_2.png")
# =========================================================================

f = h5py.File("/home/arpit/test_projects/safe-manipulation-benchmark/resources/playback_data/shelve_item_test2-playback.hdf5", "r")
breakpoint()