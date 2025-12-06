import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T

# env.external_sensors["external_sensor1"].set_position_orientation(position=[0.2130, 0.5043, 1.3359], orientation=[-0.2127,  0.4026,  0.7874, -0.4156], frame="parent")
# robot.set_joint_positions(th.tensor([0.0, -1.0]), indices=robot.camera_control_idx)

# Pdb) robot.links["eyes"].get_position_orientation()
# (tensor([ 0.2862, -1.3925,  1.1458]), tensor([ 0.0981, -0.2639, -0.8994,  0.3343]))
# (Pdb) robot.links["base_link"].get_position_orientation()
# (tensor([ 0.0859, -1.2322,  0.0762]), tensor([-6.2282e-08,  1.3792e-08, -4.1650e-01,  9.0914e-01]))

T_eyes_world = th.eye(4)
T_eyes_world[:3, :3] = T.quat2mat(th.tensor([ 0.0981, -0.2639, -0.8994,  0.3343]))
T_eyes_world[:3, 3] = th.tensor([ 0.2862, -1.3925,  1.1458])
T_base_world = th.eye(4)
T_base_world[:3, :3] = T.quat2mat(th.tensor([-6.2282e-08,  1.3792e-08, -4.1650e-01,  9.0914e-01]))
T_base_world[:3, 3] = th.tensor([ 0.0859, -1.2322,  0.0762])

T_eyes_base = T_base_world.inverse() @ T_eyes_world
print(T_eyes_base)
position = T_eyes_base[:3, 3]
orientation = T.mat2quat(T_eyes_base[:3, :3])
print(position, orientation)

# T_ext_cam_base = th.eye(4)
# T_ext_cam_base[:3, :3] = T.quat2mat(th.tensor([-0.1044,  0.3851,  0.8851, -0.2394]))
# T_ext_cam_base[:3, 3] = th.tensor([0.3833, 0.5289, 1.4386])

# T_ext_cam_eyes = T_eyes_base.inverse() @ T_ext_cam_base
# # print(T_ext_cam_eyes)

# position = T_ext_cam_eyes[:3, 3]
# orientation = T.mat2quat(T_ext_cam_eyes[:3, :3])
# print(position, orientation)


# ensor([ 0.4859, -1.8219,  1.2164]), tensor([ 0.5857, -0.0093, -0.0129,  0.8103]))
# env.external_sensors["external_sensor0"].set_position_orientation(position=[0.4859, -1.8219,  1.2164], orientation=[ 0.5857, -0.0093, -0.0129,  0.8103], frame="world")

