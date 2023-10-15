# evo Trajectory evaluation
The Python package [`evo`](https://github.com/MichaelGrupp/evo/wiki) is used to convert and (maybe) analyze pose trajectories.

Inference in the `camera_depth_optical_frame`.
* rosbag -> Julia: `to_pose.sh <data.bag>` reads the tf data from a rosbag and writes the pose messages to another bag in the current working directory

Analysis in `world` frame as the object was static and only the robot was moving.
* Julia -> rosbag: `world_bag.py` reads the TUM formatted results from Julia TUM export and the original bag.
  Writes the data to `tf_static` and `tf` in a new bag.
* Poses for analysis: `world_poses.sh` converts the previous bag to TUM files which contain poses in the world frame

# Blender models
Re-export the DAE files in Blender to obj with Y-forward and Z-up
