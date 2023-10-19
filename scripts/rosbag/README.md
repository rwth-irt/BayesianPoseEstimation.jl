# evo Trajectory evaluation
The Python package [`evo`](https://github.com/MichaelGrupp/evo/wiki) is used to convert and (maybe) analyze pose trajectories.

## Setup
```sh
cd scripts/rosbag
python 3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Scripts
Activate the virtual environment before running the scripts
```sh
cd scripts/rosbag
source venv/bin/activate
```

Inference in the `camera_depth_optical_frame`.
* rosbag -> `ros_particle_filter.jl`: `camera_bag.sh <data.bag>` reads the tf data from a rosbag and writes the pose messages to another bag in the current working directory

Analysis in `world` frame as the object was static and only the robot was moving.
* Julia -> rosbag: `tf_bag.py` reads the TUM formatted results from Julia TUM export and the original bag.
  Writes the data to `tf_static` and `tf` in a new bag.
* Poses for analysis: `world_poses.sh` converts the previous bag to TUM files which contain poses in the world frame

## Expected folder structure
* data/
  * rosbags/
    * {experiment}>/
      * *at_result.bag* - contains the `pose/only_pf` and `pose/robot_pf` from the at paper (Übelhör 2020)
      * *original.bag* - `rosbag decompress original.bag` for improved performance
      * *pose_only_pf.tum* & *pose_robot_pf.tum* are extracted from *at_result.bag* via *at_to_tum.sh* in the world frame
      * *tf_camera_depth_optical_frame.tracked_object.tum* the robot pf in the camera frame for initialization
      * *track.obj* use this model for tracking
  * exp_raw/pf/
    * {configuration}.jld2
  * exp_pro/pf/
    * {configuration}.tum
    * {configuration}.bag

# Blender models
Re-export the DAE files in Blender to obj with Y-forward and Z-up
