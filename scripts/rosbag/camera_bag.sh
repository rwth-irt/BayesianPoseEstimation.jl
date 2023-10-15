#! /bin/sh
. venv/bin/activate
# extract track_object in camera frame for initialization of PF in Julia
evo_traj bag -c evo_config.json $1 /tf:camera_depth_optical_frame.tracked_object --save_as_tum
