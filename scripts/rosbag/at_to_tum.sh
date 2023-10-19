#! /bin/sh
# extract the poses from the at paper as TUM files
evo_traj bag $1 \
        pose/robot_pf pose/only_pf \
        --save_as_tum --config scripts/rosbag/evo_config.json