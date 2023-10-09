# Reading TF data
TF data cannot be read using `RobotOSData.jl` as it returns some binary gibberish.
Use the `to_pose.sh` script to read a rosbag with [evo](https://github.com/MichaelGrupp/evo) and write them as `PoseStamped` messages to another bag in the current working directory.

# Blender DAE conversion
Re-export the DAE files in Blender to OBJ with Y-forward and Z-up.
