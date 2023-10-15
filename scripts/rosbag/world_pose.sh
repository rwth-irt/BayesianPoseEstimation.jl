#! /bin/sh
. venv/bin/activate
# extract tracked object and julia pf in world coordinates
evo_traj bag -c evo_config.json $1 /tf:world.coordinate_pf /tf:world.tracked_object /tf:world.filtered_object /tf:world.charuco --save_as_tum
