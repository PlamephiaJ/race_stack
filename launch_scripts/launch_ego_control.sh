#/bin/bash

source /opt/ros/humble/setup.bash

source ~/ws/install/setup.bash

# ros2 launch stack_master time_trials_launch.xml \
#   racecar_version:=BLUE \
#   ctrl_algo:=PP

# ros2 launch stack_master head_to_head_launch.xml \
#   racecar_version:=BLUE \
#   ctrl_algo:=MAP

ros2 launch stack_master ftg_launch.xml racecar_version:=BLUE sim_opp:=false