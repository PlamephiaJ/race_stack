#/bin/bash

source ~/ws/install/setup.bash

# ros2 launch stack_master time_trials_launch.xml \
#   racecar_version:=BLUE \
#   ctrl_algo:=PP

ros2 launch stack_master head_to_head_launch.xml \
  racecar_version:=BLUE \
  ctrl_algo:=PP