#!/bin/bash

source /opt/ros/humble/setup.bash

source ~/ws/install/setup.bash

ros2 launch stack_master time_trials_launch.xml \
  racecar_version:=BLUE \
  ctrl_algo:=PP \
  sim_opp:=true

# ros2 launch stack_master head_to_head_launch.xml \
#   racecar_version:=BLUE \
#   ctrl_algo:=PP \
#   sim_opp:=true