#!/bin/bash

source /opt/ros/humble/setup.bash

source ~/ws/install/setup.bash

ros2 launch stack_master teleop_launch.xml \
  racecar_version:=BLUE \
  use_vio:=false