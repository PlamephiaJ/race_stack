#!/bin/bash

source /opt/ros/humble/setup.bash
source ~/ws/install/setup.bash

ros2 launch stack_master base_system_launch.xml \
  sim:=true \
  racecar_version:=BLUE \
  map_name:=baku \
  use_vio:=false