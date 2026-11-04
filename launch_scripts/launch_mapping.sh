#!/bin/bash

source /opt/ros/humble/setup.bash

source ~/ws/install/setup.bash

ros2 launch stack_master mapping_launch.xml   racecar_version:=BLUE   map_name:=track_01   use_vio:=false