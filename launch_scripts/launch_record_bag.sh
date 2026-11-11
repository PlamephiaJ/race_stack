#!/usr/bin/env bash
# Record the command chain and its inputs while reproducing a control fault.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd -- "$REPO_DIR/../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$WORKSPACE_DIR/install/setup.bash"

RUN_DIR="$REPO_DIR/bags/control_fault_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_DIR"

TOPICS=(
    /drive
    /ackermann_cmd_applied
    /commands/servo/position
    /commands/motor/speed
    /sensors/servo_position_command
    /joy
    /teleop
    /state
    /local_waypoints
    /global_waypoints_scaled
    /perception/obstacles
    /state_estimation/odom
    /car_state/frenet/odom
    /odom
    /tf
    /tf_static
    /scan
    /imu
    /sensors/imu/raw
    /tracked_pose
    /tracked_pose/with_covariance
    /pf/laser_global_pose
    /sensors/core
    /rosout
)

printf '%s\n' "${TOPICS[@]}" > "$RUN_DIR/requested_topics.txt"
git -C "$REPO_DIR" rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || true

echo "Recording to: $RUN_DIR/rosbag"
echo "Start before driving. After a fault, keep recording for at least 5 seconds."
echo "Press Ctrl-C to stop and finalize the bag."

exec ros2 bag record -o "$RUN_DIR/rosbag" "${TOPICS[@]}"
