#!/usr/bin/env bash

set -eo pipefail

# ==========================================
# BLUE Brake Test
# ==========================================

WORKSPACE="/home/jetsonblue/ws"
BAG_ROOT="$WORKSPACE/src/race_stack/bags"
TEST_NAME="brake_test_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$BAG_ROOT/$TEST_NAME"

source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"

mkdir -p "$OUTPUT_DIR"

LAUNCH_PID=""
BAG_PID=""

# ==========================================
# Process group management
# ==========================================

group_alive() {
    local pgid="$1"

    ps -eo pgid=,stat= | awk -v g="$pgid" '
        $1 == g && $2 !~ /^Z/ { found=1 }
        END { exit !found }
    '
}

stop_group() {
    local pgid="$1"
    local name="$2"

    [[ -z "$pgid" ]] && return 0

    if ! group_alive "$pgid"; then
        wait "$pgid" 2>/dev/null || true
        return 0
    fi

    echo "Stopping $name (PGID=$pgid)..."

    # Stage 1: Graceful shutdown
    kill -INT -- "-$pgid" 2>/dev/null || true

    for i in $(seq 1 40); do
        if ! group_alive "$pgid"; then
            break
        fi
        sleep 0.25
    done

    # Stage 2: SIGTERM
    if group_alive "$pgid"; then
        echo "SIGINT timeout, sending SIGTERM..."
        kill -TERM -- "-$pgid" 2>/dev/null || true

        for i in $(seq 1 20); do
            if ! group_alive "$pgid"; then
                break
            fi
            sleep 0.25
        done
    fi

    # Stage 3: Force kill remaining processes
    if group_alive "$pgid"; then
        echo "SIGTERM timeout, sending SIGKILL..."
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi

    wait "$pgid" 2>/dev/null || true

    echo "$name stopped."
}

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM
    set +e

    echo
    echo "=========================================="
    echo "Stopping BLUE Brake Test"
    echo "=========================================="

    # Finalize rosbag first
    if [[ -n "$BAG_PID" ]]; then
        echo "[1/2] Stopping rosbag..."
        stop_group "$BAG_PID" "rosbag"
    fi

    # Stop vehicle launch and all child processes
    if [[ -n "$LAUNCH_PID" ]]; then
        echo "[2/2] Stopping teleop..."
        stop_group "$LAUNCH_PID" "teleop"
    fi

    echo
    echo "=========================================="
    echo "Test finished"
    echo "Output: $OUTPUT_DIR"

    if [[ -f "$OUTPUT_DIR/rosbag/metadata.yaml" ]]; then
        echo "ROS bag metadata: OK"
    else
        echo "WARNING: ROS bag metadata missing"
    fi

    echo "=========================================="

    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ==========================================
# 1. Launch BLUE Teleop
# ==========================================

echo "=========================================="
echo "BLUE Brake Test"
echo "=========================================="
echo "Vehicle: BLUE"
echo "Target speed: 0.7 m/s"
echo "Output: $OUTPUT_DIR"
echo

if pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch stack_master teleop_launch.xml' >/dev/null; then
    echo "ERROR: BLUE teleop launch is already running."
    echo "Stop the existing launch first."
    exit 1
fi

echo "[1/3] Launching BLUE teleop..."

# Independent process group
setsid ros2 launch stack_master teleop_launch.xml \
    racecar_version:=BLUE \
    use_vio:=false \
    > "$OUTPUT_DIR/launch.log" 2>&1 &

LAUNCH_PID=$!

sleep 1

if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "ERROR: ROS launch failed."
    tail -n 50 "$OUTPUT_DIR/launch.log"
    exit 1
fi

# Verify that setsid created an isolated process group
ACTUAL_PGID=$(ps -o pgid= -p "$LAUNCH_PID" | tr -d ' ')

if [[ "$ACTUAL_PGID" != "$LAUNCH_PID" ]]; then
    echo "ERROR: Launch process group isolation failed."
    exit 1
fi

# ==========================================
# 2. Wait for ROS topics
# ==========================================

echo "[2/3] Waiting for vehicle topics..."

READY=0

for i in $(seq 1 60); do

    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "ERROR: Teleop exited unexpectedly."
        tail -n 50 "$OUTPUT_DIR/launch.log"
        exit 1
    fi

    TOPICS="$(ros2 topic list 2>/dev/null || true)"

    if printf '%s\n' "$TOPICS" | grep -Fxq "/ackermann_cmd_applied" && \
       printf '%s\n' "$TOPICS" | grep -Fxq "/odom"; then
        READY=1
        break
    fi

    sleep 1
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: Required topics not available."
    exit 1
fi

echo "Vehicle topics ready."

# ==========================================
# 3. Start recording
# ==========================================

echo "[3/3] Starting ROS bag..."

ros2 topic list -t > "$OUTPUT_DIR/topics.txt"

ros2 topic info /commands/motor/brake -v \
    > "$OUTPUT_DIR/brake_topic_info.txt" 2>&1 || true

CANDIDATES=(
    /joy
    /teleop
    /drive
    /ackermann_cmd_applied
    /odom
    /state_estimation/odom
    /sensors/core
    /sensors/imu/raw
    /imu
    /commands/motor/speed
    /commands/motor/brake
    /commands/servo/position
    /sensors/servo_position_command
)

AVAILABLE="$(ros2 topic list)"
RECORD_TOPICS=()

for TOPIC in "${CANDIDATES[@]}"; do
    if printf '%s\n' "$AVAILABLE" | grep -Fxq "$TOPIC"; then
        RECORD_TOPICS+=("$TOPIC")
        echo "Recording: $TOPIC"
    fi
done

# Independent process group
setsid ros2 bag record \
    -o "$OUTPUT_DIR/rosbag" \
    "${RECORD_TOPICS[@]}" \
    > "$OUTPUT_DIR/record.log" 2>&1 &

BAG_PID=$!

sleep 2

if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "ERROR: ROS bag recording failed."
    cat "$OUTPUT_DIR/record.log"
    exit 1
fi

ACTUAL_PGID=$(ps -o pgid= -p "$BAG_PID" | tr -d ' ')

if [[ "$ACTUAL_PGID" != "$BAG_PID" ]]; then
    echo "ERROR: Bag process group isolation failed."
    exit 1
fi

echo
echo "=========================================="
echo "BLUE BRAKE TEST READY"
echo "=========================================="
echo
echo "Test A: Joystick Neutral"
echo "  1. Hold LB"
echo "  2. Accelerate to 0.7 m/s"
echo "  3. Return joystick to neutral"
echo "  4. Measure stopping distance"
echo "  5. Repeat 3 times"
echo
echo "Test B: Release LB"
echo "  1. Hold LB"
echo "  2. Accelerate to 0.7 m/s"
echo "  3. Release LB"
echo "  4. Measure stopping distance"
echo "  5. Repeat 3 times"
echo
echo "Press Ctrl+C after the vehicle has stopped."
echo "=========================================="

# Wait until recording finishes or user interrupts
wait "$BAG_PID"