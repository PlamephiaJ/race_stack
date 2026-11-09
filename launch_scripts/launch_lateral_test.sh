#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/jetsonblue/ws/install/setup.bash

python3 /dev/fd/3 3<<'PY'
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import String

ROOT = Path('/home/jetsonblue/ws/src/race_stack/bags')
OUT = ROOT / ('lateral_test_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True, exist_ok=False)


class Tester(Node):
    def __init__(self):
        super().__init__('blue_lateral_tester')

        self.drive = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )
        self.marker = self.create_publisher(
            String, '/blue_sysid/event', 10
        )
        self.create_subscription(
            Joy, '/joy', self.on_joy, 10
        )
        self.create_timer(0.05, self.on_timer)

        self.rb = self.lb = False
        self.last_joy = 0.0
        self.mode = 'IDLE'
        self.released = False
        self.speed = self.angle = 0.0
        self.started = self.stopped = 0.0
        self.done = threading.Event()
        self.result = ''
        self.label = ''

    def on_joy(self, msg):
        self.rb = (
            len(msg.buttons) > 5 and msg.buttons[5] == 1
        )
        self.lb = (
            len(msg.buttons) > 4 and msg.buttons[4] == 1
        )
        self.last_joy = time.monotonic()

    def event(self, message):
        msg = String()
        msg.data = message
        self.marker.publish(msg)
        print(message, flush=True)

    def arm(self, speed, angle):
        self.speed, self.angle = speed, angle
        self.label = f'v={speed:.2f}, steer={angle:+.2f}'
        self.released = not self.rb
        self.result = ''
        self.done.clear()
        self.mode = 'WAIT_RB'
        self.event('ARM ' + self.label)

    def finish(self, result):
        self.mode = 'DONE'
        self.result = result
        self.event(result + ' ' + self.label)
        self.done.set()

    def on_timer(self):
        now = time.monotonic()
        authorized = (
            self.rb
            and not self.lb
            and now - self.last_joy < 0.35
        )

        if self.mode == 'WAIT_RB':
            if not self.rb:
                self.released = True

            if authorized and self.released:
                self.mode = 'RUN'
                self.started = now
                self.event('START ' + self.label)

        elif self.mode == 'RUN':
            if not authorized:
                self.finish('ABORT_RB_RELEASED')
            elif now - self.started >= 20.0:
                self.mode = 'STOP'
                self.stopped = now
                self.event('STOP_COMMAND ' + self.label)

        elif self.mode == 'STOP':
            if not authorized:
                self.finish('ABORT_DURING_STOP')
            elif now - self.stopped >= 3.0:
                self.finish('COMPLETE')

        command = AckermannDriveStamped()
        command.header.stamp = (
            self.get_clock().now().to_msg()
        )
        command.header.frame_id = 'base_link'

        if self.mode == 'RUN' and authorized:
            command.drive.speed = self.speed
            command.drive.steering_angle = self.angle
        else:
            command.drive.speed = 0.0
            command.drive.steering_angle = 0.0

        self.drive.publish(command)



def read_terminal_input(prompt):
    print(prompt, end="", flush=True)

    with open("/dev/tty", "r") as tty:
        line = tty.readline()

    if not line:
        raise RuntimeError("Terminal input unavailable")

    return line

def stop_process(proc, label):
    if proc is None or proc.poll() is not None:
        return

    print(f'Stopping {label}...', flush=True)

    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=12)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


launch = bag = None
node = None
rclpy_started = False
launch_log = bag_log = None

try:
    existing = subprocess.run(
        ['ros2', 'node', 'list'],
        capture_output=True,
        text=True,
        timeout=7
    )

    if '/actuation_manager' in existing.stdout.splitlines():
        raise RuntimeError(
            '已有车辆控制系统在运行，请先停止。'
        )

    launch_log = open(OUT / 'launch.log', 'w')

    launch = subprocess.Popen(
        [
            'ros2', 'launch',
            'stack_master', 'teleop_launch.xml',
            'racecar_version:=BLUE',
            'use_vio:=false'
        ],
        stdout=launch_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )

    rclpy.init()
    rclpy_started = True
    node = Tester()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True
    )
    spin_thread.start()

    required = {
        '/joy',
        '/odom',
        '/sensors/imu/raw',
        '/ackermann_cmd_applied'
    }

    for _ in range(80):
        if launch.poll() is not None:
            raise RuntimeError(
                'Teleop 启动失败，请查看 launch.log'
            )

        known = {
            name for name, _ in
            node.get_topic_names_and_types()
        }

        if required <= known:
            break

        time.sleep(0.25)
    else:
        raise RuntimeError(
            '话题未就绪: ' +
            ', '.join(sorted(required - known))
        )

    bag_log = open(OUT / 'record.log', 'w')

    topics = [
        '/joy',
        '/drive',
        '/ackermann_cmd_applied',
        '/odom',
        '/sensors/imu/raw',
        '/sensors/core',
        '/commands/servo/position',
        '/commands/motor/speed',
        '/blue_sysid/event'
    ]

    bag = subprocess.Popen(
        [
            'ros2', 'bag', 'record',
            '-o', str(OUT / 'rosbag'),
            *topics
        ],
        stdout=bag_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )

    time.sleep(3)

    if bag.poll() is not None:
        raise RuntimeError(
            '录包启动失败，请查看 record.log'
        )

    print(f'\nDATA: {OUT}\n')
    print('每组测试：')
    print('1. Enter 准备')
    print('2. 按住 RB 开始')
    print('3. 固定转向 20 秒')
    print('4. 自动发送零速度 3 秒')
    print('5. 完全停车后松开 RB')
    print('紧急中止：立即松开 RB\n')

    tests = [(0.7, 0.25)]

    for idx, (v, s) in enumerate(tests, 1):
        while True:
            action = read_terminal_input(
                f'[{idx}/1] '
                f'v={v:.1f} m/s, '
                f'steer={s:+.2f} rad\n'
                'Enter 开始 / s 跳过 / q 退出: '
            ).strip().lower()

            if action == 'q':
                raise KeyboardInterrupt

            if action == 's':
                node.event(
                    f'SKIP {idx}: v={v}, angle={s}'
                )
                break

            node.arm(v, s)

            print(
                '已准备。车辆静止、路径清空后，'
                '按住 RB 直到 COMPLETE。'
            )

            while not node.done.wait(0.2):
                if (
                    bag.poll() is not None
                    or launch.poll() is not None
                ):
                    raise RuntimeError(
                        'ROS 或 rosbag 意外退出'
                    )

            print(
                f'本组结果：{node.result}\n'
                '确认车辆停稳，然后松开 RB。'
            )

            if node.result == 'COMPLETE':
                break

            print(
                '本组中止，可重试或跳过。'
            )

    print('\n全部测试结束，保存数据。')

except KeyboardInterrupt:
    print(
        '\n测试取消。请松开 RB，确认车辆停止。'
    )

except Exception:
    import traceback
    traceback.print_exc()

finally:
    if node is not None:
        node.mode = 'IDLE'
        time.sleep(0.4)

    stop_process(bag, 'rosbag')
    stop_process(launch, 'teleop')

    if rclpy_started:
        rclpy.shutdown()

    if node is not None:
        node.destroy_node()

    if launch_log is not None:
        launch_log.close()

    if bag_log is not None:
        bag_log.close()

    print('Saved to:', OUT)
PY