"""Safety checks for control inputs before a drive command is computed."""

from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time

from controller.controller_manager import Controller


def controller_stub():
    warnings = []
    return SimpleNamespace(
        last_odom_time=10.0, last_frenet_time=10.0,
        last_waypoints_time=10.0, max_input_age_sec=0.3,
        max_odom_age_sec=0.3,
        last_odom_stamp_ns=10_000_000_000,
        last_frenet_stamp_ns=10_000_000_000,
        last_waypoints_stamp_ns=10_000_000_000,
        max_path_start_rear_m=0.75,
        position_in_map=np.array([[0.0, 0.0, 0.0]]),
        waypoint_array_in_map=np.array([[0.1, 0.0, 1.0]]),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_100_000_000)),
        get_logger=lambda: SimpleNamespace(
            warning=lambda message, **kwargs: warnings.append(message)),
        warnings=warnings)


def test_recent_path_near_car_is_accepted():
    controller = controller_stub()
    assert Controller.inputs_ready(controller, 10.1)


def test_delayed_path_is_rejected():
    controller = controller_stub()
    controller.last_waypoints_time = 9.0
    assert not Controller.inputs_ready(controller, 10.1)
    assert 'local waypoints is stale' in controller.warnings[0]


def test_recently_delivered_but_old_stamped_odometry_is_rejected():
    controller = controller_stub()
    controller.last_odom_stamp_ns = 9_000_000_000
    assert not Controller.inputs_ready(controller, 10.1)
    assert 'odometry is stale' in controller.warnings[0]


def test_path_starting_behind_car_is_rejected():
    controller = controller_stub()
    controller.waypoint_array_in_map = np.array([[-1.2, 0.1, 1.0]])
    assert not Controller.inputs_ready(controller, 10.1)
    assert 'behind car' in controller.warnings[0]


def test_front_obstacle_latches_stop_until_clear():
    controller = controller_stub()
    controller.front_guard_enabled = True
    controller.front_clearance_m = 0.30
    controller.front_release_m = 0.65
    controller.front_decel_mps2 = 2.0
    controller.front_reaction_sec = 0.15
    controller.front_stop_latched = False
    controller.last_scan_time = 10.0
    controller.last_scan_stamp_ns = 10_000_000_000
    controller.speed_now = 1.2
    controller.scan = SimpleNamespace(
        ranges=[0.75] * 25, angle_min=-0.24, angle_increment=0.02,
        range_min=0.05, range_max=20.0)

    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.speed_now = 0.0
    controller.scan.ranges = [0.30] * 25
    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.scan.ranges = [0.80] * 25
    assert not Controller.front_obstacle_stop(controller, 10.1)


def test_stop_command_keeps_last_steering_while_car_brakes():
    published = []
    controller = SimpleNamespace(
        mode='MAP', last_safe_steer=0.23,
        inputs_ready=lambda now: False,
        front_obstacle_stop=lambda now: False,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))),
        drive_pub=SimpleNamespace(publish=published.append))

    Controller.control_loop(controller)

    assert len(published) == 1
    assert published[0].drive.speed == 0.0
    assert published[0].drive.steering_angle == 0.23
