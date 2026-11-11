"""Safety checks for control inputs before a drive command is computed."""

from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time

from controller.controller_manager import Controller
from controller.tracking_safety import RecoverySpeedLimiter


def controller_stub():
    warnings = []
    stub = SimpleNamespace(
        last_odom_time=10.0, last_frenet_time=10.0,
        last_waypoints_time=10.0, max_input_age_sec=0.3,
        max_odom_age_sec=0.3,
        last_odom_stamp_ns=10_000_000_000,
        last_frenet_stamp_ns=10_000_000_000,
        last_waypoints_stamp_ns=10_000_000_000,
        max_path_start_rear_m=0.75,
        front_speed_margin=0.8, front_speed_cap=float('inf'),
        front_ttc_stop_sec=1.0, front_ttc_release_sec=1.2,
        position_in_map=np.array([[0.0, 0.0, 0.0]]),
        position_in_map_frenet=np.array([0.0, 0.0, 1.0, 0.0]),
        waypoint_array_in_map=np.array([[0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]),
        speed_now=1.0,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_100_000_000)),
        get_logger=lambda: SimpleNamespace(
            warning=lambda message, **kwargs: warnings.append(message)),
        warnings=warnings)
    stub.path_start_too_far_behind = lambda: Controller.path_start_too_far_behind(stub)
    return stub


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
    controller.waypoint_array_in_map = np.array([[-1.2, 0.1, 1.0], [0.5, 0.1, 1.0]])
    assert not Controller.inputs_ready(controller, 10.1)
    assert 'behind car' in controller.warnings[0]


def test_front_obstacle_latches_stop_until_clear():
    controller = controller_stub()
    controller.front_guard_enabled = True
    controller.front_clearance_m = 0.30
    controller.front_decel_mps2 = 2.0
    controller.front_stop_latched = False
    controller.require_applied_feedback = True
    controller.applied_speed = 1.2
    controller.last_applied_time = 10.0
    controller.last_applied_stamp_ns = 10_000_000_000
    controller.last_scan_time = 10.0
    controller.last_scan_stamp_ns = 10_000_000_000
    controller.speed_now = 1.2
    controller.scan = SimpleNamespace(
        ranges=[0.75] * 25, angle_min=-0.24, angle_increment=0.02,
        range_min=0.05, range_max=20.0)

    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.speed_now = 0.0
    controller.applied_speed = 0.0
    controller.scan.ranges = [0.30] * 25
    assert not Controller.front_obstacle_stop(controller, 10.1)
    assert controller.front_speed_cap == 0.0
    controller.scan.ranges = [0.80] * 25
    assert not Controller.front_obstacle_stop(controller, 10.1)


def front_stub():
    controller = controller_stub()
    controller.front_guard_enabled = True
    controller.front_clearance_m = 0.3
    controller.front_decel_mps2 = 0.8
    controller.front_stop_latched = False
    controller.require_applied_feedback = True
    controller.applied_speed = 1.5
    controller.last_applied_time = controller.last_scan_time = 10.0
    controller.last_applied_stamp_ns = controller.last_scan_stamp_ns = 10_000_000_000
    controller.scan = SimpleNamespace(
        ranges=[1.8] * 25, angle_min=-0.24, angle_increment=0.02,
        range_min=0.05, range_max=20.0)
    return controller


def test_applied_reference_is_not_treated_as_current_speed_for_ttc():
    controller = front_stub()
    controller.speed_now = 0.3
    assert not Controller.front_obstacle_stop(controller, 10.1)
    assert controller.front_speed_cap < controller.applied_speed


def test_ttc_recovery_releases_stop_while_car_is_still_moving():
    controller = front_stub()
    controller.speed_now = 2.0
    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.speed_now = controller.applied_speed = 0.4
    controller.scan.ranges = [1.5] * 25
    assert not Controller.front_obstacle_stop(controller, 10.1)
    controller.speed_now = controller.applied_speed = 0.0
    assert not Controller.front_obstacle_stop(controller, 10.1)


def test_one_second_ttc_gate_and_release_hysteresis():
    controller = front_stub()
    controller.speed_now = 1.0
    controller.last_scan_time = 10.1
    controller.last_scan_stamp_ns = 10_100_000_000
    controller.scan.ranges = [1.0] * 25
    assert not Controller.front_obstacle_stop(controller, 10.1)
    controller.scan.ranges = [.99] * 25
    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.scan.ranges = [1.1] * 25
    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.scan.ranges = [1.21] * 25
    assert not Controller.front_obstacle_stop(controller, 10.1)


def test_missing_actuation_feedback_stops_real_car_but_not_direct_drive_sim():
    controller = front_stub()
    controller.applied_speed = None
    assert Controller.front_obstacle_stop(controller, 10.1)
    controller.require_applied_feedback = False
    controller.speed_now = 0.3
    assert not Controller.front_obstacle_stop(controller, 10.1)


def test_old_applied_timestamp_stops_even_if_recently_received():
    controller = front_stub()
    controller.last_applied_stamp_ns = 9_000_000_000
    assert Controller.front_obstacle_stop(controller, 10.1)


def test_stop_command_keeps_last_steering_while_car_brakes():
    published = []
    controller = SimpleNamespace(
        mode='MAP', last_safe_steer=0.23,
        last_control_time=0.0, recovery_limiter=RecoverySpeedLimiter(),
        inputs_ready=lambda now, **kwargs: False,
        front_obstacle_stop=lambda now: False,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))),
        drive_pub=SimpleNamespace(publish=published.append))

    Controller.control_loop(controller)

    assert len(published) == 1
    assert published[0].drive.speed == 0.0
    assert published[0].drive.steering_angle == 0.23


def test_front_stop_keeps_computing_steering_from_fresh_path():
    for mode in ('MAP', 'PP'):
        published = []
        calls = []
        controller = SimpleNamespace(
            mode=mode, last_safe_steer=0.03, speed_now=1.4, front_speed_cap=float('inf'),
            last_control_time=0.0, recovery_limiter=RecoverySpeedLimiter(),
            position_in_map=np.array([[0.0, 0.0, 0.0]]),
            waypoint_array_in_map=np.array([[0.1, 0.0, 2.2], [1.0, 0.3, 2.2]]),
            inputs_ready=lambda now, **kwargs: True,
            front_obstacle_stop=lambda now: True,
            path_start_too_far_behind=lambda: False,
            map_cycle=lambda **kwargs: (calls.append(kwargs) or (2.2, 0.2)),
            pp_cycle=lambda: (calls.append({}) or (2.2, 0.2)),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))),
            drive_pub=SimpleNamespace(publish=published.append))
        Controller.control_loop(controller)
        assert calls == ([{'braking': True}] if mode == 'MAP' else [{}])
        assert published[0].drive.speed == 0.0
        assert published[0].drive.steering_angle == 0.2


def test_rear_path_start_stops_speed_but_updates_steer_on_remaining_path():
    published = []
    controller = SimpleNamespace(
        mode='MAP', last_safe_steer=0.03, speed_now=1.4, front_speed_cap=float('inf'),
        last_control_time=0.0, recovery_limiter=RecoverySpeedLimiter(),
        position_in_map=np.array([[0.0, 0.0, 0.0]]),
        waypoint_array_in_map=np.array([[-0.9, 0.0, 2.2], [1.0, 0.3, 2.2]]),
        inputs_ready=lambda now, **kwargs: True,
        front_obstacle_stop=lambda now: False,
        path_start_too_far_behind=lambda: True,
        map_cycle=lambda **kwargs: (2.2, 0.2),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))),
        drive_pub=SimpleNamespace(publish=published.append))
    Controller.control_loop(controller)
    assert published[0].drive.speed == 0.0
    assert published[0].drive.steering_angle == 0.2


def test_controller_applies_continuous_front_cap_before_emergency_gate():
    published = []
    controller = SimpleNamespace(
        mode='MAP', last_safe_steer=0.03, speed_now=0.7, front_speed_cap=0.8,
        last_control_time=0.0, recovery_limiter=RecoverySpeedLimiter(),
        position_in_map=np.array([[0.0, 0.0, 0.0]]),
        waypoint_array_in_map=np.array([[0.1, 0.0, 2.2], [1.0, 0.0, 2.2]]),
        inputs_ready=lambda now, **kwargs: True,
        front_obstacle_stop=lambda now: False,
        path_start_too_far_behind=lambda: False,
        map_cycle=lambda **kwargs: (2.2, 0.2),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=10))),
        drive_pub=SimpleNamespace(publish=published.append))
    Controller.control_loop(controller)
    assert published[0].drive.speed == 0.8
    assert published[0].drive.steering_angle == 0.2


def test_continuous_governor_avoids_stop_start_cycle_with_delayed_vehicle_response():
    # Synthetic bend/wall range around the 1.1-1.7 m seen in 022513, not a
    # trajectory simulation. Command slew and wheel response close the loop.
    import math

    def run(proactive):
        controller = front_stub()
        applied = wheel = 0.0
        last_request = 0.0
        stops = 0
        moving = []
        for k in range(1200):
            dt = .025
            now = 10.0 + k * dt
            ns = round(now * 1e9)
            controller.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=ns))
            controller.last_applied_time = controller.last_scan_time = now - .02
            controller.last_applied_stamp_ns = controller.last_scan_stamp_ns = ns - 20_000_000
            controller.speed_now, controller.applied_speed = wheel, applied
            controller.scan.ranges = [1.3 + .15 * math.sin(k * dt * .5)] * 25
            emergency = Controller.front_obstacle_stop(controller, now)
            request = 0.0 if emergency else min(1.2, controller.front_speed_cap if proactive else 1.2)
            stops += last_request > 0 and request == 0
            last_request = request
            applied += float(np.clip(request - applied, -2.5 * dt, 2.5 * dt))
            # Delayed response and 11% speed overshoot; effective braking .8.
            wheel += float(np.clip((1.11 * applied - wheel) * dt / .35, -.8 * dt, 2.5 * dt))
            if k > 100:
                moving.append(wheel)
        return stops, min(moving)

    without_cap, _ = run(False)
    with_cap, minimum_speed = run(True)
    assert without_cap >= 3
    assert with_cap == 0
    assert minimum_speed > .5
