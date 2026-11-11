"""Autonomous zero targets must fade through the final reference smoother."""

from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from builtin_interfaces.msg import Time
from controller.controller_manager import Controller
from controller.tracking_safety import RecoverySpeedLimiter, SpeedSmoother


def controller_stub():
    published, cycles = [], []
    stub = SimpleNamespace(
        mode='MAP', last_safe_steer=.23, speed_now=1.0, applied_speed=1.0,
        last_control_time=10.05, max_odom_age_sec=.3, max_input_age_sec=.5,
        last_odom_time=10.0, last_frenet_time=10.0, last_waypoints_time=10.0,
        last_scan_time=10.0, last_applied_time=10.0,
        last_odom_stamp_ns=10_000_000_000, last_frenet_stamp_ns=10_000_000_000,
        last_waypoints_stamp_ns=10_000_000_000, last_scan_stamp_ns=10_000_000_000,
        last_applied_stamp_ns=10_000_000_000,
        position_in_map=np.array([[0.0, 0.0, 0.0]]),
        position_in_map_frenet=np.array([0.0, 0.0, 1.0, 0.0]),
        waypoint_array_in_map=np.array([[.1, 0, 2.2], [2., 0, 2.2]]),
        front_guard_enabled=True, require_applied_feedback=True,
        front_clearance_m=.3, collision_speed_gain=.8, front_speed_cap=0.0,
        scan=SimpleNamespace(ranges=[10.0]*25, angle_min=-.24, angle_increment=.02,
                             range_min=.05, range_max=20.0),
        recovery_limiter=RecoverySpeedLimiter(), speed_smoother=SpeedSmoother(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=10_100_000_000, to_msg=lambda: Time(sec=10, nanosec=100_000_000))),
        get_logger=lambda: SimpleNamespace(warning=lambda *a, **kw: None),
        drive_pub=SimpleNamespace(publish=published.append),
        map_cycle=lambda **kw: (cycles.append(kw) or (2.2, .2)),
        pp_cycle=lambda: (cycles.append({}) or (2.2, .2)),
        published=published, cycles=cycles)
    stub.inputs_ready=lambda now: Controller.inputs_ready(stub, now)
    stub.input_speed_scale=lambda now: Controller.input_speed_scale(stub, now)
    stub.front_collision_speed=lambda now: Controller.front_collision_speed(stub, now)
    return stub


def tick(stub):
    with patch('controller.controller_manager.time.monotonic', lambda: 10.1):
        Controller.control_loop(stub)
    return stub.published[-1].drive


def test_recent_valid_inputs_use_smooth_reference():
    stub=controller_stub()
    assert stub.input_speed_scale(10.1) == 1
    command=tick(stub)
    assert command.speed == pytest.approx(1.05)
    assert command.steering_angle == .2


def test_rear_path_start_does_not_create_zero_speed_pulses():
    stub=controller_stub()
    stub.waypoint_array_in_map=np.array([[-1.2, 0, 2.2], [2., 0, 2.2]])
    command=tick(stub)
    assert command.speed > 1.0
    assert stub.cycles


def test_ttc_below_one_second_still_uses_linear_speed_not_a_stop_latch():
    stub=controller_stub()
    stub.speed_now=stub.applied_speed=2.0
    stub.scan.ranges=[.8]*25
    command=tick(stub)
    assert command.speed == pytest.approx(1.95)
    assert stub.front_speed_cap == pytest.approx(.24)
    assert command.steering_angle == .2


def test_aging_path_scales_speed_continuously_before_expiration():
    stub=controller_stub()
    stub.last_waypoints_stamp_ns=9_725_000_000
    assert stub.input_speed_scale(10.1) == pytest.approx(.5)
    assert tick(stub).speed > 0


@pytest.mark.parametrize('stream', ['odom', 'frenet', 'waypoints', 'scan', 'applied'])
def test_expired_input_fades_instead_of_publishing_sudden_zero(stream):
    stub=controller_stub()
    setattr(stub, 'last_'+stream+'_stamp_ns', 9_000_000_000)
    command=tick(stub)
    assert command.speed == pytest.approx(.95)
    assert command.steering_angle == .23
    assert not stub.cycles


def test_invalid_geometry_fades_instead_of_publishing_sudden_zero():
    stub=controller_stub()
    stub.waypoint_array_in_map=np.array([])
    command=tick(stub)
    assert command.speed == pytest.approx(.95)
    assert command.steering_angle == .23


def test_missing_front_beams_requests_smooth_decay():
    stub=controller_stub()
    stub.scan.ranges=[float('nan')]*25
    command=tick(stub)
    assert command.speed == pytest.approx(.95)


def test_reference_slew_applies_to_pp_as_well():
    stub=controller_stub();stub.mode='PP'
    assert tick(stub).speed == pytest.approx(1.05)


def test_all_path_behind_vehicle_fades_without_chasing_a_rear_target():
    stub=controller_stub()
    stub.waypoint_array_in_map=np.array([[-2., 0, 2.2], [-1., 0, 2.2]])
    command=tick(stub)
    assert command.speed == pytest.approx(.95)
    assert command.steering_angle == .23
    assert not stub.cycles
