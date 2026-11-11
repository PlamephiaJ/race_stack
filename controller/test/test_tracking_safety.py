"""Regression cases for offset starts and the measured BLUE stopping chain."""

import math

import numpy as np
import pytest

from controller.tracking_safety import RecoverySpeedLimiter, SpeedSmoother, collision_speed_limit, freshness_scale, path_tracking_errors
from controller.speed_profiles import ftg_speed_target
from controller.map import MAP_Controller
from types import SimpleNamespace


def test_polyline_errors_use_transformed_geometry_and_ignore_unrotated_heading():
    points = np.array([[0, 0, 2.2, 0], [2, 0, 2.2, 0]])
    pose = np.array([0.8, 0.79, math.radians(36)])
    original = path_tracking_errors(pose, points)
    angle = 1.8
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    points[:, :2] = points[:, :2] @ rotation.T + [10, -7]
    pose[:2] = pose[:2] @ rotation.T + [10, -7]
    pose[2] += angle
    assert path_tracking_errors(pose, points) == pytest.approx(original)
    assert original == pytest.approx((0.79, math.radians(36), -math.sin(math.radians(36))))


def test_converging_offset_start_gets_interpolated_speed_instead_of_crawl():
    limiter = RecoverySpeedLimiter()
    errors = path_tracking_errors([0.8, -0.79, math.radians(36)], [[0, 0], [2, 0]])
    assert errors[2] > 0
    assert limiter.limit(2.23, errors, 0.025) == pytest.approx(1.5680869565)


@pytest.mark.parametrize('side', [-1, 1])
def test_same_offset_and_absolute_heading_distinguish_convergence_from_divergence(side):
    points = [[0, 0], [2, 0]]
    converging = path_tracking_errors([0.8, side * .79, -side * math.radians(36)], points)
    diverging = path_tracking_errors([0.8, side * .79, side * math.radians(36)], points)
    assert converging[:2] == pytest.approx(diverging[:2])
    assert RecoverySpeedLimiter().limit(2.23, converging, .025) > 1.5
    assert RecoverySpeedLimiter().limit(2.23, diverging, .025) == .5


def test_correct_direction_recovers_without_waiting_for_old_tight_exit_thresholds():
    limiter = RecoverySpeedLimiter()
    assert limiter.limit(2.23, (.79, math.radians(36), -.6), .025) == .5
    # Still outside 0.15 m/10 degrees, but now returning in the right direction.
    assert limiter.limit(2.23, (.4, math.radians(20), .3), .025) == pytest.approx(.525)
    for _ in range(100):
        speed = limiter.limit(2.23, (.4, math.radians(20), .3), .025)
    assert speed > 2.0


def test_parallel_offset_and_small_outward_heading_noise_use_interpolation():
    for heading in [0, math.radians(5)]:
        errors = path_tracking_errors([.8, .5, heading], [[0, 0], [2, 0]])
        assert RecoverySpeedLimiter().limit(2.23, errors, .025) > 1.9


def test_wrong_way_caps_at_half_metre_per_second_but_respects_lower_requests():
    limiter = RecoverySpeedLimiter()
    errors = path_tracking_errors([.8, .1, math.pi], [[0, 0], [2, 0]])
    assert limiter.limit(2.23, errors, .025) == .5
    assert limiter.limit(.2, errors, .025) == .2
    assert limiter.limit(0.0, errors, .025) == 0.0


def test_threshold_noise_does_not_switch_between_full_speed_and_crawl():
    limiter = RecoverySpeedLimiter()
    speeds = [limiter.limit(2.23, (distance, 0, 0), .025)
              for distance in [.34, .36, .34, .36, .34]]
    assert min(speeds) > 2.2
    assert max(abs(np.diff(speeds))) < .025


def test_invalid_path_restarts_slow_and_scheduler_pause_cannot_release_cap():
    limiter = RecoverySpeedLimiter()
    assert limiter.limit(2.23, None, 0.025) == 0
    assert limiter.limit(2.23, (0.0, 0.0, 0.0), 10.0) == pytest.approx(0.6)
    assert path_tracking_errors([0, 0, 0], [[0, 0], [0, 0]]) is None
    assert path_tracking_errors([0, 0, 0], [[0, 0], [float('nan'), 0]]) is None


def test_collision_speed_is_linear_without_ttc_thresholds_or_crawl_floor():
    assert collision_speed_limit(1.3, 1.0, .0) == pytest.approx(.8)
    assert collision_speed_limit(1.4, 1.0, .0) == pytest.approx(.88)
    assert collision_speed_limit(1.3, 1.0, .1) == pytest.approx(.72)
    assert collision_speed_limit(.4, 2.0, .0) == pytest.approx(.08)
    assert collision_speed_limit(.3, 2.0, .0) == 0.0


@pytest.mark.parametrize('kwargs', [{'gain': 0}, {'distance': float('nan')}, {'scan_age': -1}])
def test_invalid_collision_parameters_rejected(kwargs):
    values=dict(distance=1.0, speed=1.0, scan_age=.02, clearance=.3, gain=.8)
    values.update(kwargs)
    with pytest.raises(ValueError):
        collision_speed_limit(**values)


def test_freshness_is_a_continuous_linear_scale():
    assert freshness_scale(.2, .5) == 1
    assert freshness_scale(.25, .5) == 1
    assert freshness_scale(.375, .5) == .5
    assert freshness_scale(.5, .5) == 0
    assert freshness_scale(float('inf'), .5) == 0


def test_final_smoother_limits_every_jump_including_zero_and_nan_targets():
    smoother=SpeedSmoother()
    last=smoother.update(2.2, .025, 1.0)
    for target in [0.0, 2.2, .1, 2.0, float('nan'), 0.0]:
        speed=smoother.update(target, .025)
        assert abs(speed-last) <= .025 + 1e-10
        last=speed
    assert last > 0


def test_smoother_does_not_wind_up_behind_disabled_actuation():
    smoother=SpeedSmoother()
    for _ in range(100):
        speed=smoother.update(2.2, .025, 0.0)
    assert speed <= .1
    assert smoother.update(2.2, .025, .1) <= .125


def test_scheduler_pause_does_not_turn_a_zero_target_into_abrupt_stop():
    smoother=SpeedSmoother()
    smoother.speed=2.0
    assert smoother.update(0.0, 10.0) == pytest.approx(1.9)


def test_map_braking_lookup_uses_measured_speed_instead_of_path_target():
    looked_up = []
    stub = SimpleNamespace(
        state='GB_TRACK', opponent=None, speed_now=1.4, steering_speed=1.4,
        speed_lookahead_for_steer=0.0, position_in_map=np.array([[0, 0, 0]]),
        waypoint_array_in_map=np.array([[0.1, 0, 2.23], [1.0, 0.2, 2.23]]),
        nearest_waypoint=lambda *a: 0,
        speed_adjust_lat_err=lambda v, error: v,
        steer_lookup=SimpleNamespace(lookup_steer_angle=lambda a, v: (looked_up.append(v) or 0.1)),
        acc_scaling=lambda steer: steer,
        speed_steer_scaling=lambda steer, speed: steer,
        curr_steering_angle=0.0,
        logger_info=lambda *a: None, logger_warn=lambda *a: None)
    angle = MAP_Controller.calc_steering_angle(stub, np.array([1.0, .2]), 1.0, 0.0, 0.0, [1.4, 0])
    assert looked_up == [1.4]
    assert angle > 0
