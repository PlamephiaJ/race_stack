"""Verify FTG configuration affects control without opening DDS publishers."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.parameter import Parameter

from controller.follow_the_gap import ReactiveFollowGap
from controller.ftg import FTG_Controller


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def node_factory(monkeypatch):
    """Exercise real constructors and methods with only ROS I/O replaced."""
    overrides = {}
    declarations = {}

    def declare(node, name, default, descriptor=None):
        declarations[name] = descriptor
        return Parameter(name, value=overrides.get(name, default))

    monkeypatch.setattr(Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(Node, 'declare_parameter', declare)
    monkeypatch.setattr(Node, 'create_publisher', lambda *args: Mock())
    monkeypatch.setattr(Node, 'create_subscription', lambda *args: Mock())
    monkeypatch.setattr(Node, 'get_logger', lambda *args: Mock())
    monkeypatch.setattr(Node, 'get_clock', lambda *args: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=Time)))

    def create(values=None):
        overrides.clear()
        overrides.update(values or {})
        node = ReactiveFollowGap()
        assert all(descriptor.read_only for descriptor in declarations.values())
        return node

    return create


def scan(ranges, increment=0.01):
    return SimpleNamespace(
        ranges=ranges, angle_increment=increment,
        angle_min=-len(ranges) / 2 * increment)


def legacy(**tuning):
    return FTG_Controller(
        mapping=False, debug=False, safety_radius=4, max_lidar_dist=10.0,
        max_speed=5.0, range_offset=180, track_width=2.0, tuning=tuning)


def test_yaml_lists_all_standalone_tuning_and_preserves_defaults(node_factory):
    params = yaml.safe_load((ROOT / 'stack_master/config/follow_the_gap_params.yaml').read_text())['/**']['ros__parameters']
    # Calibration stays in the vehicle's vesc.yaml; optional overrides are comments.
    vehicle_calibration = {
        'steering_angle_to_servo_gain', 'steering_angle_to_servo_offset',
        'servo_min', 'servo_max',
    }
    assert set(params) == set(ReactiveFollowGap.PARAMETER_DEFAULTS) - vehicle_calibration
    assert params['max_speed'] == 4.0
    for name, value in params.items():
        if name != 'max_speed':
            assert value == ReactiveFollowGap.PARAMETER_DEFAULTS[name]
    node = node_factory(params)
    assert node.max_speed == 4.0


@pytest.mark.parametrize('values', [
    {'max_speed': 0.0}, {'max_deceleration': float('nan')},
    {'bubble_radius': -0.1}, {'preprocess_conv_size': 0},
    {'scan_range_offset': -1}, {'front_distance_index': -2},
    {'gap_weight_power': -1.0}, {'minimum_gap_angle': -1.0},
    {'speed_far_distance': 1.0}, {'max_command_dt': 0.0},
    {'stop_publish_count': 0}, {'publish_queue_depth': 0},
    {'servo_limit_margin': 0.5}, {'steering_angle_to_servo_gain': 0.0},
    {'servo_min': 0.9},
])
def test_invalid_parameters_fail_at_startup(node_factory, values):
    with pytest.raises(ValueError):
        node_factory(values)


def test_filter_window_and_distance_clip_are_used(node_factory):
    node = node_factory({'preprocess_conv_size': 1, 'max_distance': 2.0})
    assert node.preprocess_lidar([1.0, 4.0, 1.0]).tolist() == [1.0, 2.0, 1.0]
    node = node_factory({'preprocess_conv_size': 3, 'max_distance': 10.0})
    assert node.preprocess_lidar([1.0, 4.0, 1.0])[1] == pytest.approx(2.0)


def test_crop_gap_width_and_centre_weight_are_used(node_factory):
    ranges = np.array([0, 1, 3, 3, 0, 0, 0, 3, 3, 0], dtype=float)
    node = node_factory({'scan_range_offset': 0, 'minimum_gap_angle': 0.0})
    assert node.find_best_point(ranges, 0.1) == 2
    assert node_factory({'scan_range_offset': 3, 'minimum_gap_angle': 0.0}).find_best_point(ranges, 0.1) == 3
    narrow_deep_gap = np.array([0, 4, 4, 0, 0, 3, 3, 3, 3, 0], dtype=float)
    assert node.find_best_point(narrow_deep_gap, 0.1) == 2
    wider = node_factory({'scan_range_offset': 0, 'minimum_gap_angle': 0.3})
    assert wider.find_best_point(narrow_deep_gap, 0.1) == 6
    ranges = np.array([0, 1, 1, 4, 0], dtype=float)
    assert node.find_best_point(ranges, 0.1) == 3
    assert node_factory({'scan_range_offset': 0, 'gap_weight_power': 0.0}).find_best_point(ranges, 0.1) == 2


def test_bubble_radius_and_disparity_threshold_are_used(node_factory):
    values = {'scan_range_offset': 0, 'bubble_radius': 0.25}
    ranges = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    unchanged = ranges.copy()
    node_factory(dict(values, disparity_thresh=2.0)).disparity_extender(unchanged, 0.1)
    assert np.array_equal(unchanged, ranges)
    changed = ranges.copy()
    node_factory(values).disparity_extender(changed, 0.1)
    assert np.count_nonzero(changed) < np.count_nonzero(ranges)
    unchanged = ranges.copy()
    node_factory(dict(values, bubble_radius=0.0)).disparity_extender(unchanged, 0.1)
    assert np.array_equal(unchanged, ranges)
    small, large = np.full(20, 2.0), np.full(20, 2.0)
    node_factory(dict(values, bubble_radius=0.1)).closest_point_zero_out(small, 0.1)
    node_factory(dict(values, bubble_radius=0.8)).closest_point_zero_out(large, 0.1)
    assert np.count_nonzero(large) < np.count_nonzero(small)


@pytest.mark.parametrize('distance, expected', [(0.5, 0.4), (2.0, 1.2), (4.0, 5.0)])
def test_custom_speed_curve_reaches_published_command(node_factory, distance, expected):
    node = node_factory({
        'scan_range_offset': 0, 'front_distance_index': -1,
        'preprocess_conv_size': 1, 'bubble_radius': 0.0,
        'max_distance': 10.0, 'max_speed': 5.0, 'braking_stop_margin': 0.0,
        'max_deceleration': 10.0,
        'speed_near_distance': 1.0, 'speed_far_distance': 3.0,
        'speed_near': 0.4, 'speed_distance_gain': 0.5,
        'speed_distance_offset': 0.2,
    })
    node.lidar_callback(scan([distance] * 20))
    assert node.publisher.publish.call_args.args[0].drive.speed == pytest.approx(expected)


def test_front_beam_and_braking_settings_are_used(node_factory):
    node = node_factory({
        'scan_range_offset': 0, 'front_distance_index': 2,
        'preprocess_conv_size': 1, 'bubble_radius': 0.0,
        'max_speed': 10.0, 'braking_stop_margin': 0.5,
        'max_deceleration': 0.25,
    })
    node.lidar_callback(scan([3.0, 3.0, 1.0, 3.0]))
    assert node.publisher.publish.call_args.args[0].drive.speed == pytest.approx(0.5)


def test_custom_command_timestep_limits_deceleration(node_factory, monkeypatch):
    node = node_factory({'max_command_dt': 0.02, 'max_deceleration': 2.0})
    node.last_commanded_speed = 3.0
    node.last_command_time = 1.0
    monkeypatch.setattr('controller.follow_the_gap.time.monotonic', lambda: 2.0)
    node.lidar_callback(scan([1.0] * 1080))
    assert node.publisher.publish.call_args.args[0].drive.speed == pytest.approx(2.96)


def test_vehicle_calibration_determines_steering_limits(node_factory):
    node = node_factory({
        'steering_angle_to_servo_gain': -1.0,
        'steering_angle_to_servo_offset': 0.5,
        'servo_min': 0.1, 'servo_max': 0.9, 'servo_limit_margin': 0.05,
    })
    assert node.min_steering_angle == pytest.approx(-0.35)
    assert node.max_steering_angle == pytest.approx(0.35)


def test_shutdown_uses_configured_repetitions_and_interval(node_factory, monkeypatch):
    node = node_factory({'stop_publish_count': 2, 'stop_publish_interval': 0.01})
    sleep = Mock()
    monkeypatch.setattr('controller.follow_the_gap.time.sleep', sleep)
    node.last_steering_angle = 0.2
    node.stop_vehicle()
    assert node.publisher.publish.call_count == 2
    assert sleep.call_count == 2
    sleep.assert_called_with(0.01)
    command = node.publisher.publish.call_args.args[0].drive
    assert command.speed == 0.0
    assert command.steering_angle == 0.2


def test_legacy_yaml_matches_defaults_and_tuning_is_used(node_factory):
    node_factory()  # Install ROS I/O substitutes for the historical constructor.
    params = yaml.safe_load((ROOT / 'controller/config/legacy_ftg.yaml').read_text())['/**']['ros__parameters']['legacy_ftg']
    assert params.keys() == FTG_Controller.TUNING_DEFAULTS.keys()
    for name, value in params.items():
        default = FTG_Controller.TUNING_DEFAULTS[name]
        assert value == (pytest.approx(default) if isinstance(default, float) else default)
    node = legacy(**params)
    assert node._get_radius() == 1.0
    assert node._get_steer_angle(1, 1) == 0.4
    tuned = legacy(radius_max=2.0, radius_track_width_ratio=0.75,
                   radius_velocity_gain=3.0, max_steering_angle=0.2,
                   speed_scale=0.5, ultrastraights_speed_ratio=0.8)
    tuned.set_vel(5.0)
    assert tuned._get_radius() == 2.0
    assert tuned._get_steer_angle(1, 1) == 0.2
    assert tuned.ULTRASTRAIGHTS_SPEED == 2.0
    tuned.mapping = True
    tuned.mapping_speed = 0.7
    assert tuned.process_lidar(np.full(1080, 6.0))[0] == 0.7


def test_legacy_zero_crop_and_custom_filter_are_supported(node_factory):
    node_factory()
    node = legacy(preprocess_conv_size=1, lidar_fov_rad=math.pi)
    node.range_offset = 0
    assert len(node._preprocess_lidar([1.0] * 20)) == 20
    assert node.radians_per_elem == pytest.approx(math.pi / 20)


def test_manager_local_core_overrides_avoid_remote_parameter_lookup(monkeypatch):
    import controller.controller_manager as manager

    values = dict(debug=False, safety_radius=4, max_lidar_dist=10.0,
                  max_speed=5.0, range_offset=180, track_width=2.0,
                  disparity_thresh=0.8)
    stub = SimpleNamespace(
        mapping=False, get_logger=Mock(),
        get_parameters_by_prefix=lambda prefix: {
            name: Parameter(name, value=value) for name, value in values.items()},
    )
    remote = Mock(side_effect=AssertionError('Unexpected remote lookup'))
    constructor = Mock()
    monkeypatch.setattr(manager, 'get_remote_parameter', remote)
    monkeypatch.setattr(manager, 'FTG_Controller', constructor)
    manager.Controller.init_ftg_controller(stub)
    assert constructor.call_args.kwargs['max_speed'] == 5.0
    assert constructor.call_args.kwargs['tuning'] == {'disparity_thresh': 0.8}
    remote.assert_not_called()


def test_manager_retains_remote_core_parameter_fallback(monkeypatch):
    import controller.controller_manager as manager

    values = dict(debug=False, safety_radius=4, max_lidar_dist=10.0,
                  max_speed=5.0, range_offset=180, track_width=2.0)
    stub = SimpleNamespace(mapping=False, get_logger=Mock(),
                           get_parameters_by_prefix=lambda prefix: {})
    remote = Mock(side_effect=lambda node, source, name: values[name])
    constructor = Mock()
    monkeypatch.setattr(manager, 'get_remote_parameter', remote)
    monkeypatch.setattr(manager, 'FTG_Controller', constructor)
    manager.Controller.init_ftg_controller(stub)
    assert remote.call_count == 6
    assert constructor.call_args.kwargs['tuning'] == {}
