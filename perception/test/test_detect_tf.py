"""Exercise scan/TF timing without starting ROS middleware or a vehicle."""

import math
from collections import deque
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

from geometry_msgs.msg import TransformStamped

import numpy as np

import perception.detect as detection

import pytest

from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time

from sensor_msgs.msg import LaserScan

from tf2_ros import Buffer


BASE_NS = 10_000_000_000


def scan_at(stamp_ns):
    """Build a valid scan at a specified acquisition time."""
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    scan.angle_min = 0.0
    scan.angle_max = 0.01
    scan.angle_increment = 0.01
    scan.ranges = [1.0, 1.0]
    return scan


def add_tf(detector, stamp_ns):
    """Deliver a transform sample to the real TF buffer."""
    transform = TransformStamped()
    transform.header.frame_id = 'map'
    transform.child_frame_id = 'laser'
    transform.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    transform.transform.rotation.w = 1.0
    detector.tf_buffer.set_transform(transform, 'test')


@pytest.fixture
def detector(monkeypatch):
    """Construct a detector with a controlled clock and mocked ROS publications."""
    # Use the actual callbacks and TF buffer, replacing only ROS node services
    # and the downstream detection algorithm/publications.
    node = detection.Detect.__new__(detection.Detect)
    node.lock = Lock()
    node.pending_scans = deque(maxlen=10)
    node.car_s = 1.0
    node.tf_wait_timeout = 0.2
    node.tf_buffer = Buffer(Duration(seconds=5))
    node.converter = Mock()
    node.track_length = 10.0
    node.measuring = False
    node.current_stamp = None
    node.tracked_obstacles = ['previous obstacle']
    node.get_logger = Mock(return_value=Mock())
    node.scans2ObsPointCloud = Mock(return_value=['points'])
    node.obsPointClouds2obsArray = Mock(return_value=['obstacle'])
    node.checkObstacles = Mock()
    node.publishObstaclesMessage = Mock()
    node.publishObstaclesMarkers = Mock()
    node.test_clock = Mock(return_value=0.0)
    monkeypatch.setattr(detection, 'time', SimpleNamespace(monotonic=node.test_clock))
    add_tf(node, BASE_NS - 20_000_000)
    add_tf(node, BASE_NS)
    return node


def test_delayed_tf_retries_without_publishing_empty_obstacles(detector):
    """Retry the same scan once its transform arrives, publishing only once."""
    scan = scan_at(BASE_NS + 1_267_000)
    detector.laserCb(scan)

    detector.loop()

    assert len(detector.pending_scans) == 1
    assert detector.tracked_obstacles == ['previous obstacle']
    detector.scans2ObsPointCloud.assert_not_called()
    detector.checkObstacles.assert_not_called()
    detector.publishObstaclesMessage.assert_not_called()
    detector.publishObstaclesMarkers.assert_not_called()
    detector.get_logger().warn.assert_not_called()

    add_tf(detector, BASE_NS + 20_000_000)
    detector.loop()

    assert not detector.pending_scans
    assert detector.scans2ObsPointCloud.call_args.kwargs['scans'] is scan
    assert detector.current_stamp == scan.header.stamp
    detector.publishObstaclesMessage.assert_called_once()
    detector.publishObstaclesMarkers.assert_called_once()

    detector.loop()
    detector.publishObstaclesMessage.assert_called_once()


def test_new_scans_do_not_replace_scan_waiting_for_tf(detector):
    """Retain each scan and its car position while newer messages arrive."""
    first = scan_at(BASE_NS + 1_267_000)
    second = scan_at(BASE_NS + 30_000_000)
    detector.laserCb(first)
    detector.loop()
    detector.car_s = 2.0
    detector.laserCb(second)
    add_tf(detector, BASE_NS + 20_000_000)

    detector.loop()
    detector.scans2ObsPointCloud.assert_called_once()
    assert detector.scans2ObsPointCloud.call_args.kwargs['scans'] is first
    assert detector.scans2ObsPointCloud.call_args.kwargs['car_s'] == 1.0

    detector.loop()
    detector.scans2ObsPointCloud.assert_called_once()
    add_tf(detector, BASE_NS + 40_000_000)
    detector.loop()
    assert detector.scans2ObsPointCloud.call_count == 2
    assert detector.scans2ObsPointCloud.call_args.kwargs['scans'] is second
    assert detector.scans2ObsPointCloud.call_args.kwargs['car_s'] == 2.0


def test_scan_arriving_during_processing_is_retained(detector):
    """Keep callback updates separate from the scan currently being processed."""
    first = scan_at(BASE_NS)
    second = scan_at(BASE_NS + 20_000_000)
    detector.laserCb(first)

    def process_scan(**kwargs):
        detector.laserCb(second)
        assert kwargs['scans'] is first
        return []

    detector.scans2ObsPointCloud.side_effect = process_scan
    detector.loop()

    assert detector.current_stamp == first.header.stamp
    assert len(detector.pending_scans) == 1
    assert detector.pending_scans[0][0] is second


def test_tf_timeout_drops_scan_without_clearing_obstacles(detector):
    """Report a persistent TF failure without publishing a false empty result."""
    detector.laserCb(scan_at(BASE_NS + 1_267_000))
    detector.test_clock.return_value = 0.21

    detector.loop()

    assert not detector.pending_scans
    assert detector.tracked_obstacles == ['previous obstacle']
    detector.publishObstaclesMessage.assert_not_called()
    detector.checkObstacles.assert_not_called()
    detector.get_logger().warn.assert_called_once()
    assert 'map transform' in detector.get_logger().warn.call_args.args[0]


def test_expired_scan_does_not_block_fresh_scan(detector):
    """Skip an expired scan and process the next fresh scan in the same loop."""
    detector.laserCb(scan_at(BASE_NS))
    detector.test_clock.return_value = 0.21
    fresh = scan_at(BASE_NS + 20_000_000)
    detector.laserCb(fresh)
    add_tf(detector, BASE_NS + 20_000_000)

    detector.loop()

    assert not detector.pending_scans
    assert detector.scans2ObsPointCloud.call_args.kwargs['scans'] is fresh
    detector.publishObstaclesMessage.assert_called_once()


def test_pending_queue_is_bounded_and_reports_overflow(detector):
    """Limit retained scans and make persistent queue overflow observable."""
    scans = [scan_at(BASE_NS + index) for index in range(11)]
    for scan in scans:
        detector.laserCb(scan)

    assert len(detector.pending_scans) == 10
    assert detector.pending_scans[0][0] is scans[1]
    assert detector.pending_scans[-1][0] is scans[-1]
    detector.get_logger().warn.assert_called_once()


def test_uninitialized_or_empty_scans_are_ignored(detector):
    """Avoid detection until a scan with a frame and ranges has arrived."""
    detector.loop()
    detector.laserCb(LaserScan())
    scan = scan_at(BASE_NS)
    scan.ranges = []
    detector.laserCb(scan)
    detector.loop()

    assert not detector.pending_scans
    detector.scans2ObsPointCloud.assert_not_called()
    detector.publishObstaclesMessage.assert_not_called()


def test_point_cloud_uses_supplied_scan_ranges(detector):
    """Segment the supplied frame even if shared scan data has changed."""
    scan = scan_at(BASE_NS)
    detector.current_stamp = scan.header.stamp
    # A concurrent callback may have replaced the former shared scan with one
    # containing a different number of ranges.
    detector.scans = LaserScan()
    detector.lambda_angle = math.radians(10)
    detector.sigma = 0.03
    detector.min_obs_size = 1
    detector.laserPointOnTrack = Mock(return_value=True)
    detector.converter.get_frenet.side_effect = (
        lambda x, y: (np.zeros(len(x)), np.zeros(len(y))))
    detector.breakpoints_markers_pub = Mock()
    transform = detector.tf_buffer.lookup_transform('map', 'laser', Time(nanoseconds=BASE_NS))

    clouds = detection.Detect.scans2ObsPointCloud(detector, 1.0, scan, transform)

    assert sum(len(cloud) for cloud in clouds) == len(scan.ranges)


@pytest.mark.parametrize('timeout_override', [None, 0.4])
def test_constructor_leaves_callbacks_to_main_executor(monkeypatch, timeout_override):
    """Use the main executor and accept an automatically declared timeout override."""
    parameters = {
        '/measure': Parameter('/measure', value=False),
        '/from_bag': Parameter('/from_bag', value=False),
    }
    if timeout_override is not None:
        parameters['tf_wait_timeout'] = Parameter('tf_wait_timeout', value=timeout_override)

    def declare(node, name, default, *args, **kwargs):
        assert name not in parameters
        parameters[name] = Parameter(name, value=default)
        return parameters[name]

    monkeypatch.setattr(detection.Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(detection.Node, 'declare_parameter', declare)
    monkeypatch.setattr(detection.Node, 'has_parameter', lambda node, name: name in parameters)
    monkeypatch.setattr(detection.Node, 'get_parameter', lambda node, name: parameters[name])
    for method in ('get_logger', 'create_subscription', 'create_publisher',
                   'add_on_set_parameters_callback', 'create_timer'):
        monkeypatch.setattr(detection.Node, method, Mock())
    listener = Mock()
    monkeypatch.setattr(detection, 'TransformListener', listener)

    node = detection.Detect()

    listener.assert_called_once_with(node.tf_buffer, node, spin_thread=False)
    assert node.tf_wait_timeout == (0.2 if timeout_override is None else timeout_override)


def test_main_uses_two_workers_and_shuts_executor_down(monkeypatch):
    """Give TF independent execution capacity and stop workers before destroying ROS."""
    node = Mock()
    executor = Mock()
    executor.spin.side_effect = KeyboardInterrupt
    ros = SimpleNamespace(
        init=Mock(), shutdown=Mock(),
        executors=SimpleNamespace(MultiThreadedExecutor=Mock(return_value=executor)))
    monkeypatch.setattr(detection, 'Detect', Mock(return_value=node))
    monkeypatch.setattr(detection, 'rclpy', ros)

    detection.main()

    ros.executors.MultiThreadedExecutor.assert_called_once_with(num_threads=2)
    executor.add_node.assert_called_once_with(node)
    executor.shutdown.assert_called_once()
    node.destroy_node.assert_called_once()
    ros.shutdown.assert_called_once()
