# Copyright (c) 2026 Yuhao Chen

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, Transform
import numpy as np
import tf_transformations
from types import SimpleNamespace

from state_estimation import particle_filter_utils as Utils
from state_estimation.particle_filter import ParticleFiler


class _RecordingPublisher:
    def __init__(self):
        self.message = None

    def publish(self, message):
        self.message = message


class _FakeParticleFilter:
    def __init__(self):
        self.last_stamp = Time(sec=123, nanosec=456)
        self.laser_frame = 'laser'
        self.pub_fake_scan = _RecordingPublisher()


def test_fake_scan_converts_numpy_scalars_to_ros_float_fields():
    particle_filter = _FakeParticleFilter()
    angles = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    ranges = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    ParticleFiler.publish_scan(particle_filter, angles, ranges)

    scan = particle_filter.pub_fake_scan.message
    assert isinstance(scan.angle_min, float)
    assert isinstance(scan.angle_max, float)
    assert isinstance(scan.angle_increment, float)
    assert isinstance(scan.range_max, float)
    assert np.allclose(scan.ranges, [1.0, 2.0, 3.0])


def test_boxed_beams_use_uniform_angles_only_for_fake_scan_visualization():
    angles = np.array([-1.2, -1.0, -0.4, 0.0, 0.05, 0.3, 1.1], dtype=np.float32)
    queries = []

    def cast(ins, outs):
        queries.append(ins.copy())
        outs[:] = np.arange(len(outs)) + 1.0

    publisher = _RecordingPublisher()
    publisher.get_subscription_count = lambda: 1
    obj = SimpleNamespace(
        DO_VIZ=True, last_visualization_time=None, VIZ_UPDATE_PERIOD=0.2,
        pose_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        particle_pub=SimpleNamespace(get_subscription_count=lambda: 0),
        pub_fake_scan=publisher, ranges=np.zeros(len(angles)),
        inferred_pose=np.array([2.0, 3.0, 0.5]),
        laser_particles=lambda particles: particles + np.array([0.27, 0.1, 0.3]),
        downsampled_angles=angles.copy(), viz_queries=np.zeros((len(angles), 3), dtype=np.float32),
        viz_ranges=np.zeros(len(angles), dtype=np.float32),
        range_method=SimpleNamespace(calc_range_many=cast),
        laser_frame='laser', last_stamp=Time(sec=123))
    obj.publish_scan = lambda a, r: ParticleFiler.publish_scan(obj, a, r)

    ParticleFiler.visualize(obj)

    scan = publisher.message
    represented_angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    assert np.allclose(queries[0][:, 2] - 0.8, represented_angles, atol=1e-6)
    assert np.allclose(queries[0][:, :2], [2.27, 3.1])
    assert np.array_equal(obj.downsampled_angles, angles)  # MCL keeps boxed beams
    assert scan.header.frame_id == 'laser' and scan.header.stamp.sec == 123


def test_odom_pose_and_static_laser_transform_reconstruct_odom_laser():
    odom_base = Pose()
    odom_base.position.x = 2.0
    odom_base.position.y = -1.0
    odom_base.orientation.z = np.sin(0.4 / 2.0)
    odom_base.orientation.w = np.cos(0.4 / 2.0)

    base_laser = Transform()
    base_laser.translation.x = 0.27
    base_laser.translation.z = 0.11
    base_laser.rotation.w = 1.0

    odom_laser = tf_transformations.concatenate_matrices(
        Utils.pose_to_matrix(odom_base),
        Utils.transform_to_matrix(base_laser))

    expected = tf_transformations.concatenate_matrices(
        tf_transformations.translation_matrix([2.0, -1.0, 0.0]),
        tf_transformations.quaternion_matrix(
            tf_transformations.quaternion_from_euler(0.0, 0.0, 0.4)),
        tf_transformations.translation_matrix([0.27, 0.0, 0.11]))
    assert np.allclose(odom_laser, expected)


def test_map_odom_compensation_preserves_particle_filter_map_laser_pose():
    map_laser = tf_transformations.concatenate_matrices(
        tf_transformations.translation_matrix([5.0, 3.0, 0.0]),
        tf_transformations.quaternion_matrix(
            tf_transformations.quaternion_from_euler(0.0, 0.0, -0.7)))
    odom_base = tf_transformations.concatenate_matrices(
        tf_transformations.translation_matrix([100.0, -80.0, 0.0]),
        tf_transformations.quaternion_matrix(
            tf_transformations.quaternion_from_euler(0.0, 0.0, 2.1)))
    base_laser = Transform()
    base_laser.translation.x = 0.27
    base_laser.rotation.w = 1.0
    odom_laser = tf_transformations.concatenate_matrices(
        odom_base, Utils.transform_to_matrix(base_laser))

    map_odom = Utils.map_to_odom_matrix(
        map_laser, odom_base, base_laser)

    assert np.allclose(
        tf_transformations.concatenate_matrices(map_odom, odom_laser),
        map_laser)
