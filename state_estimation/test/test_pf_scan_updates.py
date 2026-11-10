"""Check the scan/TF contract at the boundary with the race stack."""
from types import SimpleNamespace
from collections import deque

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer

from state_estimation.particle_filter import ParticleFiler
from state_estimation import particle_filter_utils as Utils


def transform(parent, child, second, x=0.0, yaw=0.0):
    result = TransformStamped()
    result.header.frame_id = parent
    result.child_frame_id = child
    result.header.stamp.sec = second
    result.transform.translation.x = x
    result.transform.rotation = Utils.angle_to_quaternion(yaw)
    return result


def rig():
    buffer = Buffer()
    mount = transform('base_link', 'laser', 0, x=0.27)
    buffer.set_transform_static(mount, 'test')
    # A later pose must not be used for the first scan.
    buffer.set_transform(transform('odom', 'base_link', 1), 'test')
    buffer.set_transform(transform('odom', 'base_link', 2, yaw=np.pi), 'test')
    obj = SimpleNamespace(
        scan_queue=deque(maxlen=10), pending_initial_pose=None, tf_buffer=buffer,
        odom_frame='odom', base_frame='base_link', last_stamp=None, last_pose=None,
        particles_at_laser=True, base_to_laser_matrix=None,
        ANGLE_STEP=1, MAX_RANGE_METERS=10.0, downsampled_angles=None,
        updates=[], resets=[], get_logger=lambda: SimpleNamespace(
            warning=lambda *a, **k: None))
    obj.scan_indices = lambda msg: np.arange(0, len(msg.ranges), obj.ANGLE_STEP)
    obj.update = lambda: obj.updates.append(obj.odometry_data.copy())
    obj.initialize_particles_pose = obj.resets.append
    return obj


def scan(second):
    msg = LaserScan()
    msg.header.frame_id = 'laser'
    msg.header.stamp.sec = second
    msg.angle_min = -1.0
    msg.angle_increment = 1.0
    msg.ranges = [float('nan'), float('inf'), -1.0, 12.0]
    return msg


def test_scan_uses_matching_tf_once_and_includes_laser_lever_arm():
    obj = rig()
    ParticleFiler.lidarCB(obj, scan(1))
    ParticleFiler.process_scan(obj)
    assert np.allclose(obj.last_pose, [0.27, 0.0, 0.0])
    assert np.allclose(obj.downsampled_ranges, [10.0, 10.0, 0.0, 10.0])
    for _ in range(5):
        ParticleFiler.process_scan(obj)
    assert len(obj.updates) == 1
    ParticleFiler.lidarCB(obj, scan(2))
    ParticleFiler.process_scan(obj)
    assert len(obj.updates) == 2
    assert np.allclose(obj.updates[1][:2], [-0.54, 0.0], atol=1e-8)
    assert np.isclose(abs(obj.updates[1][2]), np.pi)
    ParticleFiler.lidarCB(obj, scan(1))  # late/duplicate scans must not rewind PF
    ParticleFiler.process_scan(obj)
    assert len(obj.updates) == 2


def test_scan_waits_for_tf_without_consuming_or_blocking():
    obj = rig()
    msg = scan(3)
    ParticleFiler.lidarCB(obj, msg)
    ParticleFiler.process_scan(obj)
    assert obj.scan_queue[0] is msg
    assert not obj.updates
    obj.tf_buffer.set_transform(transform('odom', 'base_link', 3, x=2.0), 'test')
    ParticleFiler.process_scan(obj)
    assert len(obj.updates) == 1
    assert not obj.scan_queue
    assert np.isclose(obj.last_pose[0], 2.27)


def test_new_scans_do_not_starve_a_scan_waiting_for_delayed_tf():
    obj = rig()
    ParticleFiler.lidarCB(obj, scan(3))
    ParticleFiler.process_scan(obj)
    ParticleFiler.lidarCB(obj, scan(4))
    obj.tf_buffer.set_transform(transform('odom', 'base_link', 3, x=2.0), 'test')
    ParticleFiler.process_scan(obj)
    assert len(obj.updates) == 1
    assert obj.last_stamp.sec == 3
    assert obj.scan_queue[0].header.stamp.sec == 4


def test_rviz_base_pose_is_converted_to_laser_pose():
    obj = rig()
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.pose.pose.position.x = 5.0
    msg.pose.pose.position.y = 3.0
    msg.pose.pose.orientation = Utils.angle_to_quaternion(np.pi / 2.0)
    ParticleFiler.clicked_pose(obj, msg)
    assert not obj.resets  # wait until the laser mounting TF is available
    ParticleFiler.lidarCB(obj, scan(1))
    ParticleFiler.process_scan(obj)
    assert np.allclose([obj.resets[0].position.x, obj.resets[0].position.y], [5.0, 3.27])
    assert np.isclose(Utils.quaternion_to_angle(obj.resets[0].orientation), np.pi / 2.0)
