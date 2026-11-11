"""Frenet output must not acquire a new odometry stamp from old map TF."""

from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

from frenet_odom_republisher.frenet_odom_republisher_node import FrenetOdomRepublisher


def test_stale_map_pose_is_not_republished_with_fresh_odom_stamp():
    tf = TransformStamped()
    tf.header.stamp.sec = 10
    tf.transform.rotation.w = 1.0
    odom = Odometry()
    odom.header.stamp.sec = 11
    warnings = []
    published = []
    node = SimpleNamespace(
        converter=SimpleNamespace(
            get_frenet=lambda *args: np.array([[1.0], [2.0]]),
            get_frenet_velocities=lambda *args: ([0.0], [0.0]),
            get_closest_index=lambda *args: [0]),
        base_frame='car_state/base_link', max_tf_lag_sec=0.3,
        tf_buffer=SimpleNamespace(lookup_transform=lambda *args: tf),
        get_logger=lambda: SimpleNamespace(
            warn=lambda message, **kwargs: warnings.append(message)),
        frenet_odom_publisher_=SimpleNamespace(publish=published.append))

    FrenetOdomRepublisher.odom_callback(node, odom)

    assert not published
    assert 'skipping stale Frenet pose' in warnings[0]
