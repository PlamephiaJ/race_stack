"""Keep the path and car pose on one recent TF snapshot."""

import math
import time
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from f110_msgs.msg import Wpnt, WpntArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf_transformations import quaternion_from_euler

from local_planner.local_planner import LocalPlanner


def transform(parent, child, x, y, yaw, stamp):
    msg = TransformStamped()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.header.stamp = Time(sec=int(stamp), nanosec=int((stamp % 1) * 1e9))
    msg.transform.translation.x = x
    msg.transform.translation.y = y
    q = quaternion_from_euler(0.0, 0.0, yaw)
    msg.transform.rotation.x = q[0]
    msg.transform.rotation.y = q[1]
    msg.transform.rotation.z = q[2]
    msg.transform.rotation.w = q[3]
    return msg


def planner_with_transforms(map_odom_stamp, odom_base_stamp):
    map_odom = transform('map', 'odom', 1.0, 2.0, math.pi / 2, map_odom_stamp)
    odom_base = transform('odom', 'car_state/base_link', 3.0, 4.0, 0.0,
                          odom_base_stamp)
    lookups = {('map', 'odom'): map_odom,
               ('odom', 'car_state/base_link'): odom_base}
    warnings = []
    return SimpleNamespace(
        tf_buffer=SimpleNamespace(
            lookup_transform=lambda target, source, time: lookups[(target, source)]),
        get_parameter=lambda name: SimpleNamespace(value='car_state/base_link'),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=10_050_000_000)),
        get_logger=lambda: SimpleNamespace(
            warn=lambda msg, **kwargs: warnings.append(msg),
            warning=lambda msg, **kwargs: warnings.append(msg),
            info=lambda msg, **kwargs: warnings.append(msg),
            error=lambda msg, **kwargs: warnings.append(msg)),
        max_tf_lag_sec=0.3, map_to_odom_matrix=None, current_position=None,
        max_tf_jump_m=1.0, max_tf_jump_rad=math.radians(25.0),
        tf_jump_arm_sec=3.0, first_map_odom_seen_at=None,
        localization_recovery_hold_until=0.0,
        localization_recovery_slow_until=0.0,
        initialpose_target=None, initialpose_received_ns=None,
        initialpose_received_at=None,
        tf_jump_hold_sec=0.3, tf_jump_slow_sec=2.0,
        tf_jump_recovery_speed_mps=0.3,
        last_map_odom_matrix=None, lookups=lookups, warnings=warnings)


def test_uses_latest_odom_with_one_map_correction_for_pose_and_path():
    planner = planner_with_transforms(10.0, 10.1)

    assert LocalPlanner.update_pose_from_tf(planner)
    assert np.allclose(planner.current_position, [-3.0, 5.0, math.pi / 2])
    waypoint_in_map = np.array([-3.0, 5.0, 0.0, 1.0])
    assert np.allclose(planner.map_to_odom_matrix @ waypoint_in_map,
                       [3.0, 4.0, 0.0, 1.0])


def test_stale_map_correction_skips_path_update():
    planner = planner_with_transforms(10.0, 10.5)

    assert not LocalPlanner.update_pose_from_tf(planner)
    assert planner.current_position is None
    assert planner.map_to_odom_matrix is None
    assert 'skipping stale path' in planner.warnings[0]


def test_large_localization_jump_holds_then_rebases_after_startup():
    planner = planner_with_transforms(10.0, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)
    planner.first_map_odom_seen_at = time.monotonic() - 4.0

    planner.lookups[('map', 'odom')] = transform(
        'map', 'odom', 1.0, 2.0, math.pi, 10.1)
    assert not LocalPlanner.update_pose_from_tf(planner)
    assert 'correction jumped' in planner.warnings[-1]
    assert planner.localization_recovery_hold_until > time.monotonic()
    assert planner.map_to_odom_matrix is None
    planner.localization_recovery_hold_until = time.monotonic() - 1.0
    assert LocalPlanner.update_pose_from_tf(planner)
    assert planner.localization_recovery_slow_until > time.monotonic()


def test_initial_localization_jump_does_not_enter_recovery():
    planner = planner_with_transforms(10.0, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)

    planner.lookups[('map', 'odom')] = transform(
        'map', 'odom', 1.0, 2.0, math.pi, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)
    assert planner.localization_recovery_hold_until == 0.0


def test_explicit_initialpose_waits_for_matching_new_tf_and_resets_baseline():
    planner = planner_with_transforms(10.0, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)
    planner.first_map_odom_seen_at = time.monotonic() - 4.0
    initialpose = PoseWithCovarianceStamped()
    initialpose.header.frame_id = 'map'
    initialpose.pose.pose.position.x = -2.0
    initialpose.pose.pose.position.y = -2.0
    q = quaternion_from_euler(0.0, 0.0, math.pi)
    (initialpose.pose.pose.orientation.x,
     initialpose.pose.pose.orientation.y,
     initialpose.pose.pose.orientation.z,
     initialpose.pose.pose.orientation.w) = q

    LocalPlanner.initialpose_cb(planner, initialpose)
    assert not LocalPlanner.update_pose_from_tf(planner)  # pre-reset TF
    planner.lookups[('map', 'odom')] = transform(
        'map', 'odom', 1.0, 2.0, math.pi / 2, 10.2)
    assert not LocalPlanner.update_pose_from_tf(planner)  # fresh but wrong TF
    planner.lookups[('map', 'odom')] = transform(
        'map', 'odom', 1.0, 2.0, math.pi, 10.3)
    assert LocalPlanner.update_pose_from_tf(planner)
    assert planner.initialpose_target is None
    assert planner.localization_recovery_slow_until == 0.0
    assert planner.localization_recovery_hold_until == 0.0
    assert np.allclose(planner.current_position, [-2.0, -2.0, math.pi])


def test_explicit_initialpose_wait_publishes_empty_path():
    published = []
    planner = SimpleNamespace(
        localization_recovery_hold_until=0.0,
        initialpose_target=(0.0, 0.0, 0.0),
        update_pose_from_tf=lambda: False,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=12))),
        loc_wpnt_pub=SimpleNamespace(publish=published.append),
    )
    planner.publish_stop_path = lambda: LocalPlanner.publish_stop_path(planner)

    LocalPlanner.loop(planner)

    assert len(published) == 1
    assert not published[0].wpnts


def test_invalid_initialpose_does_not_discard_current_path():
    planner = planner_with_transforms(10.0, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)
    previous_path_transform = planner.map_to_odom_matrix
    initialpose = PoseWithCovarianceStamped()
    initialpose.header.frame_id = 'map'
    initialpose.pose.pose.orientation.w = 0.0

    LocalPlanner.initialpose_cb(planner, initialpose)

    assert planner.initialpose_target is None
    assert planner.map_to_odom_matrix is previous_path_transform


def test_repeated_jump_extends_slow_recovery_without_another_hold():
    planner = planner_with_transforms(10.0, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)
    planner.first_map_odom_seen_at = time.monotonic() - 4.0
    planner.lookups[('map', 'odom')] = transform(
        'map', 'odom', 1.0, 2.0, math.pi, 10.1)
    assert not LocalPlanner.update_pose_from_tf(planner)

    planner.localization_recovery_hold_until = time.monotonic() - 1.0
    previous_slow_until = planner.localization_recovery_slow_until
    planner.lookups[('map', 'odom')] = transform(
        'map', 'odom', 1.0, 2.0, 0.0, 10.1)
    assert LocalPlanner.update_pose_from_tf(planner)
    assert planner.localization_recovery_hold_until < time.monotonic()
    assert planner.localization_recovery_slow_until >= previous_slow_until


def test_recovery_hold_publishes_empty_path_temporarily():
    published = []
    planner = SimpleNamespace(
        localization_recovery_hold_until=time.monotonic() + 1.0,
        initialpose_target=None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=12))),
        loc_wpnt_pub=SimpleNamespace(publish=published.append),
        publish_stop_path=lambda: None)
    planner.publish_stop_path = lambda: LocalPlanner.publish_stop_path(planner)

    LocalPlanner.loop(planner)

    assert len(published) == 1
    assert published[0].header.frame_id == 'odom'
    assert not published[0].wpnts


def test_recovery_caps_published_path_speed_without_changing_source_waypoint():
    published = []
    source = Wpnt(x_m=1.0, y_m=2.0, vx_mps=2.62)
    planner = SimpleNamespace(
        map_to_odom_matrix=np.eye(4),
        localization_recovery_slow_until=time.monotonic() + 1.0,
        tf_jump_recovery_speed_mps=0.3,
        max_speed=3.0,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=12))),
        loc_wpnt_pub=SimpleNamespace(publish=published.append),
        vis_loc_wpnt_pub=SimpleNamespace(publish=lambda msg: None),
    )

    LocalPlanner._pub_local_wpnts(planner, WpntArray(wpnts=[source]))

    assert published[0].wpnts[0].vx_mps == 0.3
    assert source.vx_mps == 2.62
