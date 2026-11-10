# Copyright (c) 2026 Yuhao Chen

import numpy as np

from state_estimation import particle_filter_utils as Utils


def test_angle_difference_crosses_branch_cut_by_shortest_path():
    previous = np.deg2rad(179.0)
    current = np.deg2rad(-179.0)

    difference = Utils.angle_difference(current, previous)

    assert np.isclose(difference, np.deg2rad(2.0))


def test_weighted_pose_mean_is_circular():
    poses = np.array([
        [1.0, 2.0, np.deg2rad(179.0)],
        [3.0, 4.0, np.deg2rad(-179.0)],
    ])

    mean = Utils.weighted_pose_mean(poses, np.array([0.5, 0.5]))

    assert np.allclose(mean[:2], [2.0, 3.0])
    assert np.isclose(abs(mean[2]), np.pi)


def test_weighted_pose_covariance_wraps_yaw_residuals():
    poses = np.array([
        [1.0, 2.0, np.deg2rad(179.0)],
        [1.0, 2.0, np.deg2rad(-179.0)],
    ])
    weights = np.array([0.5, 0.5])
    mean = Utils.weighted_pose_mean(poses, weights)

    covariance = Utils.weighted_pose_covariance(poses, weights, mean)

    assert np.isclose(covariance[2, 2], np.deg2rad(1.0) ** 2)
    assert np.allclose(covariance[:2, :], 0.0)
    assert np.allclose(covariance[:, :2], 0.0)


def test_wrap_angle_handles_arrays():
    angles = np.deg2rad(np.array([181.0, -181.0, 720.0]))

    wrapped = Utils.wrap_angle(angles)

    assert np.allclose(wrapped, np.deg2rad([-179.0, 179.0, 0.0]))


def test_planar_covariance_uses_ros_x_y_yaw_indices():
    planar_covariance = np.arange(9, dtype=float).reshape(3, 3)

    ros_covariance = np.array(
        Utils.planar_covariance_to_ros(planar_covariance)).reshape(6, 6)

    indices = np.array([0, 1, 5])
    assert np.array_equal(
        ros_covariance[np.ix_(indices, indices)], planar_covariance)
    ros_covariance[np.ix_(indices, indices)] = 0.0
    assert np.count_nonzero(ros_covariance) == 0
