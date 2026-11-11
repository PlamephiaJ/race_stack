"""SynPF numerical contracts, including corrections made in the ROS 2 port."""
from types import SimpleNamespace

import numpy as np
import pytest
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import PoseWithCovarianceStamped

from state_estimation import synpf_models as Models
from state_estimation.synpf import SynPF
from state_estimation.particle_filter import ParticleFiler
from state_estimation import particle_filter_utils as Utils
from pathlib import Path
import runpy

_transport = runpy.run_path(str(Path(__file__).with_name("test_pf_scan_updates.py")))
rig, scan = _transport["rig"], _transport["scan"]


@pytest.mark.parametrize('beams', [1080, 1081, 721])
def test_boxed_sampling_is_bounded_unique_and_uses_endpoints(beams):
    angles = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, beams)
    indices = Models.boxed_beam_indices(angles)
    assert len(indices) == 21
    assert indices[0] == 0 and indices[-1] == beams - 1
    assert np.all(np.diff(indices) > 0)
    assert np.all(abs(indices + indices[::-1] - (beams - 1)) <= 1)
    # The long box puts denser angular samples near the car's forward direction.
    assert np.diff(angles[indices])[10] < np.diff(angles[indices])[0]


@pytest.mark.parametrize('count', [0, 1, 5])
def test_small_scans_do_not_produce_out_of_bounds_indices(count):
    result = Models.boxed_beam_indices(np.linspace(-1.0, 1.0, count))
    assert len(np.unique(result)) == len(result)
    assert np.all(result < count)


def reference_sensor_table(size, resolution, sigma, lam):
    # Direct scalar form from the upstream Gaussian/exponential mixture.
    result = np.zeros((size + 1, size + 1))
    for d in range(size + 1):
        gaussian = np.array([np.exp(-0.5 * ((r - d) / (sigma / resolution)) ** 2)
                             for r in range(size + 1)])
        short = np.array([np.exp(-(lam / resolution) * r) if r <= d else 0.0
                          for r in range(size + 1)])
        for r in range(size + 1):
            result[r, d] = (0.85 * gaussian[r] / gaussian.sum()
                            + 0.1 * short[r] / short.sum()
                            + 0.025 * (r == size)
                            + 0.025 * (r < size) / size)
        result[:, d] /= result[:, d].sum()
    return result


@pytest.mark.parametrize('resolution', [0.05, 0.1])
def test_sensor_table_matches_upstream_normalization_and_metre_sigma(resolution):
    result = Models.sensor_model_table(40, resolution, 0.1, 0.25, 0.85, 0.1, 0.025, 0.025)
    assert np.allclose(result, reference_sensor_table(40, resolution, 0.1, 0.25))
    assert np.allclose(result.sum(axis=0), 1.0)
    assert np.isfinite(result).all() and (result >= 0).all()
    assert result.flags.c_contiguous and result.dtype == np.float64


@pytest.mark.parametrize('action', [[1.0, -0.2, -0.3], [1.0, 0.2, 0.3], [-1.0, 0.0, -0.1]])
@pytest.mark.parametrize('model', ['tum', 'amcl'])
def test_signed_turn_and_reverse_motion_mean(action, model):
    particles = np.zeros((8, 3))
    Models.differential_motion(particles, action, [0.0] * 4, model=model)
    assert np.allclose(particles, action)


@pytest.mark.parametrize('action', [[0.0, 0.0, 0.0], [0.0, 0.0, -0.2], [0.005, 0.0, 0.0]])
def test_small_motion_is_integrated_without_artificial_stationary_drift(action):
    particles = np.zeros((32, 3))
    Models.differential_motion(particles, action, [0.5, 0.015, 1.0, 0.1])
    assert np.allclose(particles, action)


def test_translation_bias_can_be_removed_without_changing_odom_motion():
    class MeanOnlyRng:
        @staticmethod
        def normal(loc=0.0, scale=1.0, size=None):
            return np.full(size, loc)

    odom_centered = np.zeros((4, 3))
    upstream_centered = np.zeros((4, 3))
    for particles, bias in ((odom_centered, 0.0),
                            (upstream_centered, 0.5)):
        Models.differential_motion(
            particles, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
            rng=MeanOnlyRng(), translation_bias_sigma=bias)

    assert np.allclose(odom_centered[:, 0], 1.0)
    assert np.allclose(upstream_centered[:, 0], 1.5)


def test_arc_model_has_finite_zero_yaw_limit():
    params = dict(x=0.0, y=0.0, theta=0.0, xy=0.0, x_min=0.0,
                  y_min=0.0, y_max=0.0, theta_min=0.0, xy_min_x=0.0)
    particles = np.zeros((4, 3))
    Models.arc_motion(particles, [1.0, 0.1, 0.0], params)
    assert np.allclose(particles, [1.0, 0.1, 0.0])


def test_base_to_laser_transform_includes_mounting_yaw_and_lateral_offset():
    mounting = np.array([[0., -1., 0., 0.27], [1., 0., 0., 0.1],
                         [0., 0., 1., 0.11], [0., 0., 0., 1.]])
    base = np.array([[2.0, 3.0, np.pi / 2]])
    result = Models.base_particles_to_laser(base, mounting)
    assert np.allclose(result[0, :2], [1.9, 3.27])
    assert np.isclose(abs(result[0, 2]), np.pi)
    assert np.allclose(base, [2.0, 3.0, np.pi / 2])


def test_map_penalty_rejects_outside_cells_without_clipping_onto_free_boundary():
    info = MapMetaData()
    info.resolution = 1.0
    info.origin.orientation.w = 1.0
    obj = SimpleNamespace(map_info=info, out_of_map_weight=0.01,
                          permissible_region=np.array([[True, False], [True, True]]))
    particles = np.array([[0.2, 0.2, 0.0], [1.2, 0.2, 0.0],
                          [-0.2, 0.2, 0.0], [2.2, 1.2, 0.0]])
    result = SynPF.weight_proposals(obj, particles, np.ones(4))
    assert np.allclose(result, [1.0, 0.01, 0.01, 0.01])


def test_synpf_motion_and_initial_pose_remain_at_base_not_laser():
    obj = rig()
    obj.particles_at_laser = False
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.pose.pose.position.x = 5.0
    msg.pose.pose.position.y = 3.0
    msg.pose.pose.orientation = Utils.angle_to_quaternion(np.pi / 2)
    ParticleFiler.clicked_pose(obj, msg)
    ParticleFiler.lidarCB(obj, scan(1))
    ParticleFiler.process_scan(obj)
    assert np.allclose([obj.resets[0].position.x, obj.resets[0].position.y], [5.0, 3.0])
    assert np.allclose(obj.last_pose, [0.0, 0.0, 0.0])
    ParticleFiler.lidarCB(obj, scan(2))
    ParticleFiler.process_scan(obj)
    assert np.allclose(obj.updates[1][:2], [0.0, 0.0])  # no laser lever-arm translation
    assert np.isclose(abs(obj.updates[1][2]), np.pi)
