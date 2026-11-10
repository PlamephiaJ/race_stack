# Copyright (c) 2024 ForzaETH
# SPDX-License-Identifier: MIT
# Adapted from ForzaETH/particle_filter, commit 85ecc1865ac687ad8cf31afadbcb826acb23262c.
# The upstream license and porting notes are in third_party/synpf/.
"""SynPF motion, boxed-beam selection and sensor-table models, independent of ROS."""

import numpy as np

from state_estimation.particle_filter_utils import wrap_angle


def boxed_beam_indices(angles, desired=21, aspect=3.5):
    """Sample equal distances along the rays' intersections with a corridor box.

    Uses the upstream TUM/SynPF box geometry with deterministic nearest-index
    sampling, including endpoints. Works for asymmetric FOVs and even scan sizes.
    """
    angles = np.asarray(angles)
    if desired < 3 or desired % 2 != 1 or aspect <= 0:
        raise ValueError('des_lidar_beams must be odd and >= 3; lidar_aspect_ratio must be > 0')
    if len(angles) < 2:
        return np.arange(len(angles))
    cosine, sine = np.cos(angles), np.sin(angles)
    radius = np.minimum(aspect / np.maximum(abs(cosine), 1e-12),
                        1.0 / np.maximum(abs(sine), 1e-12))
    intersections = np.column_stack((radius * cosine, radius * sine))
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(intersections, axis=0), axis=1))]
    targets = np.linspace(0.0, distance[-1], min(desired, len(angles)))
    right = np.searchsorted(distance, targets).clip(0, len(angles) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(targets - distance[left] <= distance[right] - targets, left, right)
    return np.unique(indices)


def sensor_model_table(max_range_px, resolution, sigma_hit, lambda_short,
                       z_hit, z_short, z_max, z_rand):
    """Upstream normalized Gaussian/exponential mixture, vectorized at startup.

    sigma_hit is expressed in metres, unlike the legacy PF's pixel-space sigma.
    lambda_short follows the upstream conversion and discrete normalization.
    """
    if max_range_px < 1 or resolution <= 0 or sigma_hit <= 0 or lambda_short <= 0:
        raise ValueError('Sensor range, resolution, sigma_hit and lambda_short must be positive')
    mixture = np.array([z_hit, z_short, z_max, z_rand], dtype=float)
    if np.any(mixture < 0) or not np.isclose(mixture.sum(), 1.0):
        raise ValueError('SynPF sensor mixture weights must be nonnegative and sum to one')
    r = np.arange(max_range_px + 1, dtype=float)[:, None]
    d = np.arange(max_range_px + 1, dtype=float)[None, :]
    sigma_px = sigma_hit / resolution
    gaussian = np.exp(-0.5 * ((r - d) / sigma_px) ** 2)
    gaussian /= gaussian.sum(axis=0, keepdims=True)
    short = np.exp(-(lambda_short / resolution) * r) * (r <= d)
    short /= short.sum(axis=0, keepdims=True)
    table = z_hit * gaussian + z_short * short
    table += z_max * (r == max_range_px)
    table += z_rand * (r < max_range_px) / max_range_px
    table /= table.sum(axis=0, keepdims=True)
    return np.ascontiguousarray(table, dtype=np.float64)


def move_local(particles, action):
    """Apply a body-frame SE(2) displacement in place."""
    c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
    particles[:, 0] += c * action[0] - s * action[1]
    particles[:, 1] += s * action[0] + c * action[1]
    particles[:, 2] = wrap_angle(particles[:, 2] + action[2])


def differential_motion(particles, action, alpha, lam_thresh=0.1,
                        model='tum', rng=np.random):
    """TUM/AMCL rot1-trans-rot2 model, including SynPF's translation bias.

    Keep signed rotation for the mean, and use rotation magnitudes only for
    dispersion. Unlike the ROS1 implementation, do not discard sub-centimetre
    displacement or yaw when stationary/turning in place.
    """
    dx, dy, yaw = action
    trans = np.hypot(dx, dy)
    if trans < 0.01:
        move_local(particles, action)
        return
    reverse = dx < 0.0
    reverse_offset = np.pi if reverse else 0.0
    rot1 = wrap_angle(np.arctan2(dy, dx) - reverse_offset)
    rot2 = wrap_angle(yaw - rot1)
    noise1 = min(abs(rot1), abs(wrap_angle(rot1 - np.pi)))
    noise2 = min(abs(rot2), abs(wrap_angle(rot2 - np.pi)))
    a1, a2, a3, a4 = alpha
    translation_noise = a2 / max(trans, lam_thresh) if model == 'tum' else a2 * trans
    spread = 1.1 if reverse else 1.0
    scale1 = (a1 * noise1 + translation_noise) * spread
    scale2 = (a1 * noise2 + translation_noise) * spread
    scale_trans = (a3 * trans + a4 * (noise1 + noise2)) * spread
    count = len(particles)
    sampled1 = rot1 + rng.normal(scale=scale1, size=count)
    sampled_trans = trans + rng.normal(loc=scale_trans / 2, scale=scale_trans, size=count)
    sampled2 = rot2 + rng.normal(scale=scale2, size=count)
    heading = particles[:, 2] + sampled1 + reverse_offset
    particles[:, 0] += sampled_trans * np.cos(heading)
    particles[:, 1] += sampled_trans * np.sin(heading)
    particles[:, 2] = wrap_angle(particles[:, 2] + sampled1 + sampled2)


def arc_motion(particles, action, params, rng=np.random):
    """Experimental upstream arc model with a stable straight-line limit."""
    dx, dy, yaw = action
    sx = max(params['x_min'], abs(dx) * params['x'])
    sy = abs(dy) * params['y'] + max(0.0, abs(dx) - params['xy_min_x']) * params['xy']
    sy = np.clip(sy, params['y_min'], params['y_max'])
    st = max(params['theta_min'], abs(yaw) * params['theta'])
    count = len(particles)
    dx = rng.normal(loc=dx, scale=sx, size=count)
    dy = rng.normal(loc=dy, scale=sy, size=count)
    yaw = rng.normal(loc=yaw, scale=st, size=count)
    # Equivalent arc integration without dividing by a possibly zero yaw.
    x_arc = dx * np.sinc(yaw / np.pi)
    y_arc = dx * 0.5 * yaw * np.sinc(yaw / (2 * np.pi)) ** 2
    c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
    final_heading = particles[:, 2] + yaw
    particles[:, 0] += c * x_arc - s * y_arc - dy * np.sin(final_heading)
    particles[:, 1] += s * x_arc + c * y_arc + dy * np.cos(final_heading)
    particles[:, 2] = wrap_angle(final_heading)


def base_particles_to_laser(particles, mounting):
    result = particles.copy()
    c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
    result[:, 0] += c * mounting[0, 3] - s * mounting[1, 3]
    result[:, 1] += s * mounting[0, 3] + c * mounting[1, 3]
    result[:, 2] = wrap_angle(particles[:, 2] + np.arctan2(mounting[1, 0], mounting[0, 0]))
    return result
