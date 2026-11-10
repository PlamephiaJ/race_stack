"""Regression checks for noise loops and periodic centerline smoothing."""

import cv2
import numpy as np
import pytest

from global_planner.global_planner_utils import extract_centerline, smooth_centerline


def test_default_centerline_ignores_small_closed_noise_loop():
    skeleton = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(skeleton, (120, 120), 70, 255, 1)
    cv2.circle(skeleton, (20, 20), 3, 255, 1)
    centerline = extract_centerline(skeleton, 0., .05)
    assert len(centerline) > 300
    assert np.all(np.linalg.norm(centerline - [120, 120], axis=1) > 65)


def test_expected_length_still_selects_matching_loop():
    skeleton = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(skeleton, (120, 120), 70, 255, 1)
    cv2.circle(skeleton, (20, 20), 10, 255, 1)
    centerline = extract_centerline(skeleton, 2 * np.pi * 10 * .05, .05)
    assert np.all(np.linalg.norm(centerline - [20, 20], axis=1) < 11)


def test_open_skeleton_has_no_centerline():
    skeleton = np.zeros((40, 40), dtype=np.uint8)
    skeleton[20, 5:35] = 255
    with pytest.raises(IOError, match='No closed contours'):
        extract_centerline(skeleton, 0., .05)


@pytest.mark.parametrize('count', [5, 6, 15, 20, 21, 41, 501, 1001, 2001])
def test_smoothing_is_finite_and_independent_of_loop_start(count):
    angles = np.arange(count) * 2 * np.pi / count
    centerline = np.column_stack((np.cos(angles), np.sin(angles)))
    smoothed = smooth_centerline(centerline)
    assert smoothed.shape == centerline.shape
    assert np.isfinite(smoothed).all()
    np.testing.assert_allclose(
        smooth_centerline(np.roll(centerline, 3, axis=0)),
        np.roll(smoothed, 3, axis=0), atol=1e-12)


@pytest.mark.parametrize('count', [0, 1, 4])
def test_too_short_centerline_has_actionable_error(count):
    with pytest.raises(ValueError, match='at least 5 points'):
        smooth_centerline(np.zeros((count, 2)))
