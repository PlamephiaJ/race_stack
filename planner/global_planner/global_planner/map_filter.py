"""Shared occupancy-grid filtering for mapping and live visualisation."""

import cv2
import numpy as np


def filter_occupancy_grid(data, width, height, occupancy_threshold, kernel_size):
    """Return a white-free/black-occupied image in OccupancyGrid row order."""
    original = np.asarray(data, dtype=np.int8).reshape(height, width)
    original = np.where(original == -1, 100, original)
    binary = np.where(original < occupancy_threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
