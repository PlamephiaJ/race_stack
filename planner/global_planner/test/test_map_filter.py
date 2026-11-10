"""Check map filtering and preservation of the map's coordinate geometry."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from nav_msgs.msg import OccupancyGrid

from global_planner.map_filter import filter_occupancy_grid
from global_planner.map_filter_node import MapFilter


def test_filter_treats_unknown_and_threshold_as_occupied_without_flipping():
    image = filter_occupancy_grid([0, 9, 10, -1, 100, 0], 3, 2, 10, 1)
    np.testing.assert_array_equal(image, [[255, 255, 0], [0, 0, 255]])


def test_filter_removes_isolated_free_pixels_and_keeps_wide_corridor():
    grid = np.full((21, 21), 100, dtype=np.int8)
    grid[4:17, 4:17] = 0
    grid[1, 1] = 0
    image = filter_occupancy_grid(grid.ravel(), 21, 21, 10, 3)
    assert image[1, 1] == 0
    assert np.all(image[4:17, 4:17] == 255)
    assert np.all(image[0] == 0)


def test_live_filter_preserves_header_origin_resolution_and_row_order():
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.header.stamp.sec = 123
    msg.info.width = 3
    msg.info.height = 2
    msg.info.resolution = 0.05
    msg.info.origin.position.x = -12.5
    msg.info.origin.position.y = -7.6
    msg.info.origin.orientation.w = 1.0
    msg.data = [0, 9, 10, -1, 100, 0]
    parameters = {'occupancy_grid_threshold': 10, 'filter_kernel_size': 1}
    node = SimpleNamespace(
        publisher=Mock(),
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )
    MapFilter.map_callback(node, msg)
    result = node.publisher.publish.call_args.args[0]
    assert result.header == msg.header
    assert result.info == msg.info
    assert list(result.data) == [0, 0, 100, 100, 100, 0]
    assert list(msg.data) == [0, 9, 10, -1, 100, 0]
