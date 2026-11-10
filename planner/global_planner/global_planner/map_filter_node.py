"""Publish a filtered view of the live SLAM map without changing /map."""

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .map_filter import filter_occupancy_grid


class MapFilter(Node):
    """Apply the mapping filter to /map_raw and publish /map_filtered."""

    def __init__(self):
        super().__init__('map_filter')
        self.declare_parameter('occupancy_grid_threshold', 10)
        self.declare_parameter('filter_kernel_size', 3)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(OccupancyGrid, '/map_filtered', qos)
        self.subscription = self.create_subscription(
            OccupancyGrid, '/map_raw', self.map_callback, qos)

    def map_callback(self, msg):
        """Preserve map geometry, timestamps and row order in the filtered view."""
        if not msg.info.width or not msg.info.height:
            return
        filtered = filter_occupancy_grid(
            msg.data, msg.info.width, msg.info.height,
            self.get_parameter('occupancy_grid_threshold').value,
            self.get_parameter('filter_kernel_size').value,
        )
        result = OccupancyGrid()
        result.header = msg.header
        result.info = msg.info
        result.data = np.where(filtered == 255, 0, 100).astype(np.int8).ravel().tolist()
        self.publisher.publish(result)


def main(args=None):
    """Run the live map filter."""
    rclpy.init(args=args)
    node = MapFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
