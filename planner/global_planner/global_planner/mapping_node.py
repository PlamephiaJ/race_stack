#!/usr/bin/env python3
import select
import termios
import tty
import os
import shutil

import yaml
import cv2
import numpy as np


import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from cartographer_ros_msgs.srv import FinishTrajectory, WriteState
from tf_transformations import euler_from_quaternion

from .global_planner_utils import get_data_path
from .map_filter import filter_occupancy_grid



class MappingNode(Node):
    """Mapping node that waits for the mapping process to be complete and saves the map."""

    def __init__(self):
        super().__init__('mapping_node', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.map_name = self.get_parameter('map_name').value
        if not self.has_parameter('loc_algorithm'):
            self.declare_parameter('loc_algorithm', 'synpf')
        self.loc_algorithm = self.get_parameter('loc_algorithm').value
        if self.loc_algorithm not in ('cartographer', 'slam', 'pf', 'synpf'):
            raise ValueError('loc_algorithm must be synpf, pf or cartographer (alias: slam)')
        self.map_dir = get_data_path('maps/' + self.map_name)
        self.rate = self.get_parameter('rate').value
        self.map_occupancy_grid_threshold = self.get_parameter('occupancy_grid_threshold').value
        self.filter_kernel_size = self.get_parameter('filter_kernel_size').value
        
        self.filtered_map = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin = None
        self.map_occupancy_grid = None
        self.pose_valid = False
        self.initial_position = None

        # Pose comes from TF, not from an odometry topic, so whatever SLAM is
        # publishing map->base_link defines the start pose written into the map.
        self.map_frame = 'map'
        self.base_frame = 'base_link'
        
        self._tty = open('/dev/tty', 'r')
        self._old_settings = termios.tcgetattr(self._tty)
        
        if self.loc_algorithm not in ('pf', 'synpf'):
            self.finish_trajectory_client = self.create_client(FinishTrajectory, '/finish_trajectory')
            self.write_state_client = self.create_client(WriteState, '/write_state')
        self.filtered_map_pub = self.create_publisher(
            OccupancyGrid, '/map_filtered', QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            ))
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.create_timer(1 / self.rate, self.loop)

    def get_key(self):
        """Wait for a key press on the terminal and return it."""
        
        try:
            tty.setcbreak(self._tty.fileno())
            rlist, _, _ = select.select([self._tty], [], [], 0.1)
            if rlist:
                return self._tty.read(1)
            return None
        finally:
            termios.tcsetattr(self._tty, termios.TCSADRAIN, self._old_settings)
            
    def latch_initial_pose(self) -> None:
        """
        Latch the start pose from the map -> base_link transform, once it exists.

        Called every loop until it succeeds; the first transform available is the
        one saved as the map's ``initial_pose``.

        Returns:
            None
        """
        if self.pose_valid:
            return
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except tf2_ros.TransformException as e:
            self.get_logger().log(
                f"Waiting for the {self.map_frame}->{self.base_frame} transform: {e}",
                severity=rclpy.logging.LoggingSeverity.WARN,
                once=True,
            )
            return

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.initial_position = (t.x, t.y, theta)
        self.pose_valid = True
        self.get_logger().info(
            f"Initial pose latched from TF: x={t.x:.3f} y={t.y:.3f} theta={theta:.3f}")

    def map_cb(self, msg: OccupancyGrid) -> None:
        """
        Updates the map with the given OccupancyGrid message.

        Args:
            msg (OccupancyGrid): The OccupancyGrid message containing map data.

        Returns:
            None
        """
        self.get_logger().log("Received a map.", severity=rclpy.logging.LoggingSeverity.INFO, once=True)
        self.map_width = msg.info.width  # uint32, [cells]
        self.map_height = msg.info.height  # uint32, [cells]
        self.map_resolution = msg.info.resolution  # float32, [m/cell]
        self.map_origin = msg.info.origin.position
        self.map_occupancy_grid = msg.data  # int8[]
        self.filter_map_occupancy_grid()

        filtered_msg = OccupancyGrid()
        filtered_msg.header = msg.header
        filtered_msg.info = msg.info
        # Keep the original OccupancyGrid row order; only saved PNGs are flipped.
        # The filtered image uses white for free and black for occupied cells.
        filtered_msg.data = np.where(self.filtered_map == 255, 0, 100).astype(np.int8).ravel().tolist()
        self.filtered_map_pub.publish(filtered_msg)
            
    def filter_map_occupancy_grid(self) -> None:
        """
        Filters the occupancy grid map by performing morphological opening and binarization.

        Returns:
            None
        """
        self.filtered_map = filter_occupancy_grid(
            self.map_occupancy_grid, self.map_width, self.map_height,
            self.map_occupancy_grid_threshold, self.filter_kernel_size,
        )
        
    def save_map(self) -> None:
        """Saves the filtered map as a PNG and YAML file in the specified directory."""       
        if self.map_occupancy_grid is None:
            self.get_logger().error("No map received yet. Cannot save the map.")
            return
        self.get_logger().info(f"Saving Map '{self.map_name}'...")

        if self.initial_position is None:
            self.get_logger().warn(
                f"No {self.map_frame}->{self.base_frame} transform was ever seen; "
                "saving the map with initial_pose (0, 0, 0).")
            self.initial_position = (0.0, 0.0, 0.0)

        _check_default_map(self.map_name, self.map_dir, get_data_path('maps/backup'), self.get_logger().warn)
        os.makedirs(self.map_dir)
 
        self.get_logger().info(f'Successfully created the folder {self.map_dir}')
        self.filter_map_occupancy_grid()
        _save_map_to_directory(self.map_dir, self.map_name, self.filtered_map, self.map_resolution, self.map_origin.x, self.map_origin.y, self.initial_position)
        _save_map_to_directory(self.map_dir, 'pf_map', self.filtered_map, self.map_resolution, self.map_origin.x, self.map_origin.y, self.initial_position)

        self.get_logger().info(f'PNG and YAML files saved in {self.map_dir}')
        if self.loc_algorithm in ('pf', 'synpf'):
            # slam_toolbox publishes the same OccupancyGrid. There is no
            # Cartographer trajectory to finish and no pbstream to export.
            return

        # Call ROS service to save PB stream
        pbstream_path = os.path.join(self.map_dir, self.map_name + ".pbstream")
        while not self.finish_trajectory_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /finish_trajectory service...')
        finish_trajectory_request = FinishTrajectory.Request()
        finish_trajectory_request.trajectory_id = 0
        self.finish_trajectory_client.call_async(finish_trajectory_request)
        
        while not self.write_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /write_state service...')
        write_state_request = WriteState.Request()
        write_state_request.filename = pbstream_path
        write_state_request.include_unfinished_submaps = True
        self.write_state_client.call_async(write_state_request)

        self.get_logger().info(f'PNG and YAML file created and saved in the {self.map_dir} folder')
        
    def loop(self):
        """Main Loop"""
        
        self.get_logger().log(
            "Mapping active. View /map_filtered in RViz and press 'y' in this terminal to save the map.",
            severity=rclpy.logging.LoggingSeverity.INFO,
            once=True,
        )
        self.latch_initial_pose()

        key = self.get_key()
        if key and key == 'y':
            self.save_map()
            
            
        

    def destroy_node(self):
        """Clean up the terminal settings before shutting down."""
        
        self._tty.close()
        super().destroy_node()

def _check_default_map(map_name: str, map_dir: str, backup_dir: str, logwarn) -> None:
    if map_name == 'latest':
        if map_dir.exists():
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(map_dir), str(backup_dir))
            logwarn(f"Map directory already exists. Backed up the old map to {backup_dir}")
            

def _save_map_to_directory(
    map_dir: str,
    map_name: str,
    filtered_map: np.ndarray,
    map_resolution: float,
    map_origin_x: float,
    map_origin_y: float,
    initial_position: list
    ) -> None:
    img_path = os.path.join(map_dir, map_name + '.png')
    flipped_map = cv2.flip(filtered_map, 0)
    cv2.imwrite(img_path, flipped_map)

    dict_map = {'image': map_name + '.png',
                'resolution': map_resolution,
                'origin': [map_origin_x, map_origin_y, 0],
                'negate': 0,
                'occupied_thresh': 0.65,
                'free_thresh': 0.196, 
                'initial_pose': [initial_position[0], initial_position[1], initial_position[2]]}

    with open(os.path.join(map_dir, map_name + ".yaml"), 'w') as file:
        _ = yaml.dump(dict_map, file, default_flow_style=False)

def main(args=None):
    """Main function to run the mapping node."""
    
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
