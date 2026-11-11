#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

import time
from visualization_msgs.msg import Marker


class ReactiveFollowGap(Node):
    # Startup defaults also allow running reactive_node without a parameter file.
    PARAMETER_DEFAULTS = {
        'debug': True,
        'bubble_radius': 0.25,
        'max_distance': 3.5,
        'disparity_thresh': 0.15,
        'scan_range_offset': 180,
        'preprocess_conv_size': 3,
        'minimum_gap_angle': 0.12,
        'gap_weight_power': 2.0,
        'front_distance_index': 540,
        'max_speed': 1.0,
        'max_acceleration': 1.0,
        'max_deceleration': 2.5,
        'braking_stop_margin': 0.30,
        'speed_distance_gain': 0.9,
        'max_command_dt': 0.10,
        'stop_publish_count': 3,
        'stop_publish_interval': 0.05,
        'publish_queue_depth': 10,
        # Fallbacks for direct execution / SIM. Real-car launch loads vesc.yaml.
        'steering_angle_to_servo_gain': -1.2135,
        'steering_angle_to_servo_offset': 0.5304,
        'servo_min': 0.15,
        'servo_max': 0.85,
        'servo_limit_margin': 0.001,
    }

    def __init__(self):
        super().__init__('reactive_follow_gap_node')
        for name, default in self.PARAMETER_DEFAULTS.items():
            value = self.declare_parameter(
                name, default, ParameterDescriptor(read_only=True)).value
            setattr(self, name, value)
        self._validate_parameters()
        self.get_logger().set_level(
            rclpy.logging.LoggingSeverity.DEBUG if self.debug
            else rclpy.logging.LoggingSeverity.INFO)
        self.get_logger().info("Node started successfully")

        self.lidarscan_topic = '/scan'
        self.drive_topic = '/drive'

        self.subscription = self.create_subscription(
            LaserScan, self.lidarscan_topic, self.lidar_callback,
            qos_profile_sensor_data)
        self.publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, self.publish_queue_depth)
        self.marker_publisher = self.create_publisher(
            Marker, '/visualization_marker', self.publish_queue_depth)

        self.last_steering_angle = 0.0
        self.last_commanded_speed = 0.0
        self.last_command_time = None

        # servo = steering_gain * steering_angle + steering_offset
        angle_at_servo_min = (
            self.servo_min + self.servo_limit_margin
            - self.steering_angle_to_servo_offset) / self.steering_angle_to_servo_gain
        angle_at_servo_max = (
            self.servo_max - self.servo_limit_margin
            - self.steering_angle_to_servo_offset) / self.steering_angle_to_servo_gain
        self.min_steering_angle = min(
            angle_at_servo_min, angle_at_servo_max)
        self.max_steering_angle = max(
            angle_at_servo_min, angle_at_servo_max)

        self.get_logger().info(
            f"Steering angle limits: [{self.min_steering_angle:.4f}, "
            f"{self.max_steering_angle:.4f}] rad")

    def _validate_parameters(self):
        positive = (
            'max_distance', 'max_speed', 'max_deceleration', 'max_command_dt',
            'speed_distance_gain', 'max_acceleration',
        )
        nonnegative = (
            'bubble_radius', 'disparity_thresh', 'minimum_gap_angle',
            'gap_weight_power', 'braking_stop_margin',
            'stop_publish_interval', 'servo_limit_margin',
        )
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                constraint = 'positive' if name in positive else 'non-negative'
                raise ValueError(f'{name} must be finite and {constraint}')
        for name, minimum in (
                ('scan_range_offset', 0), ('front_distance_index', -1),
                ('preprocess_conv_size', 1), ('stop_publish_count', 1),
                ('publish_queue_depth', 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if (not all(math.isfinite(v) for v in (
                self.steering_angle_to_servo_gain,
                self.steering_angle_to_servo_offset, self.servo_min, self.servo_max))
                or self.steering_angle_to_servo_gain == 0
                or not 0 <= self.servo_min < self.servo_max <= 1
                or 2 * self.servo_limit_margin >= self.servo_max - self.servo_min):
            raise ValueError('Invalid steering servo calibration or servo_limit_margin')

    def stop_vehicle(self):
        """Publish a zero command before the drive publisher is destroyed."""
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        # Keep the latest obstacle-avoidance steering command while the vehicle
        # decelerates instead of forcing the wheels straight on Ctrl+C.
        stop_msg.drive.steering_angle = self.last_steering_angle

        # Publish more than once so the stop command can reach the controller
        # before this node and its DDS publisher are torn down.
        for _ in range(self.stop_publish_count):
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(stop_msg)
            time.sleep(self.stop_publish_interval)

        self.get_logger().info("Stop command published")

    def disparity_extender(self, ranges, angle_increment):
        if self.bubble_radius == 0:
            return
        # Start and end indices to process only the elements in front of the car
        start_index = self.scan_range_offset
        end_index = len(ranges) - self.scan_range_offset

        i = start_index
        while i < end_index - 1:
            # Check for disparity between consecutive measurements
            if abs(ranges[i] - ranges[i + 1]) > self.disparity_thresh:
                # Determine the lower element of the disparity
                lower_index = i if ranges[i] < ranges[i + 1] else i + 1
                lower_value = ranges[lower_index]

                # Set a bubble around the lower element
                extend_step = 0
                while extend_step < end_index - start_index:
                    # Calculate the angle for the current step
                    current_angle = extend_step * angle_increment
                    # Check if the point is within the bubble radius
                    if lower_value * math.tan(current_angle) < self.bubble_radius:
                        # Set elements within the bubble to zero, checking bounds
                        if lower_index - extend_step >= start_index:
                            ranges[lower_index - extend_step] = 0
                        if lower_index + extend_step < end_index:
                            ranges[lower_index + extend_step] = 0
                        extend_step += 1
                    else:
                        break

                # Move index past the extended bubble
                i = lower_index + extend_step
            else:
                i += 1

    def closest_point_zero_out(self, ranges, angle_increment):
        start_index = self.scan_range_offset
        end_index = len(ranges) - self.scan_range_offset

        # Find the closest non-zero point in the specified range
        min_distance = float('inf')
        closest_index = -1
        for i in range(start_index, end_index):
            if ranges[i] != 0 and ranges[i] < min_distance:
                min_distance = ranges[i]
                closest_index = i

        if closest_index != -1:  # If a closest point was found
            # Calculate the number of steps to extend based on the bubble radius
            extend_step = 0
            while extend_step < end_index - start_index:
                current_angle = extend_step * angle_increment
                if min_distance * math.tan(current_angle) < self.bubble_radius:
                    # Set elements within the bubble to zero, checking bounds
                    if closest_index - extend_step >= start_index:
                        ranges[closest_index - extend_step] = 0
                    if closest_index + extend_step < end_index:
                        ranges[closest_index + extend_step] = 0
                    extend_step += 1
                else:
                    break

    def preprocess_lidar(self, ranges):
        # Apply a mean filter and threshold the distances
        proc_ranges = np.array(ranges)

        window_size = self.preprocess_conv_size
        proc_ranges = np.convolve(proc_ranges, np.ones(window_size) / window_size, mode='same')
        proc_ranges[proc_ranges > self.max_distance] = self.max_distance
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        # Find the longest sequence of consecutive non-zero values (max gap)
        gaps = np.split(free_space_ranges, np.where(free_space_ranges == 0)[0])
        max_gap = max(gaps, key=len)
        start_i = np.where(free_space_ranges == max_gap[0])[0][0]
        end_i = start_i + len(max_gap) - 1
        return start_i, end_i

    def find_best_point(self, ranges, angle_increment):
        """Choose the deepest navigable gap without a left/right preference."""
        scan_start = self.scan_range_offset
        scan_end = len(ranges) - self.scan_range_offset
        gaps = []
        gap_start = None

        for index in range(scan_start, scan_end + 1):
            in_free_space = index < scan_end and ranges[index] > 0
            if in_free_space and gap_start is None:
                gap_start = index
            elif not in_free_space and gap_start is not None:
                segment = ranges[gap_start:index]
                weights = segment ** self.gap_weight_power
                total_weight = float(np.sum(weights))
                if total_weight > 0.0:
                    indices = np.arange(gap_start, index)
                    center = float(np.sum(indices * weights) / total_weight)
                else:
                    center = (gap_start + index - 1) / 2.0
                gaps.append({
                    'start': gap_start,
                    'end': index - 1,
                    'center': center,
                    'depth': float(np.max(segment)),
                })
                gap_start = None

        if not gaps:
            return len(ranges) // 2

        angular_resolution = max(abs(float(angle_increment)), 1.0e-9)
        minimum_gap_points = max(
            1, int(math.ceil(
                self.minimum_gap_angle / angular_resolution)))
        navigable_gaps = [
            gap for gap in gaps
            if gap['end'] - gap['start'] + 1 >= minimum_gap_points
        ]
        if not navigable_gaps:
            navigable_gaps = gaps

        scan_center = 0.5 * (scan_start + scan_end - 1)
        best_gap = max(
            navigable_gaps,
            key=lambda gap: (
                gap['depth'], -abs(gap['center'] - scan_center)))
        return int(round(best_gap['center']))

    def lidar_callback(self, data):
        ranges = np.array(data.ranges)
        angle_increment = data.angle_increment
        if (len(ranges) <= 2 * self.scan_range_offset
                or len(ranges) < self.preprocess_conv_size):
            raise ValueError('Scan is too short for scan_range_offset / preprocess_conv_size')
        if not math.isfinite(angle_increment) or angle_increment <= 0:
            raise ValueError('Scan angle_increment must be finite and positive')
        front_index = self.front_distance_index
        if front_index == -1:
            # Optional geometry-based lookup for scanners with other resolutions.
            front_index = int(round(-data.angle_min / angle_increment))
        if not self.scan_range_offset <= front_index < len(ranges) - self.scan_range_offset:
            raise ValueError('front_distance_index must lie inside the processed scan sector')

        proc_ranges = self.preprocess_lidar(ranges)
        # Read speed distance before steering bubbles mask obstacle ranges.
        front_distance = float(proc_ranges[front_index])
        self.disparity_extender(proc_ranges, angle_increment)
        self.closest_point_zero_out(proc_ranges, angle_increment)
        best_index = self.find_best_point(proc_ranges, angle_increment)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        angle = (best_index - len(ranges) / 2) * angle_increment
        angle = float(np.clip(
            angle, self.min_steering_angle, self.max_steering_angle))

        drive_msg.drive.steering_angle = angle
        self.last_steering_angle = angle
        # Zero target inside the stop margin, then increase linearly from zero.
        speed_distance = max(front_distance - self.braking_stop_margin, 0.0)
        desired_speed = min(
            self.speed_distance_gain * speed_distance, self.max_speed)

        # Bound speed changes in both directions. The first scan establishes
        # the time reference and publishes zero; subsequent scans ramp up.
        now = time.monotonic()
        dt = 0.0
        if self.last_command_time is not None:
            dt = min(max(now - self.last_command_time, 0.0), self.max_command_dt)
        minimum_reachable_speed = max(
            self.last_commanded_speed - self.max_deceleration * dt, 0.0)
        maximum_reachable_speed = (
            self.last_commanded_speed + self.max_acceleration * dt)
        set_speed = min(
            max(desired_speed, minimum_reachable_speed), maximum_reachable_speed)
        self.last_command_time = now
        self.last_commanded_speed = set_speed
        drive_msg.drive.speed = set_speed
        self.publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    reactive_node = ReactiveFollowGap()
    try:
        rclpy.spin(reactive_node)
    except KeyboardInterrupt:
        reactive_node.get_logger().info("Ctrl+C received, stopping vehicle")
    finally:
        if rclpy.ok():
            reactive_node.stop_vehicle()
        reactive_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
