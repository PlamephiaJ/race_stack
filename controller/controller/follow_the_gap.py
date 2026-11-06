#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

import time
from visualization_msgs.msg import Marker


class ReactiveFollowGap(Node):
    def __init__(self):
        super().__init__('reactive_follow_gap_node')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.get_logger().info("Node started successfully")

        self.lidarscan_topic = '/scan'
        self.drive_topic = '/drive'

        self.subscription = self.create_subscription(
            LaserScan, self.lidarscan_topic, self.lidar_callback,
            qos_profile_sensor_data)
        self.publisher = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.marker_publisher = self.create_publisher(Marker, '/visualization_marker', 10)

        self.bubble_radius = 0.25  # radius to keep clear of obstacles
        self.max_distance = 3.5   # max distance to consider for a valid gap
        self.disparity_thresh = 0.15
        self.last_steering_angle = 0.0
        self.declare_parameter('max_speed', 1.0)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.declare_parameter('max_deceleration', 2.5)
        self.declare_parameter('braking_stop_margin', 0.30)
        self.max_deceleration = float(
            self.get_parameter('max_deceleration').value)
        self.braking_stop_margin = float(
            self.get_parameter('braking_stop_margin').value)
        if self.max_speed <= 0.0:
            raise ValueError('max_speed must be positive')
        if self.max_deceleration <= 0.0:
            raise ValueError('max_deceleration must be positive')
        if self.braking_stop_margin < 0.0:
            raise ValueError('braking_stop_margin cannot be negative')
        self.last_commanded_speed = 0.0
        self.last_command_time = None
        self.declare_parameter('minimum_gap_angle', 0.12)
        self.minimum_gap_angle = float(self.get_parameter(
            'minimum_gap_angle').value)
        if self.minimum_gap_angle < 0.0:
            self.get_logger().warning(
                'minimum_gap_angle must be non-negative; using 0.12 rad')
            self.minimum_gap_angle = 0.12

        # Keep these defaults synchronized with f1tenth_stack/config/vesc.yaml.
        # servo = steering_gain * steering_angle + steering_offset
        self.declare_parameter('steering_angle_to_servo_gain', -1.2135)
        self.declare_parameter('steering_angle_to_servo_offset', 0.5304)
        self.declare_parameter('servo_min', 0.15)
        self.declare_parameter('servo_max', 0.85)
        self.declare_parameter('servo_limit_margin', 0.001)

        steering_gain = self.get_parameter(
            'steering_angle_to_servo_gain').value
        steering_offset = self.get_parameter(
            'steering_angle_to_servo_offset').value
        servo_min = self.get_parameter('servo_min').value
        servo_max = self.get_parameter('servo_max').value
        servo_margin = self.get_parameter('servo_limit_margin').value

        angle_at_servo_min = (
            servo_min + servo_margin - steering_offset) / steering_gain
        angle_at_servo_max = (
            servo_max - servo_margin - steering_offset) / steering_gain
        self.min_steering_angle = min(
            angle_at_servo_min, angle_at_servo_max)
        self.max_steering_angle = max(
            angle_at_servo_min, angle_at_servo_max)

        self.get_logger().info(
            f"Steering angle limits: [{self.min_steering_angle:.4f}, "
            f"{self.max_steering_angle:.4f}] rad")

    def stop_vehicle(self):
        """Publish a zero command before the drive publisher is destroyed."""
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        # Keep the latest obstacle-avoidance steering command while the vehicle
        # decelerates instead of forcing the wheels straight on Ctrl+C.
        stop_msg.drive.steering_angle = self.last_steering_angle

        # Publish more than once so the stop command can reach the controller
        # before this node and its DDS publisher are torn down.
        for _ in range(3):
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(stop_msg)
            time.sleep(0.05)

        self.get_logger().info("Stop command published")

    def disparity_extender(self, ranges, angle_increment):
        # Start and end indices to process only the elements in front of the car
        start_index = 180
        end_index = len(ranges) - 180
        # self.get_logger().info(f"End index:  {end_index}")

        i = start_index
        while i < end_index - 1:
            # Check for disparity between consecutive measurements
            if abs(ranges[i] - ranges[i + 1]) > self.disparity_thresh:
                # Determine the lower element of the disparity
                lower_index = i if ranges[i] < ranges[i + 1] else i + 1
                lower_value = ranges[lower_index]

                # Set a bubble around the lower element
                extend_step = 0
                while True:
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
        start_index = 180
        end_index = len(ranges) - 180

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
            while True:
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

        window_size = 3
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
        scan_start = 180
        scan_end = len(ranges) - 180
        gaps = []
        gap_start = None

        for index in range(scan_start, scan_end + 1):
            in_free_space = index < scan_end and ranges[index] > 0
            if in_free_space and gap_start is None:
                gap_start = index
            elif not in_free_space and gap_start is not None:
                segment = ranges[gap_start:index]
                weights = segment ** 2
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
        # start_time = time.time()
        ranges = np.array(data.ranges)
        angle_increment = data.angle_increment

        proc_ranges = self.preprocess_lidar(ranges)
        self.disparity_extender(proc_ranges, angle_increment)
        self.closest_point_zero_out(proc_ranges, angle_increment)

        # Eliminate all points inside 'bubble'
        # self.update_safety_bubble(proc_ranges, angle_increment, self.bubble_radius)

        # Find max length gap
        # start_i, end_i = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best_index = self.find_best_point(proc_ranges, angle_increment)
        target_distance = ranges[int(best_index)]

        front_distance = proc_ranges[540]
        # front_distance = target_distance # REMOVE THIS AFTER ********************************
        # Create and publish Ackermann drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()

        angle = (best_index - len(ranges) / 2) * data.angle_increment
        angle = float(np.clip(
            angle, self.min_steering_angle, self.max_steering_angle))

        # These 3 lines are for visulizing
        # target_distance = ranges[int(best_index)]
        # angle_increment = data.angle_increment  # Ensure angle_increment is correctly passed
        # self.publish_target_marker(best_index, target_distance, angle_increment)

        # self.get_logger().info(f"Best index is: {best_index}")
        # self.get_logger().info(f"Angle is: {angle:.2f}")
        # Adjust the speed

        max_speed = self.max_speed
        # if abs(angle)>0.78:
        #    set_speed = 0.9

        # self.get_logger().info(f"The angle is over maximum: {angle:.3f}")
        # else:
        if front_distance < 1.2:   # 0.349066=20 degree
            set_speed = 0.96       # * (front_distance/self.max_distance)
        elif front_distance < 4:
            set_speed = 0.9 * front_distance + 0.2
        else:
            set_speed = max_speed

            # self.get_logger().info(f"The speed reaches maximum: {set_speed:.3f}")
            # self.get_logger().info(f"When the front distance is: {front_distance:.3f}")
        # alternate choice of speed control
        # if angle < 0.2:
        #     set_speed = max_speed
        # elif angle < 0.5:
        #     set_speed = 1
        # else:i
            # set_speed = 0.5

        drive_msg.drive.steering_angle = angle
        self.last_steering_angle = angle
        # A distance-based stopping envelope prevents requesting a speed from
        # which the measured vehicle cannot stop before the forward obstacle.
        braking_distance = max(
            float(front_distance) - self.braking_stop_margin, 0.0)
        braking_speed = math.sqrt(
            2.0 * self.max_deceleration * braking_distance)
        desired_speed = min(set_speed, max_speed, braking_speed)

        # Model the measured braking capability in the command itself.  A
        # sudden LiDAR change must not imply physically impossible instant
        # deceleration in simulation or hide a collision that would occur on
        # the real car.
        now = time.monotonic()
        if self.last_command_time is None:
            set_speed = desired_speed
        else:
            dt = min(max(now - self.last_command_time, 0.0), 0.10)
            minimum_reachable_speed = max(
                self.last_commanded_speed - self.max_deceleration * dt, 0.0)
            set_speed = max(desired_speed, minimum_reachable_speed)
        self.last_command_time = now
        self.last_commanded_speed = set_speed
        drive_msg.drive.speed = set_speed  # Set your desired speed
        self.publisher.publish(drive_msg)

        # end_time = time.time()  # End time measurement
        # elapsed_time = end_time - start_time
        # self.get_logger().info(f"Execution time: {elapsed_time:.6f} seconds")

    # def publish_target_marker(self, best_index, target_distance, angle_increment):
    #     marker = Marker()
    #     marker.header.frame_id = "laser"  # Make sure this matches your RViz fixed frame
    #     marker.header.stamp = self.get_clock().now().to_msg()

    #     marker.ns = "target_point"
    #     marker.id = 0

    #     marker.type = Marker.SPHERE
    #     marker.action = Marker.ADD

    #     # Convert index and distance to coordinates
    #     angle = (best_index - len(data.ranges) / 2) * angle_increment
    #     x = target_distance * np.cos(angle)
    #     y = target_distance * np.sin(angle)

    #     marker.pose.position.x = x
    #     marker.pose.position.y = y
    #     marker.pose.position.z = 0

    #     marker.pose.orientation.x = 0.0
    #     marker.pose.orientation.y = 0.0
    #     marker.pose.orientation.z = 0.0
    #     marker.pose.orientation.w = 1.0

    #     marker.scale.x = 0.2
    #     marker.scale.y = 0.2
    #     marker.scale.z = 0.2

    #     marker.color.a = 1.0
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0

    #     self.marker_publisher.publish(marker)


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
