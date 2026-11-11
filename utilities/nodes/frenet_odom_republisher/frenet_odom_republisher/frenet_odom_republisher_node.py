import rclpy
from stack_master.qos import LATCHED_QOS
from rclpy.node import Node
from rclpy.time import Time

from f110_msgs.msg import WpntArray
from nav_msgs.msg import Odometry
from frenet_conversion.frenet_converter import FrenetConverter
import numpy as np
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener


class FrenetOdomRepublisher(Node):
    """Publishes the car state in Frenet coordinates.

    Pose comes from TF (map -> base_frame, default car_state/base_link), velocity from /state_estimation/odom.
    """

    def __init__(self):
        super().__init__('frenet_odom_republisher')
        self.base_frame = self.declare_parameter('base_frame', 'car_state/base_link').value
        self.max_tf_lag_sec = self.declare_parameter('max_tf_lag_sec', 0.3).value
        self.converter = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, qos=10)

        self.frenet_odom_publisher_ = self.create_publisher(Odometry, '/car_state/frenet/odom', 10)

        self.create_subscription(WpntArray, '/global_waypoints', self.global_trajectory_callback, LATCHED_QOS)
        self.create_subscription(Odometry, '/state_estimation/odom', self.odom_callback, 1)

    def global_trajectory_callback(self, msg):
        wpnts = msg.wpnts
        self.converter = FrenetConverter(np.array([w.x_m for w in wpnts]),
                                         np.array([w.y_m for w in wpnts]),
                                         np.array([w.psi_rad for w in wpnts]))

    def odom_callback(self, msg):
        if self.converter is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform('map', self.base_frame, Time())
        except Exception as e:
            self.get_logger().warn(f"map->{self.base_frame} lookup failed: {e}", throttle_duration_sec=1.0)
            return

        lag = abs((Time.from_msg(msg.header.stamp) -
                   Time.from_msg(tf.header.stamp)).nanoseconds) * 1e-9
        if lag > self.max_tf_lag_sec:
            self.get_logger().warn(
                f"map->{self.base_frame} TF lag {lag:.3f}s; skipping stale Frenet pose",
                throttle_duration_sec=1.0)
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        vel = msg.twist.twist.linear

        frenet_pos = self.converter.get_frenet([t.x], [t.y])
        frenet_vel = self.converter.get_frenet_velocities(vel.x, vel.y, theta)

        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = 'frenet'
        odom_msg.child_frame_id = str(self.converter.get_closest_index([t.x], [t.y])[0])
        odom_msg.pose.pose.position.x = frenet_pos[0, 0]
        odom_msg.pose.pose.position.y = frenet_pos[1, 0]
        odom_msg.twist.twist.linear.x = frenet_vel[0][0]
        odom_msg.twist.twist.linear.y = frenet_vel[1][0]

        self.frenet_odom_publisher_.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrenetOdomRepublisher()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
