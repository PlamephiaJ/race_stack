# Copyright (c) 2024 ForzaETH
# SPDX-License-Identifier: MIT
# Ported from ForzaETH/particle_filter at 85ecc1865ac687ad8cf31afadbcb826acb23262c.
# See third_party/synpf/LICENSE and SOURCE.md for attribution and porting changes.
"""ROS 2 SynPF backend using the stack's shared GPU PF transport and EKF."""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from state_estimation.particle_filter import ParticleFiler
from state_estimation import particle_filter_utils as Utils
from state_estimation import synpf_models as Models


class SynPF(ParticleFiler):
    def __init__(self):
        super().__init__(node_name='synpf')
        self.tracked_pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_pub_topic').value, 10)
        self.pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.get_parameter('pose_pub_topic').value + '/with_covariance', 10)
        self.get_logger().info(
            f'SynPF active: motion_model={self.motion_model_name}, '
            f'motion_alpha={self.alpha}, '
            f'translation_bias_sigma={self.motion_translation_bias_sigma}, '
            f'boxed beams={self.desired_beams}, particles={self.MAX_PARTICLES}, CUDA rmgpu')

    def configure_backend(self):
        self.particles_at_laser = False  # Upstream SynPF hypotheses are base poses.
        defaults = {
            'motion_model': 'tum', 'des_lidar_beams': 21, 'lidar_aspect_ratio': 3.5,
            'lambda_short': 0.25, 'lam_thresh': 0.1,
            'alpha_1_tum': 0.5, 'alpha_2_tum': 0.015,
            'alpha_3_tum': 1.0, 'alpha_4_tum': 0.1,
            'motion_translation_bias_sigma': 0.5,
            'alpha_1_amcl': 0.5, 'alpha_2_amcl': 0.5,
            'alpha_3_amcl': 1.0, 'alpha_4_amcl': 0.1,
            'out_of_map_weight': 0.01, 'pose_pub_topic': '/tracked_pose',
            'pub_covariance': True,
        }
        arc = {'x': 0.8, 'y': 1.0, 'theta': 0.75, 'xy': 1.0, 'x_min': 0.01,
               'y_min': 0.03, 'y_max': 1.0, 'theta_min': 0.01, 'xy_min_x': 0.1}
        defaults.update({'motion_dispersion_arc_' + name: value for name, value in arc.items()})
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        get = lambda name: self.get_parameter(name).value
        self.motion_model_name = get('motion_model')
        if self.motion_model_name not in ('tum', 'amcl', 'arc', 'default'):
            raise ValueError('SynPF motion_model must be tum, amcl, arc or default')
        self.alpha = tuple(get(f'alpha_{i}_{self.motion_model_name}') for i in range(1, 5)) \
            if self.motion_model_name in ('tum', 'amcl') else None
        self.motion_translation_bias_sigma = float(get('motion_translation_bias_sigma'))
        self.lam_thresh = get('lam_thresh')
        self.lambda_short = get('lambda_short')
        self.desired_beams = get('des_lidar_beams')
        self.aspect = get('lidar_aspect_ratio')
        self.out_of_map_weight = get('out_of_map_weight')
        self.arc_params = {name: get('motion_dispersion_arc_' + name) for name in arc}
        if (self.desired_beams < 3 or self.desired_beams % 2 != 1 or self.aspect <= 0
                or self.lam_thresh <= 0 or not 0 < self.out_of_map_weight <= 1):
            raise ValueError('Invalid SynPF beam geometry, lam_thresh or out_of_map_weight')
        if self.alpha is not None and any(a < 0 or not np.isfinite(a) for a in self.alpha):
            raise ValueError('SynPF motion noise coefficients must be finite and nonnegative')
        if (not np.isfinite(self.motion_translation_bias_sigma)
                or self.motion_translation_bias_sigma < 0):
            raise ValueError('SynPF translation bias must be finite and nonnegative')
        if self.motion_model_name == 'arc':
            self.get_logger().warning('SynPF arc motion model is experimental; default tum is preferred.')
        self._scan_geometry = None

    def scan_indices(self, msg):
        geometry = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if geometry != self._scan_geometry:
            angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
            self._boxed_indices = Models.boxed_beam_indices(angles, self.desired_beams, self.aspect)
            self._scan_geometry = geometry
        return self._boxed_indices

    def precompute_sensor_model(self):
        self.sensor_model_table = Models.sensor_model_table(
            self.MAX_RANGE_PX, self.map_info.resolution, self.SIGMA_HIT, self.lambda_short,
            self.Z_HIT, self.Z_SHORT, self.Z_MAX, self.Z_RAND)
        self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, proposal_dist, action):
        if self.motion_model_name in ('tum', 'amcl'):
            Models.differential_motion(proposal_dist, action, self.alpha,
                                       self.lam_thresh, self.motion_model_name,
                                       translation_bias_sigma=self.motion_translation_bias_sigma)
        elif self.motion_model_name == 'arc':
            Models.arc_motion(proposal_dist, action, self.arc_params)
        else:
            super().motion_model(proposal_dist, action)

    def laser_particles(self, particles):
        return Models.base_particles_to_laser(particles, self.base_to_laser_matrix)

    def sensor_model(self, proposal_dist, obs, weights):
        # Ray-cast from each hypothesized laser pose, including full planar mounting yaw/xy.
        super().sensor_model(self.laser_particles(proposal_dist), obs, weights)

    def weight_proposals(self, particles, weights):
        pixels = particles.copy()
        Utils.world_to_map(pixels, self.map_info)
        x, y = np.floor(pixels[:, :2]).astype(int).T
        height, width = self.permissible_region.shape
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        free = np.zeros(len(particles), dtype=bool)
        free[inside] = self.permissible_region[y[inside], x[inside]]
        return weights * np.where(free, 1.0, self.out_of_map_weight)

    def publish_tf(self, pose, stamp=None):
        super().publish_tf(pose, stamp)
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.pose = Utils.particle_to_pose(pose)
        self.tracked_pose_pub.publish(msg)
        if self.get_parameter('pub_covariance').value:
            covariance = PoseWithCovarianceStamped()
            covariance.header = msg.header
            covariance.pose.pose = msg.pose
            covariance.pose.covariance = Utils.planar_covariance_to_ros(
                Utils.weighted_pose_covariance(self.particles, self.weights, pose))
            self.pose_cov_pub.publish(covariance)


def main(args=None):
    rclpy.init(args=args)
    node = SynPF()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
