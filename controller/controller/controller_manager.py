import os
import math
import time
import yaml
import rclpy
from stack_master.qos import LATCHED_QOS
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.time import Time
from rclpy.client import Client
from rcl_interfaces.srv import GetParameters

from ament_index_python import get_package_share_directory
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import (CarStateStamped, GapData, ObstacleArray, PidData, Wpnt, WpntArray)
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float64, String
from steering_lookup.lookup_steer_angle import LookupSteerAngle
from visualization_msgs.msg import Marker, MarkerArray
from frenet_conversion.frenet_converter import FrenetConverter

from controller.map import MAP_Controller
from controller.pp import PP_Controller
from controller.ftg import FTG_Controller
from controller.tracking_safety import RecoverySpeedLimiter, front_speed_limit, path_tracking_errors, time_to_collision
from rcl_interfaces.msg import ParameterValue, ParameterType, ParameterDescriptor, FloatingPointRange, IntegerRange
from tf_transformations import quaternion_from_euler
from stack_master.parameter_event_handler import ParameterEventHandler

from pbl_config import get_remote_parameter

from typing import List, Dict, Union


class Controller(Node):
    def __init__(self):
        super().__init__('controller_manager',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        # remote parameters
        self.map_path = get_remote_parameter(self, '/global_parameters', 'map_path')
        self.racecar_version = get_remote_parameter(self, '/global_parameters', 'racecar_version')
        self.sim = get_remote_parameter(self, '/global_parameters', 'sim')
        self.state_machine_rate = get_remote_parameter(self, 'state_machine', 'rate_hz')

        # variables
        self.rate = 40
        self.state = "GB_TRACK"

        self.LUT_name = self.get_parameter('LU_table').value  # name of lookup table
        self.get_logger().info(f"Using {self.LUT_name}")
        self.mode = self.get_parameter('mode').value
        self.mapping = self.get_parameter('mapping').value

        # Publishers
        self.publish_topic = '/drive'  # TODO adapt car topic name here
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.publish_topic, 10)
        self.steering_pub = self.create_publisher(Marker, 'steering', 10)
        self.lookahead_pub = self.create_publisher(Marker, 'lookahead_point', 10)
        self.trailing_pub = self.create_publisher(Marker, 'trailing_opponent_marker', 10)
        self.waypoint_pub = self.create_publisher(MarkerArray, 'my_waypoints', 10)
        self.l1_pub = self.create_publisher(Point, 'l1_distance', 10)
        self.gap_data = self.create_publisher(PidData, '/trailing/gap_data', 10)

        self.track_length = None
        self.opponent = None
        self.waypoint_array_in_map = None
        self.speed_now = None
        self.position_in_map = None
        self.position_in_map_frenet = None
        self.last_odom_time = None
        self.last_frenet_time = None
        self.last_waypoints_time = None
        self.last_odom_stamp_ns = None
        self.last_frenet_stamp_ns = None
        self.last_waypoints_stamp_ns = None
        self.last_scan_time = None
        self.last_scan_stamp_ns = None
        self.scan = None
        self.last_safe_steer = 0.0
        self.front_stop_latched = False
        self.front_speed_cap = 0.0
        self.front_ttc = float('inf')
        self.applied_speed = None
        self.last_applied_time = None
        self.last_applied_stamp_ns = None
        self.last_control_time = time.monotonic()
        if not self.has_parameter('max_input_age_sec'):
            self.declare_parameter('max_input_age_sec', 0.5)
        if not self.has_parameter('max_odom_age_sec'):
            self.declare_parameter('max_odom_age_sec', 0.3)
        if not self.has_parameter('max_path_start_rear_m'):
            self.declare_parameter('max_path_start_rear_m', 0.75)
        if not self.has_parameter('front_guard_enabled'):
            self.declare_parameter('front_guard_enabled', True)
        if not self.has_parameter('front_clearance_m'):
            self.declare_parameter('front_clearance_m', 0.30)
        if not self.has_parameter('front_decel_mps2'):
            self.declare_parameter('front_decel_mps2', 0.8)
        if not self.has_parameter('front_ttc_stop_sec'):
            self.declare_parameter('front_ttc_stop_sec', 1.0)
        if not self.has_parameter('front_ttc_release_sec'):
            self.declare_parameter('front_ttc_release_sec', 1.2)
        if not self.has_parameter('front_speed_margin'):
            self.declare_parameter('front_speed_margin', 0.8)
        if not self.has_parameter('require_applied_feedback'):
            # The gym consumes /drive directly and has no actuator feedback stream.
            self.declare_parameter('require_applied_feedback', not self.sim)
        self.max_input_age_sec = float(self.get_parameter('max_input_age_sec').value)
        self.max_odom_age_sec = float(self.get_parameter('max_odom_age_sec').value)
        self.max_path_start_rear_m = float(self.get_parameter('max_path_start_rear_m').value)
        self.front_guard_enabled = bool(self.get_parameter('front_guard_enabled').value)
        self.front_clearance_m = float(self.get_parameter('front_clearance_m').value)
        self.front_decel_mps2 = float(self.get_parameter('front_decel_mps2').value)
        self.front_ttc_stop_sec = float(self.get_parameter('front_ttc_stop_sec').value)
        self.front_ttc_release_sec = float(self.get_parameter('front_ttc_release_sec').value)
        self.front_speed_margin = float(self.get_parameter('front_speed_margin').value)
        self.require_applied_feedback = bool(self.get_parameter('require_applied_feedback').value)
        safety_values = (self.max_input_age_sec, self.max_odom_age_sec,
                         self.max_path_start_rear_m, self.front_clearance_m,
                         self.front_decel_mps2, self.front_ttc_stop_sec, self.front_ttc_release_sec,
                         self.front_speed_margin)
        if (not all(math.isfinite(v) for v in safety_values)
                or self.max_input_age_sec <= 0 or self.max_odom_age_sec <= 0
                or self.max_path_start_rear_m < 0 or self.front_clearance_m <= 0
                or self.front_decel_mps2 <= 0 or self.front_ttc_stop_sec <= 0
                or self.front_ttc_release_sec <= self.front_ttc_stop_sec
                or not 0 < self.front_speed_margin < 1):
            raise ValueError('Invalid controller input or front clearance limit')

        recovery_defaults = {
            'recovery_speed_mps': 0.5,
            'recovery_full_speed_distance_m': 0.35,
            'recovery_min_speed_distance_m': 1.5,
            'recovery_full_speed_heading_deg': 25.0,
            'recovery_min_speed_heading_deg': 90.0,
            'recovery_direction_tolerance_deg': 10.0,
            'recovery_release_accel_mps2': 1.0,
        }
        for name, default in recovery_defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
        self.recovery_limiter = RecoverySpeedLimiter(
            speed=float(self.get_parameter('recovery_speed_mps').value),
            full_speed_distance=float(self.get_parameter('recovery_full_speed_distance_m').value),
            min_speed_distance=float(self.get_parameter('recovery_min_speed_distance_m').value),
            full_speed_heading=math.radians(float(self.get_parameter('recovery_full_speed_heading_deg').value)),
            min_speed_heading=math.radians(float(self.get_parameter('recovery_min_speed_heading_deg').value)),
            direction_tolerance=math.radians(float(self.get_parameter('recovery_direction_tolerance_deg').value)),
            release_acceleration=float(self.get_parameter('recovery_release_accel_mps2').value))

        # buffers for improved computation
        self.waypoint_array_buf = MarkerArray()
        self.markers_buf = [Marker() for _ in range(1000)]

        # controller choice
        # TODO having to set prioritize_dyn like this is really ugly.
        if self.mode == "MAP":
            self.get_logger().info("Initializing MAP  controller")
            self.init_map_controller()
            self.prioritize_dyn = self.l1_params["prioritize_dyn"]
        elif self.mode == "PP":
            self.get_logger().info("Initializing PP controller")
            self.init_pp_controller()
            self.prioritize_dyn = self.l1_params["prioritize_dyn"]
        elif self.mode == "FTG":
            self.get_logger().info("Initializing FTG controller")
            self.init_ftg_controller()
            self.prioritize_dyn = False
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")
            return

        # Subscribers
        # self.create_subscription(Config,
        # '/l1_param_tuner/parameter_updates',self.l1_params_cb) #l1 param
        # tuning/updating
        self.create_subscription(String, '/state', self.state_cb, 10)
        self.create_subscription(WpntArray, '/global_waypoints', self.track_length_cb, LATCHED_QOS)
        self.create_subscription(ObstacleArray, '/perception/obstacles', self.obstacle_cb, 10)
        self.create_subscription(WpntArray, '/local_waypoints', self.local_waypoint_cb,
                                 1)  # waypoints (x, y, v, norm trackbound, s, kappa)
        self.create_subscription(Odometry, '/state_estimation/odom', self.odom_cb, 1)  # car speed
        self.create_subscription(
            Odometry,
            '/car_state/frenet/odom',
            self.car_state_frenet_cb,
            1)  # car frenet coordinates
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 1)  # lidar scan
        self.create_subscription(
            AckermannDriveStamped, '/ackermann_cmd_applied', self.applied_command_cb, 1)

        # Block until relevant data is here
        self.wait_for_messages()

        # Init converter
        self.converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])

        # dyn params
        self.param_handler = ParameterEventHandler(self)
        self.callback_handle = self.param_handler.add_parameter_event_callback(
            callback=self.l1_param_cb,
        )
        param_dicts = [{'name': 't_clip_min',
                        'default': self.l1_params["t_clip_min"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
                       {'name': 't_clip_max',
                        'default': self.l1_params["t_clip_max"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
                       {'name': 'm_l1',
                        'default': self.l1_params["m_l1"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.001)])},
                       {'name': 'q_l1',
                        'default': self.l1_params["q_l1"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=-1.0, to_value=1.0, step=0.001)])},
                       {'name': 'speed_lookahead',
                        'default': self.l1_params["speed_lookahead"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
                       {'name': 'lat_err_coeff',
                        'default': self.l1_params["lat_err_coeff"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
                       {'name': 'acc_scaler_for_steer',
                        'default': self.l1_params["acc_scaler_for_steer"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
                       {'name': 'dec_scaler_for_steer',
                        'default': self.l1_params["dec_scaler_for_steer"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.5, step=0.01)])},
                       {'name': 'start_scale_speed',
                        'default': self.l1_params["start_scale_speed"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
                       {'name': 'end_scale_speed',
                        'default': self.l1_params["end_scale_speed"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=10.0, step=0.01)])},
                       {'name': 'downscale_factor',
                        'default': self.l1_params["downscale_factor"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.01)])},
                       {'name': 'speed_lookahead_for_steer',
                        'default': self.l1_params["speed_lookahead_for_steer"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.2, step=0.01)])},
                       {'name': 'prioritize_dyn',
                        'default': self.prioritize_dyn,
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)},
                       {'name': 'trailing_gap',
                        'default': self.l1_params["trailing_gap"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.1)])},
                       {'name': 'trailing_p_gain',
                        'default': self.l1_params["trailing_p_gain"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.01)])},
                       {'name': 'trailing_i_gain',
                        'default': self.l1_params["trailing_i_gain"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.5, step=0.001)])},
                       {'name': 'trailing_d_gain',
                        'default': self.l1_params["trailing_d_gain"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
                       {'name': 'blind_trailing_speed',
                        'default': self.l1_params["blind_trailing_speed"],
                        'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=3.0, step=0.01)])}]
        params = self.delcare_dyn_parameters(param_dicts)
        self.set_parameters(params)

        # main loop
        self.create_timer(1 / self.rate, self.control_loop)

        self.get_logger().info("Controller ready")

    def wait_for_messages(self):
        self.get_logger().info('Controller Manager waiting for messages...')
        track_length_print = False
        waypoint_array_print = False
        state_print = False

        while not track_length_print or not waypoint_array_print or not state_print:
            rclpy.spin_once(self)
            if self.track_length is not None and not track_length_print:
                self.get_logger().info('Received track length')
                track_length_print = True
            if self.waypoint_array_in_map is not None and not waypoint_array_print:
                self.get_logger().info('Received waypoint array')
                waypoint_array_print = True
            if self.speed_now is not None and self.position_in_map is not None and self.position_in_map_frenet is not None and not state_print:
                self.get_logger().info('Received car state messages')
                state_print = True

        self.get_logger().info('All required messages received. Continuing...')

    def delcare_dyn_parameters(self, param_dicts):
        params = []
        for param_dict in param_dicts:
            param = self.declare_parameter(param_dict['name'], param_dict['default'], param_dict['descriptor'])
            params.append(param)
        return params

    def init_map_controller(self):

        # get l1 parameres
        stack_master_path = get_package_share_directory('stack_master')
        config_path = os.path.join(stack_master_path, 'config', self.racecar_version, 'l1_params.yaml')
        with open(config_path, 'r') as f:
            self.l1_params = yaml.safe_load(f)
            self.l1_params = self.l1_params['controller']['ros__parameters']

        self.create_subscription(Imu, '/vesc/sensors/imu/raw', self.imu_cb,
                                 10)  # acceleration subscriber for steer change
        self.acc_now = np.zeros(10)  # rolling buffer for acceleration

        self.map_controller = MAP_Controller(
            self.l1_params["t_clip_min"],
            self.l1_params["t_clip_max"],
            self.l1_params["m_l1"],
            self.l1_params["q_l1"],
            self.l1_params["speed_lookahead"],
            self.l1_params["lat_err_coeff"],
            self.l1_params["acc_scaler_for_steer"],
            self.l1_params["dec_scaler_for_steer"],
            self.l1_params["start_scale_speed"],
            self.l1_params["end_scale_speed"],
            self.l1_params["downscale_factor"],
            self.l1_params["speed_lookahead_for_steer"],
            self.l1_params["prioritize_dyn"],
            self.l1_params["trailing_gap"],
            self.l1_params["trailing_p_gain"],
            self.l1_params["trailing_i_gain"],
            self.l1_params["trailing_d_gain"],
            self.l1_params["blind_trailing_speed"],
            self.rate,
            self.LUT_name)

    def init_pp_controller(self):

        # get l1 parameres
        stack_master_path = get_package_share_directory('stack_master')
        config_path = os.path.join(stack_master_path, 'config', self.racecar_version, 'l1_params.yaml')
        with open(config_path, 'r') as f:
            self.l1_params = yaml.safe_load(f)
            self.l1_params = self.l1_params['controller']['ros__parameters']

        # get wheelbase
        if self.sim:
            config_file = (
                'sim_params.yaml'
                if self.racecar_version == 'SIM'
                else 'car_model.yaml'
            )
            config_path = os.path.join(
                stack_master_path, 'config',
                self.racecar_version, config_file
            )
            with open(config_path, 'r') as f:
                car_params = yaml.safe_load(f)
                self.wheelbase = car_params['lr'] + car_params['lf']
        else:
            config_path = os.path.join(
                stack_master_path, 'config',
                self.racecar_version, 'vesc.yaml'
            )
            with open(config_path, 'r') as f:
                car_params = yaml.safe_load(f)
                self.wheelbase = car_params[
                    'vesc_to_odom_node'
                ]['ros__parameters']['wheelbase']

        self.create_subscription(Imu, '/vesc/sensors/imu/raw', self.imu_cb,
                                 10)  # acceleration subscriber for steer change
        self.acc_now = np.zeros(10)  # rolling buffer for acceleration

        self.pp_controller = PP_Controller(
            self.l1_params["t_clip_min"],
            self.l1_params["t_clip_max"],
            self.l1_params["m_l1"],
            self.l1_params["q_l1"],
            self.l1_params["speed_lookahead"],
            self.l1_params["lat_err_coeff"],
            self.l1_params["acc_scaler_for_steer"],
            self.l1_params["dec_scaler_for_steer"],
            self.l1_params["start_scale_speed"],
            self.l1_params["end_scale_speed"],
            self.l1_params["downscale_factor"],
            self.l1_params["speed_lookahead_for_steer"],
            self.l1_params["prioritize_dyn"],
            self.l1_params["trailing_gap"],
            self.l1_params["trailing_p_gain"],
            self.l1_params["trailing_i_gain"],
            self.l1_params["trailing_d_gain"],
            self.l1_params["blind_trailing_speed"],
            self.rate,
            self.wheelbase)

    def init_ftg_controller(self):

        # TODO fix to work without alternative value
        self.state_machine_debug = get_remote_parameter(self, 'state_machine', 'debug')
        self.state_machine_safety_radius = get_remote_parameter(self, 'state_machine', 'safety_radius')
        self.state_machine_max_lidar_dist = get_remote_parameter(self, 'state_machine', 'max_lidar_dist')
        self.state_machine_max_speed = get_remote_parameter(self, 'state_machine', 'max_speed')
        self.state_machine_range_offset = get_remote_parameter(self, 'state_machine', 'range_offset')
        self.state_machine_track_width = get_remote_parameter(self, 'state_machine', 'track_width')

        self.get_logger().info(
            f"FTG Controller parameters: {self.state_machine_debug}, {self.state_machine_safety_radius}, {self.state_machine_max_lidar_dist}, {self.state_machine_max_speed}, {self.state_machine_range_offset}, {self.state_machine_track_width}")

        self.ftg_controller = FTG_Controller(
            mapping=self.mapping,
            debug=self.state_machine_debug,
            safety_radius=self.state_machine_safety_radius,
            max_lidar_dist=self.state_machine_max_lidar_dist,
            max_speed=self.state_machine_max_speed,
            range_offset=self.state_machine_range_offset,
            track_width=self.state_machine_track_width)

    #############
    # CALLBACKS #
    #############

    def l1_param_cb(self, parameter_event):
        """
        Notices the change in the parameters and alters the spline params accordingly
        """

        if parameter_event.node != self.get_fully_qualified_name() or self.mode == "FTG":
            return

        if self.mode == "MAP":
            self.map_controller.t_clip_min = self.get_parameter('t_clip_min').value
            self.map_controller.t_clip_max = self.get_parameter('t_clip_max').value
            self.map_controller.m_l1 = self.get_parameter('m_l1').value
            self.map_controller.q_l1 = self.get_parameter('q_l1').value
            self.map_controller.speed_lookahead = self.get_parameter('speed_lookahead').value
            self.map_controller.lat_err_coeff = self.get_parameter('lat_err_coeff').value
            self.map_controller.acc_scaler_for_steer = self.get_parameter('acc_scaler_for_steer').value
            self.map_controller.dec_scaler_for_steer = self.get_parameter('dec_scaler_for_steer').value
            self.map_controller.start_scale_speed = self.get_parameter('start_scale_speed').value
            self.map_controller.end_scale_speed = self.get_parameter('end_scale_speed').value
            self.map_controller.downscale_factor = self.get_parameter('downscale_factor').value
            self.map_controller.speed_lookahead_for_steer = self.get_parameter('speed_lookahead_for_steer').value
            self.map_controller.prioritize_dyn = self.get_parameter('prioritize_dyn').value
            self.map_controller.trailing_gap = self.get_parameter('trailing_gap').value
            self.map_controller.trailing_p_gain = self.get_parameter('trailing_p_gain').value
            self.map_controller.trailing_i_gain = self.get_parameter('trailing_i_gain').value
            self.map_controller.trailing_d_gain = self.get_parameter('trailing_d_gain').value
            self.map_controller.blind_trailing_speed = self.get_parameter('blind_trailing_speed').value
        elif self.mode == "PP":
            self.pp_controller.t_clip_min = self.get_parameter('t_clip_min').value
            self.pp_controller.t_clip_max = self.get_parameter('t_clip_max').value
            self.pp_controller.m_l1 = self.get_parameter('m_l1').value
            self.pp_controller.q_l1 = self.get_parameter('q_l1').value
            self.pp_controller.speed_lookahead = self.get_parameter('speed_lookahead').value
            self.pp_controller.lat_err_coeff = self.get_parameter('lat_err_coeff').value
            self.pp_controller.acc_scaler_for_steer = self.get_parameter('acc_scaler_for_steer').value
            self.pp_controller.dec_scaler_for_steer = self.get_parameter('dec_scaler_for_steer').value
            self.pp_controller.start_scale_speed = self.get_parameter('start_scale_speed').value
            self.pp_controller.end_scale_speed = self.get_parameter('end_scale_speed').value
            self.pp_controller.downscale_factor = self.get_parameter('downscale_factor').value
            self.pp_controller.speed_lookahead_for_steer = self.get_parameter('speed_lookahead_for_steer').value
            self.pp_controller.prioritize_dyn = self.get_parameter('prioritize_dyn').value
            self.pp_controller.trailing_gap = self.get_parameter('trailing_gap').value
            self.pp_controller.trailing_p_gain = self.get_parameter('trailing_p_gain').value
            self.pp_controller.trailing_i_gain = self.get_parameter('trailing_i_gain').value
            self.pp_controller.trailing_d_gain = self.get_parameter('trailing_d_gain').value
            self.pp_controller.blind_trailing_speed = self.get_parameter('blind_trailing_speed').value
        self.get_logger().info("Updated parameters")

    def scan_cb(self, data: LaserScan):
        self.scan = data
        self.last_scan_time = time.monotonic()
        self.last_scan_stamp_ns = Time.from_msg(data.header.stamp).nanoseconds

    def applied_command_cb(self, data: AckermannDriveStamped):
        self.applied_speed = data.drive.speed
        self.last_applied_time = time.monotonic()
        self.last_applied_stamp_ns = Time.from_msg(data.header.stamp).nanoseconds

    def track_length_cb(self, data: WpntArray):
        self.track_length = data.wpnts[-1].s_m
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m, wpnt.psi_rad] for wpnt in data.wpnts])

    def obstacle_cb(self, data: ObstacleArray):
        if len(data.obstacles) > 0 and \
                self.position_in_map_frenet is not None and len(self.position_in_map_frenet) and \
                self.track_length is not None:

            self.opponent_s = None
            static_flag = False  # If we have a Static and a Dynamic obstacle we prefer the dynamic
            closest_opp = self.track_length
            for obstacle in data.obstacles:
                opponent_dist = (obstacle.s_start - self.position_in_map_frenet[0]) % self.track_length
                if opponent_dist < closest_opp or (static_flag and not obstacle.is_static):
                    closest_opp = opponent_dist
                    opponent_static = obstacle.is_static
                    opponent_s = obstacle.s_center
                    opponent_d = obstacle.d_center
                    opponent_vs = obstacle.vs
                    opponent_visible = obstacle.is_visible
                    if opponent_static:
                        static_flag = self.prioritize_dyn  # Chosen Obstacle is static
                    else:
                        static_flag = False  # Chosen obstacle is dynamic
                    self.opponent = [opponent_s, opponent_d, opponent_vs, opponent_static, opponent_visible]
        else:
            self.opponent = None

    def odom_cb(self, data: Odometry):
        self.last_odom_time = time.monotonic()
        self.last_odom_stamp_ns = Time.from_msg(data.header.stamp).nanoseconds
        self.speed_now = data.twist.twist.linear.x
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        rot = Rotation.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                                    data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        rot_euler = rot.as_euler('xyz', degrees=False)
        theta = rot_euler[2]
        self.position_in_map = np.array([x, y, theta])[np.newaxis]

    def local_waypoint_cb(self, data: WpntArray):
        self.last_waypoints_time = time.monotonic()
        self.last_waypoints_stamp_ns = Time.from_msg(data.header.stamp).nanoseconds
        self.waypoint_list_in_map = []
        for waypoint in data.wpnts:
            waypoint_in_map = [waypoint.x_m, waypoint.y_m]
            speed = waypoint.vx_mps
            if waypoint.d_right + waypoint.d_left != 0:
                self.waypoint_list_in_map.append([waypoint_in_map[0],
                                                  waypoint_in_map[1],
                                                  speed,
                                                  min(waypoint.d_left,
                                                      waypoint.d_right) / (waypoint.d_right + waypoint.d_left),
                                                  waypoint.s_m,
                                                  waypoint.kappa_radpm,
                                                  waypoint.psi_rad,
                                                  waypoint.ax_mps2])
            else:
                self.waypoint_list_in_map.append([waypoint_in_map[0],
                                                  waypoint_in_map[1],
                                                  speed,
                                                  0,
                                                  waypoint.s_m,
                                                  waypoint.kappa_radpm,
                                                  waypoint.psi_rad,
                                                  waypoint.ax_mps2])
        self.waypoint_array_in_map = np.array(self.waypoint_list_in_map)

    def imu_cb(self, data):
        # save acceleration in a rolling buffer
        self.acc_now[1:] = self.acc_now[:-1]
        self.acc_now[0] = -data.linear_acceleration.y  # vesc is rotated 90 deg, so (-acc_y) == (long_acc)

    def car_state_frenet_cb(self, data: Odometry):
        self.last_frenet_time = time.monotonic()
        self.last_frenet_stamp_ns = Time.from_msg(data.header.stamp).nanoseconds
        s = data.pose.pose.position.x
        d = data.pose.pose.position.y
        vs = data.twist.twist.linear.x
        vd = data.twist.twist.linear.y
        self.position_in_map_frenet = np.array([s, d, vs, vd])

    def l1_params_cb(self):
        pass

    def state_cb(self, data):
        self.state = data.data

    def map_cycle(self, braking=False):
        speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint = self.map_controller.main_loop(self.state,
                                                                                                                               self.position_in_map,
                                                                                                                               self.waypoint_array_in_map,
                                                                                                                               self.speed_now,
                                                                                                                               self.opponent,
                                                                                                                               self.position_in_map_frenet,
                                                                                                                               self.acc_now,
                                                                                                                               self.track_length,
                                                                                                                               steering_speed=self.speed_now if braking else None)
        self.set_lookahead_marker(L1_point, 100)
        self.visualize_steering(steering_angle)
        self.visualize_trailing_opponent()
        self.l1_pub.publish(Point(x=float(idx_nearest_waypoint), y=L1_distance))

        # Input age is checked in seconds before this cycle, including braking.
        self.map_controller.flag1 = False

        # Publish PID data of the Trailing controller for tuning
        if self.map_controller.flag1:
            pid_msg = self.create_pid_msg(self.map_controller.gap_should,
                                          self.map_controller.gap,
                                          self.map_controller.gap_error,
                                          self.map_controller.v_diff,
                                          self.map_controller.i_gap,
                                          self.map_controller.trailing_command)
            self.gap_data.publish(pid_msg)

        return speed, steering_angle

    def pp_cycle(self):
        speed, acceleration, jerk, steering_angle, L1_point, L1_distance, idx_nearest_waypoint = self.pp_controller.main_loop(self.state,
                                                                                                                              self.position_in_map,
                                                                                                                              self.waypoint_array_in_map,
                                                                                                                              self.speed_now,
                                                                                                                              self.opponent,
                                                                                                                              self.position_in_map_frenet,
                                                                                                                              self.acc_now,
                                                                                                                              self.track_length)
        self.set_lookahead_marker(L1_point, 100)
        self.visualize_steering(steering_angle)
        self.visualize_trailing_opponent()
        self.l1_pub.publish(Point(x=float(idx_nearest_waypoint), y=L1_distance))

        # Input age is checked in seconds before this cycle, including braking.
        self.pp_controller.flag1 = False

        # Publish PID data of the Trailing controller for tuning
        if self.pp_controller.flag1:
            pid_msg = self.create_pid_msg(self.pp_controller.gap_should,
                                          self.pp_controller.gap,
                                          self.pp_controller.gap_error,
                                          self.pp_controller.v_diff,
                                          self.pp_controller.i_gap,
                                          self.pp_controller.trailing_command)
            self.gap_data.publish(pid_msg)
        return speed, steering_angle

    def ftg_cycle(self):
        speed, steer = self.ftg_controller.process_lidar(self.scan.ranges)
        self.get_logger().warning("[STATE MACHINE] FTGONLY!!!")
        return speed, steer

    def create_pid_msg(self, should, actual, error, d_value, i_value, input):
        pid_msg = PidData()
        pid_msg.header.stamp = self.get_clock().now().to_msg()
        pid_msg.should = should
        pid_msg.actual = actual
        pid_msg.error = error
        pid_msg.d_value = d_value
        pid_msg.i_value = i_value
        pid_msg.input = input
        return pid_msg

    #############
    # MAIN LOOP #
    #############
    def inputs_ready(self, now, check_path_start=True):
        """Reject missing, delayed, or clearly misplaced control inputs."""
        ros_now_ns = self.get_clock().now().nanoseconds
        for name, received, stamp_ns, limit in (
                ('odometry', self.last_odom_time, self.last_odom_stamp_ns,
                 self.max_odom_age_sec),
                ('Frenet pose', self.last_frenet_time, self.last_frenet_stamp_ns,
                 self.max_input_age_sec),
                ('local waypoints', self.last_waypoints_time,
                 self.last_waypoints_stamp_ns, self.max_input_age_sec)):
            receive_age = float('inf') if received is None else now - received
            stamp_age = (float('inf') if stamp_ns is None else
                         (ros_now_ns - stamp_ns) * 1e-9)
            if receive_age > limit or abs(stamp_age) > limit:
                self.get_logger().warning(
                    f'[controller_manager] {name} is stale '
                    f'(receive {receive_age:.2f}s, stamp {stamp_age:.2f}s); stopping',
                    throttle_duration_sec=1.0)
                return False

        if (self.position_in_map is None or self.waypoint_array_in_map is None
                or self.waypoint_array_in_map.ndim != 2
                or path_tracking_errors(self.position_in_map[0], self.waypoint_array_in_map) is None
                or self.position_in_map_frenet is None
                or not np.isfinite(self.position_in_map_frenet).all()
                or self.speed_now is None or not math.isfinite(self.speed_now)):
            self.get_logger().warning(
                '[controller_manager] No usable local waypoints; stopping',
                throttle_duration_sec=1.0)
            return False

        if check_path_start and self.path_start_too_far_behind():
            return False
        return True

    def path_start_too_far_behind(self):
        x, y, yaw = self.position_in_map[0]
        first = self.waypoint_array_in_map[0]
        forward = ((first[0] - x) * math.cos(yaw)
                   + (first[1] - y) * math.sin(yaw))
        if forward < -self.max_path_start_rear_m:
            self.get_logger().warning(
                f'[controller_manager] Local path starts {abs(forward):.2f} m behind car; stopping',
                throttle_duration_sec=1.0)
            return True
        return False

    def front_obstacle_stop(self, now):
        """Update a continuous approach-speed cap plus an emergency stop latch."""
        if not self.front_guard_enabled:
            self.front_speed_cap = float('inf')
            return False
        self.front_speed_cap = 0.0
        ros_now_ns = self.get_clock().now().nanoseconds
        if self.require_applied_feedback and (
                self.applied_speed is None or not math.isfinite(self.applied_speed)
                or self.last_applied_time is None or self.last_applied_stamp_ns is None
                or now - self.last_applied_time > self.max_odom_age_sec
                or abs(ros_now_ns - self.last_applied_stamp_ns) * 1e-9 > self.max_odom_age_sec):
            self.get_logger().warning(
                '[controller_manager] Applied command feedback is stale; stopping',
                throttle_duration_sec=1.0)
            return True
        if (self.scan is None or self.last_scan_time is None
                or self.last_scan_stamp_ns is None
                or now - self.last_scan_time > self.max_input_age_sec
                or abs(ros_now_ns - self.last_scan_stamp_ns) * 1e-9 > self.max_input_age_sec):
            self.get_logger().warning(
                '[controller_manager] Front scan is stale; stopping',
                throttle_duration_sec=1.0)
            return True

        scan = self.scan
        front = sorted(
            distance for index, distance in enumerate(scan.ranges)
            if abs(scan.angle_min + index * scan.angle_increment) <= math.radians(12.0)
            and math.isfinite(distance)
            and scan.range_min <= distance <= scan.range_max)
        if not front:
            self.get_logger().warning(
                '[controller_manager] No valid front scan beams; stopping',
                throttle_duration_sec=1.0)
            return True

        distance = front[min(3, len(front) - 1)]
        scan_age = max(0.0, (ros_now_ns - self.last_scan_stamp_ns) * 1e-9,
                       now - self.last_scan_time)
        self.front_speed_cap = front_speed_limit(
            distance, self.front_clearance_m, self.front_ttc_stop_sec,
            scan_age, self.front_decel_mps2, self.front_speed_margin)
        self.front_ttc = time_to_collision(distance, self.speed_now, scan_age)
        if self.front_ttc < self.front_ttc_stop_sec:
            self.front_stop_latched = True
        elif self.front_ttc > self.front_ttc_release_sec:
            # Release when TTC recovers, without requiring a full physical stop.
            self.front_stop_latched = False
        if self.front_stop_latched:
            self.get_logger().warning(
                f'[controller_manager] Front obstacle {distance:.2f}m '
                f'(TTC {self.front_ttc:.2f}s, stop < {self.front_ttc_stop_sec:.2f}s); stopping',
                throttle_duration_sec=1.0)
        return self.front_stop_latched

    def control_loop(self):
        now = time.monotonic()
        if self.mode in ("MAP", "PP"):
            dt = now - self.last_control_time
            self.last_control_time = now
            if not self.inputs_ready(now, check_path_start=False):
                # Reacquisition after an input outage also starts at low speed.
                self.recovery_limiter.limit(0.0, None, dt)
                speed, steer = 0.0, self.last_safe_steer
            else:
                braking = self.front_obstacle_stop(now) or self.path_start_too_far_behind()
                x, y, yaw = self.position_in_map[0]
                points = self.waypoint_array_in_map[:, :2]
                ahead = ((points[:, 0] - x) * math.cos(yaw)
                         + (points[:, 1] - y) * math.sin(yaw))
                if not np.any(ahead > 0.05) or (braking and abs(self.speed_now) < 0.05):
                    speed, steer = 0.0, self.last_safe_steer
                elif self.mode == "MAP":
                    speed, steer = self.map_cycle(braking=braking)
                else:
                    speed, steer = self.pp_cycle()
                speed = self.recovery_limiter.limit(
                    speed, path_tracking_errors(self.position_in_map[0], points), dt)
                # Limit acceleration before the stop envelope is crossed. Without
                # this, releasing a stop simply accelerates into the same stop again.
                speed = min(speed, self.front_speed_cap)
                if braking:
                    speed = 0.0
                if math.isfinite(speed) and math.isfinite(steer):
                    self.last_safe_steer = steer
                else:
                    speed, steer = 0.0, self.last_safe_steer
        elif self.mode == "FTG":
            speed, steer = self.ftg_cycle()

        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = 'base_link'
        ack_msg.drive.steering_angle = float(steer)
        ack_msg.drive.speed = float(speed)
        self.drive_pub.publish(ack_msg)

    ############################################ MSG CREATION############################################
    # visualization utilities
    def visualize_steering(self, theta):

        quaternions = quaternion_from_euler(0, 0, theta)

        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "car_state/base_link"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 0
        lookahead_marker.id = 50
        lookahead_marker.scale.x = 0.6
        lookahead_marker.scale.y = 0.05
        lookahead_marker.scale.z = 0.01
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = 0.0
        lookahead_marker.pose.position.y = 0.0
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = quaternions[0]
        lookahead_marker.pose.orientation.y = quaternions[1]
        lookahead_marker.pose.orientation.z = quaternions[2]
        lookahead_marker.pose.orientation.w = quaternions[3]
        self.steering_pub.publish(lookahead_marker)

    def set_waypoint_markers(self, waypoints):
        wpnt_id = 0

        for waypoint in waypoints:
            waypoint_marker = self.markers_buf[wpnt_id]
            waypoint_marker.header.frame_id = "map"
            waypoint_marker.header.stamp = self.get_clock().now().to_msg()
            waypoint_marker.type = 2
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.r = 0.0
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 1.0
            waypoint_marker.color.a = 1.0
            waypoint_marker.pose.position.x = waypoint[0]
            waypoint_marker.pose.position.y = waypoint[1]
            waypoint_marker.pose.position.z = 0.0
            waypoint_marker.pose.orientation.x = 0.0
            waypoint_marker.pose.orientation.y = 0.0
            waypoint_marker.pose.orientation.z = 0.0
            waypoint_marker.pose.orientation.w = 1.0
            waypoint_marker.id = wpnt_id + 1
            wpnt_id += 1
        self.waypoint_array_buf.markers = self.markers_buf[:wpnt_id]
        self.waypoint_pub.publish(self.waypoint_array_buf)

    def set_lookahead_marker(self, lookahead_point, id):
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 2
        lookahead_marker.id = id
        lookahead_marker.scale.x = 0.15
        lookahead_marker.scale.y = 0.15
        lookahead_marker.scale.z = 0.15
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = lookahead_point[0]
        lookahead_marker.pose.position.y = lookahead_point[1]
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = 0.0
        lookahead_marker.pose.orientation.y = 0.0
        lookahead_marker.pose.orientation.z = 0.0
        lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_pub.publish(lookahead_marker)

    def visualize_trailing_opponent(self):
        if (self.state == "TRAILING" and (self.opponent is not None)):
            on = True
        else:
            on = False
        opponent_marker = Marker()
        opponent_marker.header.frame_id = "map"
        opponent_marker.header.stamp = self.get_clock().now().to_msg()
        opponent_marker.type = 2
        opponent_marker.scale.x = 0.3
        opponent_marker.scale.y = 0.3
        opponent_marker.scale.z = 0.3
        opponent_marker.color.r = 1.0
        opponent_marker.color.g = 0.0
        opponent_marker.color.b = 0.0
        opponent_marker.color.a = 1.0
        if self.opponent is not None:
            pos = self.converter.get_cartesian([self.opponent[0]], [self.opponent[1]])
            opponent_marker.pose.position.x = pos[0][0]
            opponent_marker.pose.position.y = pos[1][0]
            opponent_marker.pose.position.z = 0.0

        opponent_marker.pose.orientation.x = 0.0
        opponent_marker.pose.orientation.y = 0.0
        opponent_marker.pose.orientation.z = 0.0
        opponent_marker.pose.orientation.w = 1.0
        if not on:
            opponent_marker.action = Marker.DELETE
        self.trailing_pub.publish(opponent_marker)


def main():
    rclpy.init()
    node = Controller()
    rclpy.spin(node)
    rclpy.shutdown()
