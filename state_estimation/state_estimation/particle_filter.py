# MIT License

# Copyright (c) 2020 Hongrui Zheng, Corey Walsh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ros2 python
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

# libraries
import numpy as np
import range_libc
import time
from collections import deque
from threading import Lock
from state_estimation import particle_filter_utils as Utils
from state_estimation.particle_filter_gpu import require_cuda

# TF
# import tf.transformations
# import tf
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
import tf_transformations

# messages
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

'''
These flags indicate several variants of the sensor model. Only one of them is used at a time.
'''
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFiler(Node):
    '''
    This class implements Monte Carlo Localization based on odometry and a laser scanner.
    '''

    def __init__(self, node_name='particle_filter'):
        super().__init__(node_name)

        # declare parameters
        self.declare_parameter('angle_step')
        self.declare_parameter('max_particles')
        self.declare_parameter('max_viz_particles')
        self.declare_parameter('squash_factor')
        self.declare_parameter('max_range')
        self.declare_parameter('theta_discretization')
        self.declare_parameter('range_method')
        self.declare_parameter('rangelib_variant')
        self.declare_parameter('fine_timing')
        self.declare_parameter('viz')
        self.declare_parameter('visualization_update_period', 0.2)
        self.declare_parameter('z_short')
        self.declare_parameter('z_max')
        self.declare_parameter('z_rand')
        self.declare_parameter('z_hit')
        self.declare_parameter('sigma_hit')
        self.declare_parameter('motion_dispersion_x')
        self.declare_parameter('motion_dispersion_y')
        self.declare_parameter('motion_dispersion_theta')
        self.declare_parameter('scan_topic')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('initial_var_x', 0.5)
        self.declare_parameter('initial_var_y', 0.5)
        self.declare_parameter('initial_var_theta', 0.4)
        self.declare_parameter(
            'laser_global_pose_topic', '/pf/laser_global_pose')

        # parameters
        self.ANGLE_STEP           = self.get_parameter('angle_step').value
        self.MAX_PARTICLES        = self.get_parameter('max_particles').value
        self.MAX_VIZ_PARTICLES    = self.get_parameter('max_viz_particles').value
        self.INV_SQUASH_FACTOR    = 1.0 / self.get_parameter('squash_factor').value
        self.MAX_RANGE_METERS     = self.get_parameter('max_range').value
        self.THETA_DISCRETIZATION = self.get_parameter('theta_discretization').value
        self.WHICH_RM             = self.get_parameter('range_method').value
        self.RANGELIB_VAR         = self.get_parameter('rangelib_variant').value
        if self.WHICH_RM != 'rmgpu' or self.RANGELIB_VAR != 2:
            raise ValueError('PF requires range_method=rmgpu and rangelib_variant=2; CPU fallback is disabled')
        gpu_count = require_cuda()
        self.get_logger().info(f'PF CUDA enabled: {gpu_count} GPU(s), RangeLibc rmgpu variant 2')
        self.SHOW_FINE_TIMING     = self.get_parameter('fine_timing').value
        self.DO_VIZ               = self.get_parameter('viz').value
        self.VIZ_UPDATE_PERIOD    = self.get_parameter(
            'visualization_update_period').value
        if self.VIZ_UPDATE_PERIOD <= 0.0:
            raise ValueError('visualization_update_period must be > 0')

        # sensor model constants
        self.Z_SHORT   = self.get_parameter('z_short').value
        self.Z_MAX     = self.get_parameter('z_max').value
        self.Z_RAND    = self.get_parameter('z_rand').value
        self.Z_HIT     = self.get_parameter('z_hit').value
        self.SIGMA_HIT = self.get_parameter('sigma_hit').value

        # motion model constants
        self.MOTION_DISPERSION_X     = self.get_parameter('motion_dispersion_x').value
        self.MOTION_DISPERSION_Y     = self.get_parameter('motion_dispersion_y').value
        self.MOTION_DISPERSION_THETA = self.get_parameter('motion_dispersion_theta').value
        
        # various data containers used in the MCL algorithm
        self.MAX_RANGE_PX = None
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.laser = None
        self.iters = 0
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.last_pose = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.range_method = None
        self.last_time = None
        self.last_visualization_time = None
        self.last_stamp = None
        self.first_sensor_update = True
        self.odom_to_base_matrix = None
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.laser_frame = 'laser'
        self.particles_at_laser = True
        self.base_to_laser_matrix = None
        self.scan_queue = deque(maxlen=10)
        self.pending_initial_pose = None
        self.state_lock = Lock()

        # cache this to avoid memory allocation in motion model
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3))

        # cache this for the sensor model computation
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None

        # particle poses and weights
        self.inferred_pose = None
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        self.configure_backend()

        # initialize the state
        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)
        # map service client
        self.map_client = self.create_client(GetMap, '/map_server/map')
        self.get_omap()
        self.precompute_sensor_model()
        self.initialize_global()

        # Kept only for the observation topic's existing twist field.
        self.current_speed = 0.0

        # Pub Subs
        # these topics are for visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/viz/inferred_pose', 1)
        self.particle_pub = self.create_publisher(PoseArray, '/pf/viz/particles', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, '/pf/viz/fake_scan', 1)
        self.rect_pub = self.create_publisher(PolygonStamped, '/pf/viz/poly1', 1)

        # This Odometry message reports the laser frame pose in the map frame.
        self.laser_global_pose_pub = self.create_publisher(
            Odometry,
            self.get_parameter('laser_global_pose_topic').value,
            1)

        # these topics are for coordinate space things
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_tf = TransformBroadcaster(self)
        self.tf_lookup_warning_emitted = False

        # these topics are to receive data from the racecar
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.lidarCB,
            qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odometry_topic').value,
            self.odomCB,
            qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.clicked_pose,
            qos_profile_sensor_data)
        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_pose,
            qos_profile_sensor_data)

        # Retry scans when matching odometry TF arrives. A bounded queue avoids
        # starvation when TF consistently arrives after the next LiDAR callback.
        self.create_timer(0.01, self.process_scan)
        self.get_logger().info('Finished initializing, waiting on messages...')

    def configure_backend(self):
        """Configure backend-specific models before map initialization."""

    def scan_indices(self, msg):
        return np.arange(0, len(msg.ranges), self.ANGLE_STEP)

    def laser_particles(self, particles):
        """Express particle hypotheses at the laser for ray casting/observations."""
        return particles

    def weight_proposals(self, particles, weights):
        return weights

    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''

        while rclpy.ok() and not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get map service not available, waiting...')
        while rclpy.ok():
            future = self.map_client.call_async(GetMap.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.done() and future.result() is not None:
                map_msg = future.result().map
                if map_msg.info.width and map_msg.info.height:
                    break
            else:
                future.cancel()
            # The map service can exist before the lifecycle server is active.
            rclpy.spin_once(self, timeout_sec=0.2)
        else:
            raise RuntimeError('Shutdown while waiting for a non-empty map')
        self.map_info = map_msg.info

        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        self.get_logger().info('Initializing range method: ' + self.WHICH_RM)
        if self.WHICH_RM == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
        elif 'cddt' in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
            if self.WHICH_RM == 'pcddt':
                self.get_logger().info('Pruning...')
                self.range_method.prune()
        elif self.WHICH_RM == 'rm':
            self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'glt':
            self.range_method = range_libc.PyGiantLUTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        else:
            raise ValueError('Unsupported range_method: ' + self.WHICH_RM)
        self.get_logger().info('Done loading map')

         # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255==0] = 1
        self.map_initialized = True

    def publish_tf(self, pose, stamp=None):
        """Publish map -> odom so the inferred laser pose is represented by the TF tree."""
        laser_pose = self.laser_particles(np.asarray(pose)[None, :])[0]
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        try:
            base_laser = self.tf_buffer.lookup_transform(
                self.base_frame, self.laser_frame, Time())

            map_laser_matrix = tf_transformations.concatenate_matrices(
                tf_transformations.translation_matrix([laser_pose[0], laser_pose[1], 0.0]),
                tf_transformations.quaternion_matrix(
                    tf_transformations.quaternion_from_euler(0.0, 0.0, laser_pose[2])))
            # The PF estimates x/y/yaw only. Keep the base on the ground,
            # instead of lowering it by the laser's mounting height.
            map_laser_matrix[2, 3] = base_laser.transform.translation.z
            map_odom_matrix = Utils.map_to_odom_matrix(
                map_laser_matrix,
                self.odom_to_base_matrix,
                base_laser.transform)

            translation = tf_transformations.translation_from_matrix(map_odom_matrix)
            rotation = tf_transformations.quaternion_from_matrix(map_odom_matrix)

            t = TransformStamped()
            # This map-to-odom correction composes with live odometry. Stamp
            # the TF when published so newer scans need not wait for the next
            # slow MCL update; keep the pose observation at its scan stamp.
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = self.odom_frame
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]
            t.transform.rotation.x = rotation[0]
            t.transform.rotation.y = rotation[1]
            t.transform.rotation.z = rotation[2]
            t.transform.rotation.w = rotation[3]
            self.pub_tf.sendTransform(t)
            self.tf_lookup_warning_emitted = False
        except TransformException as ex:
            if not self.tf_lookup_warning_emitted:
                self.get_logger().warning(
                    'Cannot publish map -> odom: base -> laser transform unavailable: '
                    + str(ex))
                self.tf_lookup_warning_emitted = True

        # Preserve the former pose observation payload under an explicit name.
        # pose.pose is T_map_laser; this is not wheel odometry or T_map_odom.
        laser_global_pose = Odometry()
        laser_global_pose.header.stamp = stamp
        laser_global_pose.header.frame_id = 'map'
        laser_global_pose.child_frame_id = self.laser_frame
        laser_global_pose.pose.pose.position.x = laser_pose[0]
        laser_global_pose.pose.pose.position.y = laser_pose[1]
        laser_global_pose.pose.pose.orientation = Utils.angle_to_quaternion(laser_pose[2])
        pose_covariance = Utils.weighted_pose_covariance(
            self.laser_particles(self.particles), self.weights, laser_pose)
        laser_global_pose.pose.covariance = Utils.planar_covariance_to_ros(
            pose_covariance)
        laser_global_pose.twist.twist.linear.x = self.current_speed
        self.laser_global_pose_pub.publish(laser_global_pose)

        return

    def visualize(self):
        '''
        Publish various visualization messages.
        '''
        if not self.DO_VIZ:
            return

        now = time.monotonic()
        if (self.last_visualization_time is not None and
                now - self.last_visualization_time < self.VIZ_UPDATE_PERIOD):
            return
        self.last_visualization_time = now

        if self.pose_pub.get_subscription_count() > 0 and isinstance(self.inferred_pose, np.ndarray):
            # Publish the inferred pose for visualization
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = 'map'
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            self.pose_pub.publish(ps)

        if self.particle_pub.get_subscription_count() > 0:
            # publish a downsampled version of the particle distribution to avoid a lot of latency
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                # randomly downsample particles
                proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
                # proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES)
                self.publish_particles(self.particles[proposal_indices,:])
            else:
                self.publish_particles(self.particles)

        if self.pub_fake_scan.get_subscription_count() > 0 and isinstance(self.ranges, np.ndarray):
            # generate the scan from the point of view of the inferred position for visualization
            laser_pose = self.laser_particles(self.inferred_pose[None, :])[0]
            # LaserScan represents uniformly spaced angles. SynPF's boxed
            # observation beams are nonuniform, so ray-cast a separate uniform
            # set for visualization without changing the localization beams.
            viz_angles = np.linspace(
                np.min(self.downsampled_angles), np.max(self.downsampled_angles),
                len(self.downsampled_angles), dtype=np.float32)
            self.viz_queries[:,0] = laser_pose[0]
            self.viz_queries[:,1] = laser_pose[1]
            self.viz_queries[:,2] = viz_angles + laser_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self.publish_scan(viz_angles, self.viz_ranges)

    def publish_particles(self, particles):
        # publish the given particles as a PoseArray object
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def publish_scan(self, angles, ranges):
        # publish the given angels and ranges as a laser scan message
        ls = LaserScan()
        ls.header.stamp = self.last_stamp
        ls.header.frame_id = self.laser_frame
        ls.angle_min = float(np.min(angles))
        ls.angle_max = float(np.max(angles))
        ls.angle_increment = float(np.abs(angles[0] - angles[1]))
        ls.range_min = 0.0
        ls.range_max = float(np.max(ranges))
        ls.ranges = np.asarray(ranges, dtype=float).tolist()
        self.pub_fake_scan.publish(ls)

    def lidarCB(self, msg):
        self.scan_queue.append(msg)

    def odomCB(self, msg):
        # Geometry is read from TF at the scan timestamp, not at callback time.
        self.current_speed = msg.twist.twist.linear.x

    def process_scan(self):
        if not self.scan_queue:
            return
        # Keep localization near real time when scans arrive faster than MCL
        # updates. Motion is integrated between the processed scan timestamps.
        last_error = None
        selected = None
        for msg in reversed(self.scan_queue):
            stamp = Time.from_msg(msg.header.stamp)
            if self.last_stamp is not None and stamp <= Time.from_msg(self.last_stamp):
                break
            laser_frame = msg.header.frame_id.lstrip('/')
            try:
                odom_base = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, stamp)
                base_laser = self.tf_buffer.lookup_transform(
                    self.base_frame, laser_frame, stamp)
            except TransformException as error:
                last_error = error
                continue
            selected = (msg, laser_frame, odom_base, base_laser)
            break
        if selected is None:
            if last_error is not None:
                self.get_logger().warning(
                    f'Waiting for odometry/laser TF at scan time: {last_error}',
                    throttle_duration_sec=2.0)
            else:
                self.scan_queue.clear()
            return
        msg, laser_frame, odom_base, base_laser = selected

        if self.pending_initial_pose is not None:
            map_base = Utils.pose_to_matrix(self.pending_initial_pose)
            map_laser = (map_base @ Utils.transform_to_matrix(base_laser.transform)
                         if self.particles_at_laser else map_base)
            pose = Pose()
            pose.position.x = float(map_laser[0, 3])
            pose.position.y = float(map_laser[1, 3])
            pose.orientation = Utils.angle_to_quaternion(
                tf_transformations.euler_from_matrix(map_laser)[2])
            self.initialize_particles_pose(pose)
            self.pending_initial_pose = None
            self.last_pose = None

        # Discard the selected scan and older ones; keep any newer scans that
        # are still waiting for their odometry TF.
        while self.scan_queue:
            if self.scan_queue.popleft() is msg:
                break
        self.laser_frame = laser_frame
        self.odom_to_base_matrix = Utils.transform_to_matrix(odom_base.transform)
        self.base_to_laser_matrix = Utils.transform_to_matrix(base_laser.transform)
        odom_laser = (self.odom_to_base_matrix @ self.base_to_laser_matrix
                      if self.particles_at_laser else self.odom_to_base_matrix)
        pose = np.array([odom_laser[0, 3], odom_laser[1, 3],
                         tf_transformations.euler_from_matrix(odom_laser)[2]])
        # Compute motion at the point represented by this backend's particles.
        if self.last_pose is None:
            self.odometry_data = np.zeros(3)
        else:
            delta = pose[:2] - self.last_pose[:2]
            yaw = self.last_pose[2]
            self.odometry_data = np.array([
                np.cos(yaw) * delta[0] + np.sin(yaw) * delta[1],
                -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1],
                Utils.angle_difference(pose[2], yaw)])
        self.last_pose = pose
        self.last_stamp = msg.header.stamp

        # Laser geometry can change when replaying bags from another sensor.
        angles = (msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment)
        indices = self.scan_indices(msg)
        downsampled = angles[indices].astype(np.float32)
        if len(downsampled) < 2:
            return
        if self.downsampled_angles is None or not np.array_equal(
                downsampled, self.downsampled_angles):
            self.laser_angles = angles
            self.downsampled_angles = downsampled
            self.viz_queries = np.zeros((len(downsampled), 3), dtype=np.float32)
            self.viz_ranges = np.zeros(len(downsampled), dtype=np.float32)
            self.first_sensor_update = True
        ranges = np.asarray(msg.ranges, dtype=np.float32)[indices]
        self.downsampled_ranges = np.nan_to_num(
            ranges, nan=self.MAX_RANGE_METERS, posinf=self.MAX_RANGE_METERS,
            neginf=0.0).clip(0.0, self.MAX_RANGE_METERS)
        self.lidar_initialized = True
        self.odom_initialized = True
        self.update()

    def clicked_pose(self, msg):
        if msg.header.frame_id.lstrip('/') != 'map':
            self.get_logger().warning('Initial pose must be expressed in the map frame.')
            return
        if isinstance(msg, PointStamped):
            self.pending_initial_pose = None
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            pose = msg.pose.pose
            values = [pose.position.x, pose.position.y, pose.orientation.x,
                      pose.orientation.y, pose.orientation.z, pose.orientation.w]
            if not np.all(np.isfinite(values)) or np.linalg.norm(values[2:]) < 1e-6:
                self.get_logger().warning('Ignoring an invalid initial pose.')
                return
            # RViz specifies the base pose; convert to a laser pose once TF exists.
            self.pending_initial_pose = pose

    def initialize_particles_pose(self, pose):
        '''
        Initialize particles in the general region of the provided pose.
        '''
        self.get_logger().info('SETTING POSE')
        self.get_logger().info(str([pose.position.x, pose.position.y]))
        self.state_lock.acquire()
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=self.get_parameter('initial_var_x').value,size=self.MAX_PARTICLES)
        self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=self.get_parameter('initial_var_y').value,size=self.MAX_PARTICLES)
        yaw = Utils.quaternion_to_angle(pose.orientation)
        self.particles[:, 2] = Utils.wrap_angle(
            yaw + np.random.normal(
                loc=0.0, scale=self.get_parameter('initial_var_theta').value, size=self.MAX_PARTICLES))
        self.state_lock.release()

    def initialize_global(self):
        '''
        Spread the particle distribution over the permissible region of the state space.
        '''
        self.get_logger().info('GLOBAL INITIALIZATION')
        # randomize over grid coordinate space
        self.state_lock.acquire()
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

        permissible_states = np.zeros((self.MAX_PARTICLES,3))
        permissible_states[:,0] = permissible_y[indices]
        permissible_states[:,1] = permissible_x[indices]
        permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.MAX_PARTICLES
        self.state_lock.release()

    def precompute_sensor_model(self):
        '''
        Generate and store a table which represents the sensor model. For each discrete computed
        range value, this provides the probability of measuring any (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        '''
        self.get_logger().info('Precomputing sensor model')
        # sensor model constants
        z_short = self.Z_SHORT
        z_max   = self.Z_MAX
        z_rand  = self.Z_RAND
        z_hit   = self.Z_HIT
        sigma_hit = self.SIGMA_HIT
        
        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width,table_width))

        t = time.time()
        # d is the computed range from RangeLibc
        for d in range(table_width):
            norm = 0.0
            sum_unkown = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(r-d)
                # reflects from the intended object
                prob += z_hit * np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / (sigma_hit * np.sqrt(2.0*np.pi))

                # observed range is less than the predicted range - short reading
                if r < d:
                    prob += 2.0 * z_short * (d - r) / float(d)

                # erroneous max range measurement
                if int(r) == int(self.MAX_RANGE_PX):
                    prob += z_max

                # random measurement
                if r < int(self.MAX_RANGE_PX):
                    prob += z_rand * 1.0/float(self.MAX_RANGE_PX)

                norm += prob
                self.sensor_model_table[int(r),int(d)] = prob

            # normalize
            self.sensor_model_table[:,int(d)] /= norm

        # upload the sensor model to RangeLib for ultra fast resolution
        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, proposal_dist, action):
        '''
        The motion model applies the odometry to the particle distribution. Since there the odometry
        data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.

        Vectorized motion model. Computing the motion model over all particles is thousands of times
        faster than doing it for each particle individually due to vectorization and reduction in
        function call overhead
        
        TODO this could be better, but it works for now
            - fixed random noise is not very realistic
            - ackermann model provides bad estimates at high speed
        '''
        # rotate the action into the coordinate space of each particle
        # t1 = time.time()
        cosines = np.cos(proposal_dist[:,2])
        sines = np.sin(proposal_dist[:,2])

        self.local_deltas[:,0] = cosines*action[0] - sines*action[1]
        self.local_deltas[:,1] = sines*action[0] + cosines*action[1]
        self.local_deltas[:,2] = action[2]

        proposal_dist[:,:] += self.local_deltas
        proposal_dist[:,0] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_X,size=self.MAX_PARTICLES)
        proposal_dist[:,1] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_Y,size=self.MAX_PARTICLES)
        proposal_dist[:,2] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_THETA,size=self.MAX_PARTICLES)
        proposal_dist[:, 2] = Utils.wrap_angle(proposal_dist[:, 2])

    def sensor_model(self, proposal_dist, obs, weights):
        '''
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        '''
        
        num_rays = self.downsampled_angles.shape[0]
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros((num_rays*self.MAX_PARTICLES,3), dtype=np.float32)
            else:
                self.queries = np.zeros((self.MAX_PARTICLES,3), dtype=np.float32)

            self.ranges = np.zeros(num_rays*self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
            self.first_sensor_update = False

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if 'cddt' in self.WHICH_RM:
                self.queries[:,:] = proposal_dist[:,:]
                self.range_method.calc_range_many_radial_optimized(num_rays, self.downsampled_angles[0], self.downsampled_angles[-1], self.queries, self.ranges)

                # evaluate the sensor model
                self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
                # apply the squash factor
                self.weights = np.power(self.weights, self.INV_SQUASH_FACTOR)
            else:
                self.get_logger().info('Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2')
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            self.queries[:,:] = proposal_dist[:,:]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(self.queries, self.downsampled_angles, obs, self.weights)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            self.queries[:,:] = proposal_dist[:,:]
            if self.SHOW_FINE_TIMING:
                t_init = time.time()
            self.range_method.calc_range_repeat_angles(self.queries, self.downsampled_angles, self.ranges)
            if self.SHOW_FINE_TIMING:
                t_range = time.time()
            # Evaluate sensor likelihoods in native CPU code; CUDA handles ray casting.
            self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            if self.SHOW_FINE_TIMING:
                t_eval = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(str(['sensor_model: init: ', np.round((t_init-t_start)/t_total, 2), 'range:', np.round((t_range-t_init)/t_total, 2), \
                      'eval:', np.round((t_eval-t_range)/t_total, 2), 'squash:', np.round((t_squash-t_eval)/t_total, 2)]))
        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:,0] = np.repeat(proposal_dist[:,0], num_rays)
            self.queries[:,1] = np.repeat(proposal_dist[:,1], num_rays)
            self.queries[:,2] = np.repeat(proposal_dist[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # Evaluate sensor likelihoods in native CPU code; CUDA handles ray casting.
            self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:,0] = np.repeat(proposal_dist[:,0], num_rays)
            self.queries[:,1] = np.repeat(proposal_dist[:,1], num_rays)
            self.queries[:,2] = np.repeat(proposal_dist[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges[ranges > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(self.sensor_model_table[intobs,intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            self.get_logger().info('PLEASE SET rangelib_variant PARAM to 0-4')

    def MCL(self, a, o):
        '''
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights

        This is in the critical path of code execution, so it is optimized for speed.
        '''
        if self.SHOW_FINE_TIMING:
            t = time.time()
        # draw the proposal distribution from the old particles
        proposal_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
        proposal_distribution = self.particles[proposal_indices,:]
        if self.SHOW_FINE_TIMING:
            t_propose = time.time()

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, a)
        if self.SHOW_FINE_TIMING:
            t_motion = time.time()

        # compute the sensor model
        self.sensor_model(proposal_distribution, o, self.weights)
        if self.SHOW_FINE_TIMING:
            t_sensor = time.time()

        # normalize importance weights
        self.weights = self.weight_proposals(proposal_distribution, self.weights)
        total = np.sum(self.weights)
        if np.isfinite(total) and total > 0.0 and np.all(np.isfinite(self.weights)):
            self.weights /= total
        else:
            self.get_logger().warning('Sensor weights collapsed; retaining a uniform proposal.')
            self.weights.fill(1.0 / self.MAX_PARTICLES)
        if self.SHOW_FINE_TIMING:
            t_norm = time.time()
            t_total = (t_norm - t)/100.0

        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            self.get_logger().info(str(['MCL: propose: ', np.round((t_propose-t)/t_total, 2), 'motion:', np.round((t_motion-t_propose)/t_total, 2), \
                  'sensor:', np.round((t_sensor-t_motion)/t_total, 2), 'norm:', np.round((t_norm-t_sensor)/t_total, 2)]))

        # save the particles
        self.particles = proposal_distribution
    
    def expected_pose(self):
        # returns the expected value of the pose given the particle distribution
        return Utils.weighted_pose_mean(self.particles, self.weights)

    def update(self):
        '''
        Apply the MCL function to update particle filter state. 

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            if self.state_lock.locked():
                self.get_logger().info('Concurrency error avoided')
            else:
                self.state_lock.acquire()
                self.timer.tick()
                self.iters += 1

                t1 = time.time()
                observation = np.copy(self.downsampled_ranges).astype(np.float32)
                action = np.copy(self.odometry_data)
                self.odometry_data = np.zeros(3)

                # run the MCL update algorithm
                self.MCL(action, observation)

                # compute the expected value of the robot pose
                self.inferred_pose = self.expected_pose()
                self.state_lock.release()
                t2 = time.time()

                # publish transformation frame based on inferred pose
                self.publish_tf(self.inferred_pose, self.last_stamp)

                # this is for tracking particle filter speed
                ips = 1.0 / (t2 - t1)
                self.smoothing.append(ips)
                if self.iters % 10 == 0:
                    self.get_logger().info(str(['iters per sec:', int(self.timer.fps()), ' possible:', int(self.smoothing.mean())]))

                self.visualize()

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    try:
        rclpy.spin(pf)
    finally:
        pf.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

