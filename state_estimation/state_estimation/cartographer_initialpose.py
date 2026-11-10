"""Reinitialize Cartographer localization from RViz's map-frame initial pose."""

from copy import deepcopy
import math
from pathlib import Path
import time

from cartographer_ros_msgs.msg import StatusCode, TrajectoryStates
from cartographer_ros_msgs.srv import (
    FinishTrajectory, GetTrajectoryStates, StartTrajectory, TrajectoryQuery,
)
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf_transformations import quaternion_inverse, quaternion_matrix, quaternion_multiply


def normalized_pose(pose):
    """Validate finite coordinates and return a pose with a unit quaternion."""
    pose = deepcopy(pose)
    p, q = pose.position, pose.orientation
    if not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
        raise ValueError('Pose coordinates and orientation must be finite.')
    norm = math.hypot(q.x, q.y, q.z, q.w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Pose orientation must have a finite, nonzero quaternion norm.')
    q.x, q.y, q.z, q.w = (v / norm for v in (q.x, q.y, q.z, q.w))
    return pose


def pose_relative_to(reference, target):
    """Express a map-frame target relative to the frozen trajectory's first pose."""
    reference, target = normalized_pose(reference), normalized_pose(target)
    p, q = reference.position, reference.orientation
    inverse = quaternion_inverse([q.x, q.y, q.z, q.w])
    delta = [target.position.x - p.x, target.position.y - p.y, target.position.z - p.z]
    translation = quaternion_matrix(inverse)[:3, :3] @ delta
    q = target.orientation
    rotation = quaternion_multiply(inverse, [q.x, q.y, q.z, q.w])
    result = Pose()
    result.position.x, result.position.y, result.position.z = map(float, translation)
    (result.orientation.x, result.orientation.y,
     result.orientation.z, result.orientation.w) = map(float, rotation)
    return normalized_pose(result)


class TrajectoryReset:
    """Serialize asynchronous query, finish and start requests without blocking ROS."""

    def __init__(self, clients, logger, configuration_directory,
                 configuration_basename='f110_2d_loc.lua', map_frame='map',
                 reference_trajectory_id=-1, timeout=10.0, clock=time.monotonic):
        self.clients = clients
        self.logger = logger
        self.directory = str(Path(configuration_directory).expanduser().resolve())
        self.basename = configuration_basename
        self.map_frame = map_frame.lstrip('/')
        self.reference_id = reference_trajectory_id
        self.timeout = timeout
        self.clock = clock
        if not self.map_frame or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('map_frame must be nonempty and service timeout must be positive.')
        if reference_trajectory_id < -1:
            raise ValueError('reference_trajectory_id must be -1 (automatic) or nonnegative.')
        self._validate_config()
        self.pending = None
        self.busy = False
        self.blocked = False
        self.future = None
        self.finished_old = False

    def _validate_config(self):
        if Path(self.basename).name != self.basename or not self.basename.endswith('.lua'):
            raise ValueError('configuration_basename must be a Lua filename.')
        if not Path(self.directory, self.basename).is_file():
            raise ValueError(
                f'Cartographer configuration not found: {self.directory}/{self.basename}')

    def request(self, message):
        """Keep the latest valid click while an earlier reset is being processed."""
        if self.blocked:
            self.logger.error(
                'Previous service outcome is unknown; restart the bridge before retrying.')
            return
        try:
            if message.header.frame_id.lstrip('/') != self.map_frame:
                raise ValueError(
                    f'Initial pose must be in {self.map_frame}; set RViz Fixed Frame accordingly.')
            pose = normalized_pose(message.pose.pose)
        except ValueError as error:
            self.logger.error(f'Ignoring initial pose: {error}')
            return
        self.pending = pose
        self.wait_deadline = self.clock() + self.timeout
        self.logger.info('Received RViz initial pose; queued Cartographer localization reset.')

    def poll(self):
        """Advance one service response, preserving serialization on delayed replies."""
        if self.future is not None:
            if self.future.done():
                future, handler, mutating = self.future, self.handler, self.mutating
                self.future = None
                try:
                    response = future.result()
                except Exception as error:
                    self._fail(f'{self.operation} response failed: {error}', uncertain=mutating)
                    return
                if response.status.code != StatusCode.OK:
                    self._fail(f'{self.operation} rejected: {response.status.message}')
                    return
                try:
                    handler(response)
                except (ValueError, RuntimeError) as error:
                    self._fail(str(error))
            elif self.clock() >= self.deadline:
                if self.mutating:
                    # A timed-out start may still create a trajectory. Never submit it twice.
                    self.logger.error(
                        f'{self.operation} response timed out; waiting for the same request. '
                        'No additional trajectory will be started until its outcome is known.')
                    self.deadline = math.inf
                else:
                    self.clients[self.operation].remove_pending_request(self.future)
                    self.future.cancel()
                    self.future = None
                    self._fail(
                        f'{self.operation} timed out; draw the initial pose again to retry.')
            return
        if self.busy or self.pending is None or self.blocked:
            return
        unavailable = [name for name, client in self.clients.items()
                       if not client.service_is_ready()]
        if unavailable:
            if self.clock() >= self.wait_deadline:
                self.pending = None
                self.logger.error('Cartographer services unavailable: ' + ', '.join(unavailable))
            return
        try:
            self._validate_config()
        except ValueError as error:
            self.pending = None
            self.logger.error(str(error))
            return
        self.target, self.pending = self.pending, None
        self.busy = True
        self.finished_old = False
        self._call('get_trajectory_states', GetTrajectoryStates.Request(), self._states)

    def _call(self, name, request, handler):
        self.operation, self.handler = name, handler
        self.mutating = name in ('finish_trajectory', 'start_trajectory')
        self.deadline = self.clock() + self.timeout
        if not self.clients[name].service_is_ready():
            self._fail(f'{name} is unavailable; draw the initial pose again to retry.')
            return
        try:
            self.future = self.clients[name].call_async(request)
        except Exception as error:
            self._fail(f'{name} request failed: {error}', uncertain=self.mutating)

    def _fail(self, message, uncertain=False):
        self.busy = False
        self.blocked = uncertain
        if uncertain:
            self.pending = None
            message += (' Outcome unknown; restart the bridge after checking '
                        'Cartographer trajectory states.')
        elif self.finished_old:
            message += (' Previous localization trajectory was finished; '
                        'draw the initial pose again to restart.')
        self.logger.error(message)

    def _states(self, response):
        states = response.trajectory_states
        if len(states.trajectory_id) != len(states.trajectory_state):
            raise ValueError('Cartographer returned inconsistent trajectory states.')
        frozen = sorted(t for t, s in zip(states.trajectory_id, states.trajectory_state)
                        if s == TrajectoryStates.FROZEN)
        active = [t for t, s in zip(states.trajectory_id, states.trajectory_state)
                  if s == TrajectoryStates.ACTIVE]
        if not frozen:
            raise ValueError(
                'No frozen map trajectory loaded; initial pose reset is for localization only.')
        if len(active) > 1:
            raise ValueError(
                'Multiple active trajectories; refusing to finish an ambiguous localization '
                'trajectory.')
        self.reference = frozen[0] if self.reference_id == -1 else self.reference_id
        if self.reference not in frozen:
            raise ValueError(
                f'Reference trajectory {self.reference} is not a frozen map trajectory.')
        self.active = active[0] if active else None
        request = TrajectoryQuery.Request()
        request.trajectory_id = self.reference
        self._call('trajectory_query', request, self._reference_pose)

    def _reference_pose(self, response):
        if not response.trajectory:
            raise ValueError(
                f'Frozen trajectory {self.reference} has no poses; '
                'cannot initialize relative to it.')
        first = min(response.trajectory,
                    key=lambda p: (p.header.stamp.sec, p.header.stamp.nanosec))
        if first.header.frame_id.lstrip('/') != self.map_frame:
            raise ValueError(
                'Frozen trajectory pose is not expressed in the configured map frame.')
        # StartTrajectory anchors to the reference trajectory at time zero, clamped to
        # its first node: T_map_target = T_map_reference_first * T_reference_target.
        self.start_request = StartTrajectory.Request()
        self.start_request.configuration_directory = self.directory
        self.start_request.configuration_basename = self.basename
        self.start_request.use_initial_pose = True
        self.start_request.relative_to_trajectory_id = self.reference
        self.start_request.initial_pose = pose_relative_to(first.pose, self.target)
        if self.active is None:
            self._start()
        else:
            request = FinishTrajectory.Request()
            request.trajectory_id = self.active
            self._call('finish_trajectory', request, self._finished)

    def _finished(self, _response):
        self.finished_old = True
        self._start()

    def _start(self):
        self._call('start_trajectory', self.start_request, self._started)

    def _started(self, response):
        self.busy = False
        self.logger.info(
            f'Localization restarted as trajectory {response.trajectory_id}, relative to frozen '
            f'trajectory {self.reference}; initial map position '
            f'({self.target.position.x:.3f}, {self.target.position.y:.3f}). '
            'Check laser/map alignment as localization converges.')


class CartographerInitialPose(Node):
    """Connect RViz initial poses to Cartographer's asynchronous service interface."""

    def __init__(self, **kwargs):
        super().__init__('cartographer_initialpose', **kwargs)
        defaults = {'configuration_directory': '', 'configuration_basename': 'f110_2d_loc.lua',
                    'map_frame': 'map', 'reference_trajectory_id': -1,
                    'service_timeout_sec': 10.0, 'initialpose_topic': '/initialpose'}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        clients = {name: self.create_client(service, name) for name, service in (
            ('get_trajectory_states', GetTrajectoryStates), ('trajectory_query', TrajectoryQuery),
            ('finish_trajectory', FinishTrajectory), ('start_trajectory', StartTrajectory))}
        self.reset = TrajectoryReset(
            clients, self.get_logger(), self.get_parameter('configuration_directory').value,
            self.get_parameter('configuration_basename').value,
            self.get_parameter('map_frame').value,
            self.get_parameter('reference_trajectory_id').value,
            self.get_parameter('service_timeout_sec').value)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped, self.get_parameter('initialpose_topic').value,
            self.reset.request, qos)
        self.timer = self.create_timer(
            0.1, self.reset.poll, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('Ready for RViz 2D Pose Estimate in the map frame.')


def main(args=None):
    """Run the bridge without blocking the executor on service responses."""
    rclpy.init(args=args)
    node = None
    try:
        node = CartographerInitialPose()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
