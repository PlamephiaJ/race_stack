"""Check coordinate conversion and reset transactions without a live vehicle."""

from concurrent.futures import Future
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

from cartographer_ros_msgs.msg import StatusCode, TrajectoryStates
from cartographer_ros_msgs.srv import (
    FinishTrajectory, GetTrajectoryStates, StartTrajectory, TrajectoryQuery,
)
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
import pytest

from state_estimation.cartographer_initialpose import (
    normalized_pose, pose_relative_to, TrajectoryReset,
)


def pose(x=0.0, y=0.0, yaw=0.0):
    result = Pose()
    result.position.x, result.position.y = float(x), float(y)
    result.orientation.z = math.sin(yaw / 2)
    result.orientation.w = math.cos(yaw / 2)
    return result


def click(x=0.0, y=0.0, yaw=0.0, frame='map'):
    message = PoseWithCovarianceStamped()
    message.header.frame_id = frame
    message.pose.pose = pose(x, y, yaw)
    return message


def states(active=(42,), frozen=(7,)):
    response = GetTrajectoryStates.Response()
    response.trajectory_states.trajectory_id = list(frozen) + list(active)
    response.trajectory_states.trajectory_state = (
        [TrajectoryStates.FROZEN] * len(frozen) + [TrajectoryStates.ACTIVE] * len(active))
    return response


def query(frame='map', empty=False):
    response = TrajectoryQuery.Response()
    if not empty:
        first = PoseStamped()
        first.header.frame_id = frame
        first.header.stamp.sec = 20
        first.pose = pose(10.0, -3.0, math.pi / 2)
        later = deepcopy(first)
        later.header.stamp.sec = 30
        later.pose.position.x = 100.0
        response.trajectory = [later, first]
    return response


class FakeClient:
    """Delay service responses until a test explicitly completes the request."""

    def __init__(self):
        self.ready = True
        self.calls = []
        self.removed = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.calls.append(deepcopy(request))
        self.future = Future()
        return self.future

    def remove_pending_request(self, future):
        self.removed.append(future)


@pytest.fixture
def rig(tmp_path):
    directory = tmp_path / 'BLUE/slam'
    directory.mkdir(parents=True)
    (directory / 'f110_2d_loc.lua').write_text('return {}')
    clients = {name: FakeClient() for name in (
        'get_trajectory_states', 'trajectory_query', 'finish_trajectory', 'start_trajectory')}
    messages, now = [], [0.0]
    logger = SimpleNamespace(info=lambda msg: messages.append(('info', msg)),
                             error=lambda msg: messages.append(('error', msg)))
    reset = TrajectoryReset(clients, logger, directory, clock=lambda: now[0])
    return SimpleNamespace(reset=reset, clients=clients, messages=messages, now=now)


def reply(rig, name, response):
    assert rig.reset.operation == name
    rig.clients[name].future.set_result(response)
    rig.reset.poll()


def preflight(rig, active=(42,)):
    rig.reset.request(click(8.0, 1.0, math.pi))
    rig.reset.poll()
    reply(rig, 'get_trajectory_states', states(active=active))
    reply(rig, 'trajectory_query', query())


def test_map_pose_converted_relative_to_rotated_translated_reference():
    target = pose(8.0, 1.0, math.pi)
    relative = pose_relative_to(pose(10.0, -3.0, math.pi / 2), target)
    assert relative.position.x == pytest.approx(4.0)
    assert relative.position.y == pytest.approx(2.0)
    assert relative.orientation.z == pytest.approx(math.sqrt(0.5))
    assert relative.orientation.w == pytest.approx(math.sqrt(0.5))
    assert target.position.x == 8.0


def test_reset_queries_map_before_finishing_only_live_trajectory(rig):
    preflight(rig)
    assert rig.clients['trajectory_query'].calls[0].trajectory_id == 7
    assert rig.clients['finish_trajectory'].calls[0].trajectory_id == 42
    assert not rig.clients['start_trajectory'].calls
    reply(rig, 'finish_trajectory', FinishTrajectory.Response())
    request = rig.clients['start_trajectory'].calls[0]
    assert request.use_initial_pose
    assert request.relative_to_trajectory_id == 7
    assert request.configuration_directory.endswith('BLUE/slam')
    assert request.configuration_basename == 'f110_2d_loc.lua'
    assert request.initial_pose.position.x == pytest.approx(4.0)
    assert request.initial_pose.position.y == pytest.approx(2.0)
    reply(rig, 'start_trajectory', StartTrajectory.Response(trajectory_id=43))
    assert not rig.reset.busy
    assert 'trajectory 43' in rig.messages[-1][1]


@pytest.mark.parametrize('response', [states(frozen=()), states(active=(42, 43))])
def test_unsafe_trajectory_states_never_finish_a_trajectory(rig, response):
    rig.reset.request(click())
    rig.reset.poll()
    reply(rig, 'get_trajectory_states', response)
    assert not rig.reset.busy
    assert not rig.clients['finish_trajectory'].calls
    assert not rig.clients['start_trajectory'].calls


@pytest.mark.parametrize('response', [query(empty=True), query(frame='odom')])
def test_invalid_reference_query_preserves_current_localization(rig, response):
    rig.reset.request(click())
    rig.reset.poll()
    reply(rig, 'get_trajectory_states', states())
    reply(rig, 'trajectory_query', response)
    assert not rig.clients['finish_trajectory'].calls
    assert not rig.reset.busy


def test_explicit_reference_must_be_frozen(rig):
    rig.reset.reference_id = 42
    rig.reset.request(click())
    rig.reset.poll()
    reply(rig, 'get_trajectory_states', states())
    assert not rig.clients['trajectory_query'].calls
    assert not rig.clients['finish_trajectory'].calls


def test_recovery_when_previous_localization_was_already_finished(rig):
    preflight(rig, active=())
    assert not rig.clients['finish_trajectory'].calls
    assert len(rig.clients['start_trajectory'].calls) == 1


def test_finish_failure_does_not_start_another_trajectory(rig):
    preflight(rig)
    response = FinishTrajectory.Response()
    response.status.code = StatusCode.INVALID_ARGUMENT
    response.status.message = 'not active'
    reply(rig, 'finish_trajectory', response)
    assert not rig.clients['start_trajectory'].calls
    assert not rig.reset.busy


def test_start_rejection_can_be_retried_without_finishing_frozen_map(rig):
    preflight(rig)
    reply(rig, 'finish_trajectory', FinishTrajectory.Response())
    response = StartTrajectory.Response()
    response.status.code = StatusCode.UNAVAILABLE
    reply(rig, 'start_trajectory', response)
    assert 'Previous localization trajectory was finished' in rig.messages[-1][1]
    preflight(rig, active=())
    assert len(rig.clients['finish_trajectory'].calls) == 1
    assert len(rig.clients['start_trajectory'].calls) == 2


def test_query_timeout_does_not_stop_localization(rig):
    rig.reset.request(click())
    rig.reset.poll()
    rig.now[0] = 11.0
    rig.reset.poll()
    assert not rig.reset.busy
    assert rig.clients['get_trajectory_states'].removed
    assert not rig.clients['finish_trajectory'].calls


@pytest.mark.parametrize('operation', ['finish_trajectory', 'start_trajectory'])
def test_mutation_timeout_waits_for_same_reply_and_never_duplicates_request(rig, operation):
    preflight(rig)
    if operation == 'start_trajectory':
        reply(rig, 'finish_trajectory', FinishTrajectory.Response())
    original_future = rig.reset.future
    rig.reset.request(click(100.0, 200.0))
    rig.now[0] = 11.0
    rig.reset.poll()
    rig.now[0] = 30.0
    rig.reset.poll()
    assert rig.reset.future is original_future
    assert len(rig.clients[operation].calls) == 1
    assert len(rig.clients['get_trajectory_states'].calls) == 1
    response = (FinishTrajectory.Response() if operation == 'finish_trajectory'
                else StartTrajectory.Response())
    reply(rig, operation, response)
    assert rig.reset.future is not original_future


def test_mutation_transport_failure_blocks_further_unknown_side_effects(rig):
    preflight(rig)
    rig.reset.future.set_exception(RuntimeError('connection lost'))
    rig.reset.poll()
    assert rig.reset.blocked
    rig.reset.request(click())
    rig.reset.poll()
    assert len(rig.clients['get_trajectory_states'].calls) == 1
    assert not rig.clients['start_trajectory'].calls


def test_rapid_clicks_keep_latest_pose_and_serialize_resets(rig):
    preflight(rig)
    rig.reset.request(click(10.0, 11.0))
    rig.reset.request(click(20.0, 21.0))
    reply(rig, 'finish_trajectory', FinishTrajectory.Response())
    reply(rig, 'start_trajectory', StartTrajectory.Response(trajectory_id=43))
    rig.reset.poll()
    assert rig.reset.target.position.x == 20.0
    assert rig.reset.target.position.y == 21.0
    assert len(rig.clients['finish_trajectory'].calls) == 1
    reply(rig, 'get_trajectory_states', states(active=(43,)))
    reply(rig, 'trajectory_query', query())
    assert rig.clients['finish_trajectory'].calls[-1].trajectory_id == 43


@pytest.mark.parametrize('invalid', ['frame', 'nan', 'zero_quaternion', 'infinite_norm'])
def test_invalid_pose_never_reaches_cartographer(rig, invalid):
    message = click()
    if invalid == 'frame':
        message.header.frame_id = 'odom'
    elif invalid == 'nan':
        message.pose.pose.position.x = math.nan
    elif invalid == 'zero_quaternion':
        message.pose.pose.orientation.w = 0.0
    else:
        q = message.pose.pose.orientation
        q.x = q.y = q.z = q.w = 1e308
    rig.reset.request(message)
    rig.reset.poll()
    assert rig.reset.pending is None
    assert not rig.clients['get_trajectory_states'].calls


def test_nonunit_quaternion_normalized_without_modifying_input():
    original = pose()
    original.orientation.w = 2.0
    assert normalized_pose(original).orientation.w == 1.0
    assert original.orientation.w == 2.0


def test_services_not_ready_at_start_are_waited_for_without_finishing(rig):
    rig.clients['start_trajectory'].ready = False
    rig.reset.request(click())
    rig.reset.poll()
    assert not rig.clients['get_trajectory_states'].calls
    rig.clients['start_trajectory'].ready = True
    rig.reset.poll()
    assert len(rig.clients['get_trajectory_states'].calls) == 1


def test_missing_services_and_missing_config_report_errors_before_mutation(rig):
    rig.clients['start_trajectory'].ready = False
    rig.reset.request(click())
    rig.now[0] = 11.0
    rig.reset.poll()
    assert rig.reset.pending is None
    assert 'unavailable' in rig.messages[-1][1]
    rig.clients['start_trajectory'].ready = True
    Path(rig.reset.directory, rig.reset.basename).unlink()
    rig.reset.request(click())
    rig.reset.poll()
    assert not rig.clients['get_trajectory_states'].calls
    assert 'configuration not found' in rig.messages[-1][1]


def test_bridge_launch_is_scoped_to_localization_group():
    launch = Path(__file__).resolve().parents[1] / 'launch/state_estimation_launch.xml'
    root = ElementTree.parse(launch).getroot()
    nodes = root.findall(".//node[@exec='cartographer_initialpose']")
    localization = root.find(".//group[@unless='$(var mapping_bool)']")
    assert len(nodes) == 1
    assert nodes[0] in list(localization)
    assert nodes[0].get('if') == '$(var enable_initialpose)'
    params = {p.get('name'): p.get('value') for p in nodes[0].findall('param')}
    assert params['configuration_directory'] == '$(var slam_config_path)'
    assert params['configuration_basename'] == 'f110_2d_loc.lua'
