"""Exercise the real Cartographer services in an isolated ROS domain and namespace."""

import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from cartographer_ros_msgs.msg import TrajectoryStates
from cartographer_ros_msgs.srv import GetTrajectoryStates, TrajectoryQuery
from geometry_msgs.msg import PoseWithCovarianceStamped
import pytest
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.utilities import get_rmw_implementation_identifier

from state_estimation.cartographer_initialpose import CartographerInitialPose


pytestmark = pytest.mark.skipif(
    os.environ.get('STATE_ESTIMATION_TEST_CARTOGRAPHER') != '1',
    reason='Set STATE_ESTIMATION_TEST_CARTOGRAPHER=1 to run isolated Cartographer integration.')


def test_rviz_best_effort_pose_restarts_real_cartographer(tmp_path, monkeypatch):
    assert get_rmw_implementation_identifier() == 'rmw_fastrtps_cpp', (
        'Start Python with RMW_IMPLEMENTATION=rmw_fastrtps_cpp '
        'for this isolated integration test.')
    source = Path(__file__).resolve().parents[2]
    directory = source / 'stack_master/config/BLUE/slam'
    map_file = source / 'stack_master/maps/GLC_smile_small/GLC_smile_small.pbstream'
    map_before = map_file.read_bytes()
    namespace = '/initialpose_test_' + uuid.uuid4().hex[:12]
    domain = 213
    ros_settings = {'ROS_DOMAIN_ID': str(domain), 'ROS_LOCALHOST_ONLY': '1',
                    'ROS_LOG_DIR': str(tmp_path / 'ros_logs'),
                    'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp'}
    environment = dict(os.environ, **ros_settings)
    for key, value in ros_settings.items():
        monkeypatch.setenv(key, value)
    output = (tmp_path / 'cartographer.log').open('w')
    process = subprocess.Popen([
        '/opt/ros/humble/lib/cartographer_ros/cartographer_node',
        '-configuration_directory', str(directory),
        '-configuration_basename', 'f110_2d_loc.lua',
        '-load_state_filename', str(map_file), '-load_frozen_state=true', '--minloglevel=1',
        '--ros-args', '-r', '__ns:=' + namespace,
        '-r', '/tf:=' + namespace + '/tf', '-r', '/tf_static:=' + namespace + '/tf_static',
    ], env=environment, stdout=output, stderr=subprocess.STDOUT)
    context = Context()
    executor = None
    nodes = []
    try:
        context.init(domain_id=domain)
        executor = SingleThreadedExecutor(context=context)
        bridge = CartographerInitialPose(
            context=context, namespace=namespace, parameter_overrides=[
                Parameter('configuration_directory', value=str(directory)),
                Parameter('initialpose_topic', value='initialpose'),
                Parameter('service_timeout_sec', value=20.0),
            ])
        probe = Node('probe', namespace=namespace, context=context)
        nodes.extend((bridge, probe))
        for node in nodes:
            executor.add_node(node)

        def until(predicate, timeout=30.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                assert process.poll() is None, (tmp_path / 'cartographer.log').read_text()
                if predicate():
                    return
                executor.spin_once(timeout_sec=0.05)
            pytest.fail('ROS integration timeout:\n'
                        + repr(probe.get_service_names_and_types()) + '\n'
                        + (tmp_path / 'cartographer.log').read_text()[-6000:])

        def call(client, request):
            until(client.service_is_ready)
            future = client.call_async(request)
            until(future.done)
            response = future.result()
            assert response.status.code == 0, response.status.message
            return response

        state_client = probe.create_client(GetTrajectoryStates, 'get_trajectory_states')
        initial = call(state_client, GetTrajectoryStates.Request()).trajectory_states
        frozen = [t for t, s in zip(initial.trajectory_id, initial.trajectory_state)
                  if s == TrajectoryStates.FROZEN]
        old_active = [t for t, s in zip(initial.trajectory_id, initial.trajectory_state)
                      if s == TrajectoryStates.ACTIVE]
        assert frozen and len(old_active) == 1
        reference = min(frozen)
        query_client = probe.create_client(TrajectoryQuery, 'trajectory_query')
        reference_poses = call(
            query_client, TrajectoryQuery.Request(trajectory_id=reference)).trajectory
        assert reference_poses
        target = reference_poses[0].pose
        target.position.x += 1.0
        message = PoseWithCovarianceStamped()
        message.header.frame_id = 'map'
        message.pose.pose = target
        publisher = probe.create_publisher(
            PoseWithCovarianceStamped, 'initialpose',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        until(lambda: publisher.get_subscription_count() > 0)

        def reset_once():
            bridge.reset.finished_old = False
            publisher.publish(message)
            until(lambda: bridge.reset.finished_old and not bridge.reset.busy
                  and bridge.reset.future is None)
            assert not bridge.reset.blocked
            result = call(state_client, GetTrajectoryStates.Request()).trajectory_states
            states = dict(zip(result.trajectory_id, result.trajectory_state))
            assert all(states[t] == TrajectoryStates.FROZEN for t in frozen)
            active = [t for t, state in states.items() if state == TrajectoryStates.ACTIVE]
            assert len(active) == 1
            return active[0], states

        first_id, first_states = reset_once()
        assert first_id not in old_active
        assert first_states[old_active[0]] in (TrajectoryStates.FINISHED, TrajectoryStates.DELETED)
        second_id, second_states = reset_once()
        assert second_id != first_id
        assert second_states[first_id] in (TrajectoryStates.FINISHED, TrajectoryStates.DELETED)
        assert map_file.read_bytes() == map_before
    finally:
        if executor is not None:
            executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        if context.ok():
            context.shutdown()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        output.close()
