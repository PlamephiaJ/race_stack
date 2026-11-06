"""Launch standalone Follow the Gap with optional typed parameter overrides."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_ftg(context):
    config_dir = os.path.join(
        get_package_share_directory('stack_master'), 'config')
    racecar_version = LaunchConfiguration('racecar_version').perform(context)
    sim_opp = IfCondition(LaunchConfiguration('sim_opp')).evaluate(context)

    parameters = []
    # Use the same servo calibration as the selected car's actuator stack.
    # SIM has no VESC; retain the node's steering defaults in that case.
    if racecar_version != 'SIM':
        parameters.append(os.path.join(config_dir, racecar_version, 'vesc.yaml'))
    parameters.append(LaunchConfiguration('param_file'))

    overrides = {}
    for name in ('max_speed', 'max_deceleration', 'minimum_gap_angle'):
        if LaunchConfiguration(name).perform(context).strip():
            overrides[name] = ParameterValue(
                LaunchConfiguration(name), value_type=float)
    parameters.append(overrides)

    return [Node(
        package='controller',
        executable='reactive_node',
        name='reactive_follow_gap_node',
        namespace='opp_racecar' if sim_opp else '',
        output='screen',
        parameters=parameters,
        remappings=[
            ('/scan', '/opp_scan'),
            ('/drive', '/opp_drive'),
            ('/visualization_marker', '/opp_racecar/visualization_marker'),
        ] if sim_opp else [],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'racecar_version', default_value=EnvironmentVariable(
                'RACECAR_VERSION', default_value='BLUE')),
        DeclareLaunchArgument('sim_opp', default_value='false'),
        DeclareLaunchArgument(
            'param_file', default_value=os.path.join(
                get_package_share_directory('stack_master'),
                'config', 'follow_the_gap_params.yaml')),
        DeclareLaunchArgument('max_speed', default_value=''),
        DeclareLaunchArgument('max_deceleration', default_value=''),
        DeclareLaunchArgument('minimum_gap_angle', default_value=''),
        OpaqueFunction(function=launch_ftg),
    ])
