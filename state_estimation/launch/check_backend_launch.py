"""Fail early for unsupported backends or maps without Cartographer state."""

from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def check_backend(context):
    algorithm = LaunchConfiguration('loc_algorithm').perform(context)
    if algorithm not in ('cartographer', 'slam', 'pf', 'synpf'):
        raise RuntimeError(
            f"Unknown loc_algorithm '{algorithm}'; choose synpf, pf or cartographer (alias: slam).")
    mapping = LaunchConfiguration('mapping_bool').perform(context).lower() in ('true', '1')
    if algorithm in ('pf', 'synpf'):
        if mapping:
            get_package_prefix('slam_toolbox')
        else:
            from state_estimation.particle_filter_gpu import require_cuda
            require_cuda()
    elif not mapping:
        map_file = Path(LaunchConfiguration('map_file').perform(context) + '.pbstream')
        if not map_file.is_file():
            raise RuntimeError(
                f'Cartographer needs {map_file}. For a PNG/YAML map use '
                'loc_algorithm:=synpf (or loc_algorithm:=pf).')
    return []


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('loc_algorithm', default_value='synpf'),
        DeclareLaunchArgument('mapping_bool', default_value='true'),
        DeclareLaunchArgument('map_file', default_value=''),
        OpaqueFunction(function=check_backend),
    ])
