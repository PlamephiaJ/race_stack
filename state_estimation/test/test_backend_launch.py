"""Resolve actual launch actions without starting sensors or processes."""
from pathlib import Path
import runpy

import pytest
from launch import LaunchDescription, LaunchService
from launch import LaunchContext
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.utilities import perform_substitutions


@pytest.mark.parametrize('algorithm', [None, 'cartographer', 'slam', 'pf', 'synpf'])
@pytest.mark.parametrize('mapping', ['true', 'false'])
def test_backend_exclusivity(monkeypatch, tmp_path, algorithm, mapping):
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    # This test inspects launch selection; real CUDA access is checked separately.
    monkeypatch.setattr('state_estimation.particle_filter_gpu.require_cuda', lambda: 1)
    commands = []

    def record(process, context):
        commands.append([perform_substitutions(context, part) for part in process.cmd])
        return []

    monkeypatch.setattr(ExecuteProcess, 'execute', record)
    path = Path(__file__).resolve().parents[1] / 'launch/state_estimation_launch.xml'
    arguments = {'mapping_bool': mapping, 'map_name': 'GLC_smile_small',
                 'racecar_version': 'BLUE', 'use_vio': 'false'}
    if algorithm is not None:
        arguments['loc_algorithm'] = algorithm
    action = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(str(path)), launch_arguments=arguments.items())
    algorithm = algorithm or 'synpf'
    service = LaunchService()
    service.include_launch_description(LaunchDescription([action]))
    assert service.run() == 0
    names = [Path(command[0]).name for command in commands]
    assert names.count('ekf_node') == 1
    assert names.count('carstate_node') == 1
    if algorithm in ('pf', 'synpf'):
        assert 'cartographer_node' not in names
        assert 'cartographer_initialpose' not in names
        expected = 'async_slam_toolbox_node' if mapping == 'true' else ('synpf' if algorithm == 'synpf' else 'particle_filter')
        assert names.count(expected) == 1
        other = 'particle_filter' if algorithm == 'synpf' else 'synpf'
        assert other not in names
    else:
        assert names.count('cartographer_node') == 1
        assert 'particle_filter' not in names
        assert 'synpf' not in names
        assert 'async_slam_toolbox_node' not in names
        assert ('cartographer_initialpose' in names) == (mapping == 'false')
    assert names.count('map_server') == (0 if mapping == 'true' else 1)


def test_unknown_backend_reports_supported_choices():
    path = Path(__file__).resolve().parents[1] / 'launch/check_backend_launch.py'
    check = runpy.run_path(str(path))['check_backend']
    context = LaunchContext()
    context.launch_configurations['loc_algorithm'] = 'unknown'
    with pytest.raises(RuntimeError, match='choose synpf'):
        check(context)


def test_cartographer_map_without_pbstream_reports_pf_option(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'launch/check_backend_launch.py'
    check = runpy.run_path(str(path))['check_backend']
    context = LaunchContext()
    context.launch_configurations.update(
        loc_algorithm='cartographer', mapping_bool='false',
        map_file=str(tmp_path / 'occupancy_only'))
    with pytest.raises(RuntimeError, match='loc_algorithm:=pf'):
        check(context)
