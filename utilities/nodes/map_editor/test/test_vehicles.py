"""Vehicle selection must affect the optimizer inputs, not just its label."""
import configparser
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from map_editor.parameters import PlannerParameters
from map_editor.vehicles import vehicle_settings

CONFIG = Path(__file__).resolve().parents[4] / 'stack_master/config'


def test_blue_model_and_hardware_limits():
    values, vehicle = vehicle_settings(CONFIG / 'BLUE')
    model = yaml.safe_load((CONFIG / 'BLUE/car_model.yaml').read_text())
    assert values['veh_params.mass'] == model['m']
    assert values['veh_params.width'] == .295
    assert values['veh_params.v_max'] == pytest.approx(23250 / 4176.3)
    assert values['veh_params.curvlim'] == pytest.approx(math.tan(.3465959328) / .321)
    assert vehicle['model_speed'] == 10
    assert vehicle['acceleration_limit'] == 2.5  # Manager is stricter than car_model's 3.
    assert vehicle['deceleration_limit'] == 2.5
    assert yaml.safe_load((CONFIG / 'BLUE/car_model.yaml').read_text())['racecar_version'] == 'BLUE'


def test_vehicle_snapshot_changes_ini_and_caps_csv_without_changing_project(tmp_path):
    source = CONFIG / 'global_planner'
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    current = PlannerParameters.from_config(source)
    current.values['safety_width'] = .85
    current.values['stepsize_opts.stepsize_reg'] = .15
    parameters = current.for_vehicle(CONFIG / 'BLUE', PlannerParameters.from_config(source))
    destination = parameters.snapshot(source, tmp_path / 'job')
    parser = configparser.ConfigParser()
    parser.read(destination / 'racecar_f110.ini')
    vehicle = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
    assert vehicle['mass'] == parameters.values['veh_params.mass'] and vehicle['width'] == .295
    assert vehicle['v_max'] == pytest.approx(5.567128798218518)
    assert vehicle['length'] == .535 and vehicle['dragcoeff'] == .0136
    assert parameters.values['safety_width'] == .85
    assert parameters.values['stepsize_opts.stepsize_reg'] == .15
    ggv = np.loadtxt(destination / 'veh_dyn_info/ggv.csv', delimiter=',')
    original = np.loadtxt(source / 'veh_dyn_info/ggv.csv', delimiter=',')
    np.testing.assert_allclose(ggv[:, 0], original[:, 0])
    np.testing.assert_allclose(ggv[:, 2], original[:, 2])  # Lateral grip is not inferred.
    np.testing.assert_allclose(ggv[:, 1], np.minimum(original[:, 1], 2.5))
    machine = np.loadtxt(destination / 'veh_dyn_info/ax_max_machines.csv', delimiter=',')
    assert np.max(machine[:, 1]) <= 2.5
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_saved_snapshot_does_not_silently_reload_vehicle_source(tmp_path):
    parameters = PlannerParameters().for_vehicle(CONFIG / 'BLUE')
    path = tmp_path / 'vehicle_profile.yaml'
    parameters.export(path)
    loaded = PlannerParameters.load(path)
    assert loaded.profile() == parameters.profile()
    assert loaded.vehicle['name'] == 'BLUE'
    values = dict(loaded.values, **{'veh_params.v_max': 6.})
    with pytest.raises(ValueError, match='exceeds BLUE limit'):
        PlannerParameters(values, loaded.vehicle)
    malformed = deepcopy(loaded.profile())
    malformed['vehicle']['acceleration_limit'] = float('nan')
    with pytest.raises(ValueError, match='acceleration_limit'):
        PlannerParameters.from_profile(malformed)
    # Old profiles continue to work and have no vehicle-specific caps.
    assert PlannerParameters.from_profile({'version': 1, 'parameters': {}}).vehicle is None


def test_missing_vesc_uses_model_and_invalid_inputs_do_not_fall_back_silently(tmp_path):
    directory = tmp_path / 'car'
    directory.mkdir()
    shutil.copyfile(CONFIG / 'BLUE/car_model.yaml', directory / 'car_model.yaml')
    values, vehicle = vehicle_settings(directory)
    assert values['veh_params.v_max'] == 10 and vehicle['vesc_speed'] is None
    model = yaml.safe_load((directory / 'car_model.yaml').read_text())
    model['wheelbase'] = 0
    (directory / 'car_model.yaml').write_text(yaml.safe_dump(model))
    with pytest.raises(ValueError, match='wheelbase'):
        vehicle_settings(directory)


def test_driver_specific_erpm_and_offset_override_shared_values(tmp_path):
    directory = tmp_path / 'car'
    shutil.copytree(CONFIG / 'BLUE', directory)
    path = directory / 'vesc.yaml'
    data = yaml.safe_load(path.read_text())
    data['ackermann_to_vesc_node']['ros__parameters']['speed_to_erpm_offset'] = 100
    data['vesc_driver_node']['ros__parameters']['speed_max'] = 10000
    path.write_text(yaml.safe_dump(data))
    values, vehicle = vehicle_settings(directory)
    assert values['veh_params.v_max'] == pytest.approx((10000 - 100) / 4176.3)
    assert vehicle['vesc_speed'] == values['veh_params.v_max']
