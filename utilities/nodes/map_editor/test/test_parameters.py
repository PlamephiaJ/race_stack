"""Profiles, validation and copying real planner inputs without changing defaults."""
import configparser
import json
from pathlib import Path

import pytest
import yaml

from map_editor.parameters import PlannerParameters

CONFIG = Path(__file__).resolve().parents[4] / 'stack_master/config/global_planner'


def test_config_snapshot_applies_overrides_preserves_source_and_other_options(tmp_path):
    original = {path.relative_to(CONFIG): path.read_bytes() for path in CONFIG.rglob('*') if path.is_file()}
    parameters = PlannerParameters.from_config(CONFIG)
    parameters.values.update({'veh_params.v_max': 4., 'veh_params.mass': 4.1, 'safety_width': .85,
                              'stepsize_opts.stepsize_interp_after_opt': .2,
                              'optim_opts_mincurv.iqp_iters_min': 7,
                              'vel_calc_opts.vel_profile_conv_filt_window': 5, 'reverse_mapping': True})
    destination = parameters.snapshot(CONFIG, tmp_path / 'job-config')
    parser = configparser.ConfigParser()
    parser.read(destination / 'racecar_f110.ini')
    vehicle = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
    assert vehicle['v_max'] == 4.
    assert vehicle['mass'] == 4.1
    assert vehicle['g'] == 9.81  # Untouched keys in the same JSON object survive.
    steps = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
    assert steps['stepsize_interp_after_opt'] == .2
    assert steps['stepsize_reg'] == parameters.values['stepsize_opts.stepsize_reg']
    assert json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mincurv'))['iqp_iters_min'] == 7
    assert parameters.values == PlannerParameters.from_config(destination).values
    assert original == {path.relative_to(CONFIG): path.read_bytes() for path in CONFIG.rglob('*') if path.is_file()}
    for name in ('ggv.csv', 'ax_max_machines.csv'):
        assert (destination / 'veh_dyn_info' / name).read_bytes() == (CONFIG / 'veh_dyn_info' / name).read_bytes()


def test_profile_export_import_preserves_values(tmp_path):
    parameters = PlannerParameters.from_config(CONFIG)
    parameters.values['veh_params.v_max'] = 3.75
    parameters.values['vel_calc_opts.vel_profile_conv_filt_window'] = 7
    path = tmp_path / 'profiles/test.yaml'
    parameters.export(path)
    assert PlannerParameters.load(path).values == parameters.values
    partial = {'version': 1, 'parameters': {'veh_params.v_max': 2.5}}
    loaded = PlannerParameters.from_profile(partial, parameters)
    assert loaded.values['veh_params.v_max'] == 2.5
    assert loaded.values['vel_calc_opts.vel_profile_conv_filt_window'] == 7


@pytest.mark.parametrize('key,value', [
    ('veh_params.v_max', float('nan')), ('veh_params.mass', -1.), ('veh_params.width', .8),
    ('safety_width', True), ('reverse_mapping', 'false'), ('optim_opts_mincurv.iqp_iters_min', 2.5),
    ('vel_calc_opts.vel_profile_conv_filt_window', 4), ('vel_calc_opts.vel_profile_conv_filt_window', 0),
    ('ggv_file', '/tmp/ggv.csv'), ('ggv_file', '../ggv.csv'), ('ggv_file', ''), ('typo', 1),
])
def test_invalid_profiles_rejected(key, value):
    with pytest.raises(ValueError):
        PlannerParameters.from_profile({'version': 1, 'parameters': {key: value}})


@pytest.mark.parametrize('profile', [None, {}, {'version': True, 'parameters': {}},
                                     {'version': 2, 'parameters': {}}, {'version': 1, 'parameters': []}])
def test_malformed_profile_rejected(profile):
    with pytest.raises(ValueError):
        PlannerParameters.from_profile(profile)


def test_missing_dynamics_file_reports_error(tmp_path):
    parameters = PlannerParameters({'ggv_file': 'missing.csv'})
    with pytest.raises(ValueError, match='Missing dynamics CSV'):
        parameters.snapshot(CONFIG, tmp_path / 'config')


def test_project_node_defaults_loaded(tmp_path):
    import shutil
    shutil.copytree(CONFIG, tmp_path / 'config')
    path = tmp_path / 'config/global_planner_params.yaml'
    data = yaml.safe_load(path.read_text())
    data['global_planner']['ros__parameters']['safety_width'] = .95
    data['global_planner']['ros__parameters']['reverse_mapping'] = True
    path.write_text(yaml.safe_dump(data))
    parameters = PlannerParameters.from_config(tmp_path / 'config')
    assert parameters.values['safety_width'] == .95
    assert parameters.values['reverse_mapping'] is True
