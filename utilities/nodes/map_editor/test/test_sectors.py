"""Sector persistence, runtime interpolation parity and trajectory binding."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from map_editor.document import MapDocument
from map_editor.readiness import validate_saved_map
from map_editor.sectors import SectorConfig

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / 'stack_master/maps/GLC_smile_small/GLC_smile_small.yaml'


@pytest.fixture
def document():
    return MapDocument.load(EXAMPLE)


def test_original_yaml_roundtrip_preserves_unrelated_fields(document, tmp_path):
    speed = document.sectors['speed']
    speed.raw['other_node'] = {'ros__parameters': {'keep': 'untouched'}}
    speed.params['custom'] = {'nested': [1, 2]}
    speed.sections[0]['only_FTG'] = True
    speed.sections[0]['no_FTG'] = True
    before = deepcopy(speed.raw)
    document.edit_sectors('speed', lambda config: config.sections[0].update(scaling=.2))
    expected = deepcopy(before)
    expected[speed.node_key]['ros__parameters']['Sector0']['scaling'] = .2
    path = document.save(tmp_path / 'track/track.yaml')
    assert yaml.safe_load((path.parent / 'speed_scaling.yaml').read_text()) == expected
    assert yaml.safe_load((path.parent / 'ot_sectors.yaml').read_text()) == document.sectors['overtaking'].raw
    loaded = MapDocument.load(path)
    assert not loaded.sector_issues() and not loaded.dirty
    assert loaded.sectors['speed'].sections[-1]['end'] == len(loaded.plan['global_traj_wpnts_iqp']['wpnts'])
    assert validate_saved_map(loaded).sim_ready


def test_independent_split_move_delete_and_undo(document):
    original_ot = deepcopy(document.sectors['overtaking'].raw)
    count = len(document.plan['global_traj_wpnts_iqp']['wpnts'])
    document.accept_sector_indices('speed', reset=True)
    assert document.edit_sectors('speed', lambda config: config.split(70)) == 1
    assert len(document.sectors['speed'].sections) == 2
    document.edit_sectors('speed', lambda config: config.move_boundary(1, 90, count))
    assert document.sectors['speed'].sections[0]['end'] == 89
    document.edit_sectors('speed', lambda config: config.delete_boundary(1))
    assert len(document.sectors['speed'].sections) == 1
    document.undo()
    assert document.sectors['speed'].sections[1]['start'] == 90
    document.undo()
    assert document.sectors['speed'].sections[1]['start'] == 70
    document.redo()
    assert document.sectors['speed'].sections[1]['start'] == 90
    assert document.sectors['overtaking'].raw == original_ot
    with pytest.raises(ValueError, match='fixed'):
        document.edit_sectors('speed', lambda config: config.delete_boundary(0))
    with pytest.raises(ValueError, match='between'):
        document.edit_sectors('speed', lambda config: config.move_boundary(1, 0, count))


def test_interpolation_matches_actual_runtime_method():
    # Extract the production method without importing rclpy/ROS messages. This
    # catches drift in seam semantics and branch ordering for short sections.
    source = (REPO / 'utilities/nodes/sector_tuner/sector_tuner/sector_tuner.py').read_text()
    node = ast.parse(source)
    cls = next(item for item in node.body if isinstance(item, ast.ClassDef) and item.name == 'SectorTuner')
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == 'get_vel_scaling')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<runtime method>', 'exec'), namespace)
    for starts in ([0], [0, 50, 100], [0, 3, 8, 60, 140]):
        for terminal in (159, 160):
            config = SectorConfig.default('speed', 160)
            for index in starts[1:]:
                config.split(index)
            for i, section in enumerate(config.sections):
                section['scaling'] = .15 + .1 * i
            config.sections[-1]['end'] = terminal
            runtime = SimpleNamespace(n_sectors=len(starts), sectors_params=config.params)
            expected = [namespace['get_vel_scaling'](runtime, index) for index in range(160)]
            np.testing.assert_allclose(config.speed_factors(160), expected)


def test_same_count_changed_geometry_requires_explicit_review(document, tmp_path):
    plan = deepcopy(document.plan)
    plan['global_traj_wpnts_iqp']['wpnts'][70]['x_m'] += .05
    document.attach_plan(plan)
    assert document.plan_current
    assert any('trajectory changed' in issue for issue in document.sector_issues())
    with pytest.raises(ValueError, match='Resolve Sector'):
        document.save(tmp_path / 'track/track.yaml')
    assert not (tmp_path / 'track').exists()
    for kind in ('speed', 'overtaking'):
        document.accept_sector_indices(kind)
    assert not document.sector_issues()
    document.save(tmp_path / 'track/track.yaml')
    assert validate_saved_map(document).sim_ready


def test_invalid_after_shorter_replan_can_reset(document):
    plan = deepcopy(document.plan)
    plan['global_traj_wpnts_iqp']['wpnts'] = plan['global_traj_wpnts_iqp']['wpnts'][:50]
    document.attach_plan(plan)
    assert document.sector_issues('speed', check_binding=False)
    with pytest.raises(ValueError, match='indices'):
        document.accept_sector_indices('speed')
    document.accept_sector_indices('speed', reset=True)
    assert not document.sector_issues('speed')
    assert document.sectors['speed'].sections[0]['end'] == 49


def test_map_edit_pauses_sectors_and_persisted_binding_survives_draft_save(document, tmp_path):
    old_plan = deepcopy(document.plan)
    pixels = document.pixels.copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    document.commit(pixels)
    with pytest.raises(ValueError, match='current trajectory'):
        document.edit_sectors('speed', lambda config: config.split(10))
    document.undo()
    assert document.plan_current and not document.sector_issues()
    document.redo()
    path = document.save(tmp_path / 'track/track.yaml')
    assert not validate_saved_map(document).sim_ready
    loaded = MapDocument.load(path)
    assert not loaded.plan_current
    old_plan['global_traj_wpnts_iqp']['wpnts'][30]['x_m'] += .1
    loaded.attach_plan(old_plan)
    assert loaded.sector_issues('speed')


def test_default_files_and_detect_external_trajectory_replacement(document, tmp_path):
    path = document.save(tmp_path / 'track/track.yaml')
    (path.parent / 'ot_sectors.yaml').unlink()
    loaded = MapDocument.load(path)
    assert loaded.plan_current and 'overtaking' in loaded.sectors
    assert not validate_saved_map(loaded).sim_ready
    loaded.save(path)
    assert validate_saved_map(loaded).sim_ready
    plan = deepcopy(loaded.plan)
    plan['global_traj_wpnts_iqp']['wpnts'][20]['vx_mps'] += .1
    (path.parent / 'global_waypoints.json').write_text(json.dumps(plan))
    assert not MapDocument.load(path).plan_current
    assert not validate_saved_map(loaded).sim_ready


def test_malformed_yaml_preserved_until_explicit_reset(document, tmp_path):
    path = document.save(tmp_path / 'track/track.yaml')
    sector_path = path.parent / 'speed_scaling.yaml'
    sector_path.write_text('broken: [')
    loaded = MapDocument.load(path)
    assert loaded.plan_current and loaded.sector_load_errors
    with pytest.raises(ValueError, match='Sector configuration'):
        loaded.save(path)
    assert sector_path.read_text() == 'broken: ['
    loaded.accept_sector_indices('speed', reset=True)
    loaded.save(path)
    assert validate_saved_map(loaded).sim_ready


def test_pbstream_missing_renamed_geometry_changed_and_replaced(document, tmp_path):
    document.localization_baseline = None
    path = document.save(tmp_path / 'track/track.yaml')
    report = validate_saved_map(document)
    assert report.sim_ready and report.localization_status == 'missing'
    assert 'non-scanned' in report.localization
    (path.parent / 'wrong_name.pbstream').write_bytes(b'not verified')
    assert 'wrong_name.pbstream' in validate_saved_map(document).localization
    path.with_suffix('.pbstream').write_bytes(b'baseline state')
    loaded = MapDocument.load(path)
    assert validate_saved_map(loaded).localization_status == 'unverified'
    path.with_suffix('.pbstream').write_bytes(b'other state')
    assert validate_saved_map(loaded).localization_status == 'changed'
    path.with_suffix('.pbstream').write_bytes(b'baseline state')
    pixels = loaded.pixels.copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    loaded.commit(pixels)
    assert validate_saved_map(loaded).localization_status == 'changed'
    loaded.save(path)
    assert validate_saved_map(MapDocument.load(path)).localization_status == 'changed'


def test_overtaking_bad_transition_rejected_before_mutation():
    config = SectorConfig.default('overtaking', 50)
    config.split(45)
    config.sections[1]['ot_flag'] = True
    with pytest.raises(ValueError, match='transition exceeds'):
        config.validate(50)


def test_reset_repairs_invalid_required_fields_and_preserves_extras(document):
    config = document.sectors['speed']
    config.params['global_limit'] = 'invalid'
    config.params['keep'] = 123
    config.sections[0]['scaling'] = float('nan')
    config.sections[0]['only_FTG'] = True
    config.sections[0]['no_FTG'] = True
    document.accept_sector_indices('speed', reset=True)
    assert not document.sector_issues('speed')
    repaired = document.sectors['speed']
    assert repaired.params['keep'] == 123
    assert repaired.sections[0]['only_FTG'] and repaired.sections[0]['no_FTG']


def test_overtaking_adjacent_enabled_sections_skip_transition_loops():
    config = SectorConfig.default('overtaking', 50)
    config.split(45)
    for section in config.sections:
        section['ot_flag'] = True
    config.validate(50)  # Runtime uses clipped slices for adjoining enabled sections.


def test_speed_only_replan_keeps_indices_bound(document):
    plan = deepcopy(document.plan)
    for waypoint in plan['global_traj_wpnts_iqp']['wpnts']:
        waypoint['vx_mps'] *= .7
    document.attach_plan(plan)
    assert not document.sector_issues()
