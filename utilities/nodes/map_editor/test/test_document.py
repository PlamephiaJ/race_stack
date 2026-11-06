"""Persistence and geometry regressions; independent of Qt and ROS."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from map_editor.document import MapDocument
from map_editor.overlays import plan_layers, transform_plan

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / 'stack_master/maps/GLC_smile_small/GLC_smile_small.yaml'


@pytest.fixture
def plan():
    return json.loads((EXAMPLE.parent / 'global_waypoints.json').read_text())


def test_existing_map_roundtrip_and_save_as(tmp_path):
    document = MapDocument.load(EXAMPLE)
    assert document.plan_current
    assert not document.dirty
    path = document.save(tmp_path / 'custom_map/custom_map.yaml')
    loaded = MapDocument.load(path)
    np.testing.assert_array_equal(loaded.pixels, document.pixels)
    assert loaded.metadata == document.metadata
    assert loaded.plan_current
    assert len(plan_layers(loaded.plan)) == 4
    assert loaded.metadata['image'] == 'custom_map.png'
    assert not list(tmp_path.rglob('*.pbstream'))


def test_edit_invalidates_plan_undo_restores_and_save_archives_stale(tmp_path, plan):
    document = MapDocument.new(50, 50, .05, background=255)
    document.attach_plan(plan)
    path = document.save(tmp_path / 'map.yaml')
    pixels = document.pixels.copy()
    pixels[10:15, 10:15] = 0
    assert document.commit(pixels)
    assert document.dirty and not document.plan_current
    document.undo()
    assert not document.dirty and document.plan_current
    document.redo()
    assert document.dirty and not document.plan_current
    document.save(path)
    assert not (tmp_path / 'global_waypoints.json').exists()
    assert len(list(tmp_path.glob('global_waypoints.stale-*.json'))) == 1
    assert not MapDocument.load(path).plan_current


def test_metadata_undo_and_empty_operations():
    document = MapDocument.new(32, 32, .1)
    assert not document.commit(document.pixels.copy())
    assert not document.undo_stack
    metadata = deepcopy(document.metadata)
    metadata['initial_pose'] = [1., 2., .5]
    document.commit(metadata=metadata)
    document.undo()
    assert 'initial_pose' not in document.metadata
    document.redo()
    assert document.metadata['initial_pose'] == [1., 2., .5]
    document.undo()
    document.commit(np.full((32, 32), 255, np.uint8))
    assert not document.redo_stack


def test_world_pixel_roundtrip_rotated_origin():
    document = MapDocument.new(400, 300, .03, origin=(-4.5, 2.8, 1.2))
    pixels = np.array([[0., 0.], [399., 299.], [156.5, 200.2]])
    np.testing.assert_allclose(document.world_to_pixel(document.pixel_to_world(pixels)), pixels, atol=1e-12)
    np.testing.assert_allclose(document.pixel_to_world([0, 299]), [-4.5, 2.8])


def test_noise_removes_only_small_interior_obstacles():
    document = MapDocument.new(100, 100, .05, background=255)
    pixels = document.pixels.copy()
    pixels[:2, :] = 0
    pixels[20:22, 20:22] = 0
    pixels[50:70, 50:70] = 0
    pixels[30, 30] = 205
    document.commit(pixels)
    assert document.clean_noise(4)
    assert document.pixels[20, 20] == 255
    assert document.pixels[0, 0] == 0
    assert document.pixels[50, 50] == 0
    assert document.pixels[30, 30] == 205
    document.undo()
    np.testing.assert_array_equal(document.pixels, pixels)


def test_negate_and_relative_image_preserve_occupancy(tmp_path):
    image = np.array([[0, 255, 50]], dtype=np.uint8)
    cv2.imwrite(str(tmp_path / 'raw.png'), image)
    metadata = {'image': 'raw.png', 'resolution': .1, 'origin': [-1., 2., .8],
                'negate': 1, 'free_thresh': .196, 'occupied_thresh': .65, 'custom': 'preserved'}
    (tmp_path / 'raw.yaml').write_text(yaml.safe_dump(metadata))
    document = MapDocument.load(tmp_path / 'raw.yaml')
    np.testing.assert_array_equal(document.pixels, 255 - image)
    assert document.free_mask().tolist() == [[True, False, False]]
    document.save(tmp_path / 'output/map.yaml')
    loaded = MapDocument.load(tmp_path / 'output/map.yaml')
    np.testing.assert_array_equal(loaded.free_mask(), document.free_mask())
    assert loaded.metadata['custom'] == 'preserved'


@pytest.mark.parametrize('resolution,origin', [
    (0., (0, 0, 0)), (float('nan'), (0, 0, 0)), (.05, (0, 0, float('inf'))),
])
def test_reject_invalid_geometry(resolution, origin):
    with pytest.raises(ValueError):
        MapDocument.new(100, 100, resolution, origin)


def test_transformed_plan_has_matching_overlay_and_heading(plan):
    original = deepcopy(plan)
    transformed = transform_plan(plan, [-4., 2., .7])
    for name, points in plan_layers(original).items():
        rotation = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
        np.testing.assert_allclose(plan_layers(transformed)[name], points @ rotation.T + [-4., 2.])
    before = original['global_traj_wpnts_iqp']['wpnts'][0]['psi_rad']
    after = transformed['global_traj_wpnts_iqp']['wpnts'][0]['psi_rad']
    assert np.cos(after - before) == pytest.approx(np.cos(.7))


def test_corrupt_optional_trajectory_does_not_block_map(tmp_path):
    document = MapDocument.new(50, 50, .05)
    path = document.save(tmp_path / 'map.yaml')
    (tmp_path / 'global_waypoints.json').write_text('{"centerline_waypoints": {}}')
    loaded = MapDocument.load(path)
    assert loaded.plan is None


def test_unknown_space_is_not_plannable():
    document = MapDocument.new(3, 1, .05)
    document.commit(np.array([[0, 205, 255]], np.uint8))
    assert document.free_mask().tolist() == [[False, False, True]]
