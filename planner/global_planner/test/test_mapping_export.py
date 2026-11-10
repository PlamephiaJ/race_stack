"""Both map backends export the same grid, metadata and initial base pose."""
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
import pytest

from global_planner.mapping_node import MappingNode


@pytest.mark.parametrize('backend', ['pf', 'synpf'])
def test_pf_save_exports_standard_artifacts_without_cartographer(tmp_path, backend):
    grid = np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8)
    obj = SimpleNamespace(
        map_occupancy_grid=[100, 0, 100, 0, 0, 100], map_name='test_pf',
        map_dir=tmp_path / 'test_pf', filtered_map=grid,
        map_resolution=0.05, map_origin=SimpleNamespace(x=-2.0, y=-3.0),
        initial_position=(1.0, 2.0, 0.4), loc_algorithm=backend,
        filter_map_occupancy_grid=lambda: None,
        get_logger=lambda: SimpleNamespace(info=lambda *a: None, warn=lambda *a: None))
    # There are deliberately no finish/write clients: PF saving must not use them.
    MappingNode.save_map(obj)
    assert {p.name for p in obj.map_dir.iterdir()} == {
        'test_pf.png', 'test_pf.yaml', 'pf_map.png', 'pf_map.yaml'}
    for name in ['test_pf', 'pf_map']:
        metadata = yaml.safe_load((obj.map_dir / f'{name}.yaml').read_text())
        assert metadata == dict(image=f'{name}.png', resolution=0.05,
                                origin=[-2.0, -3.0, 0], negate=0,
                                occupied_thresh=0.65, free_thresh=0.196,
                                initial_pose=[1.0, 2.0, 0.4])
        assert np.array_equal(cv2.imread(str(obj.map_dir / f'{name}.png'), 0), grid[::-1])
