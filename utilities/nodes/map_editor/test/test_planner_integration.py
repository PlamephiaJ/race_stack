"""Opt-in end-to-end Qt -> existing optimizer -> preview -> save -> ROS reader."""
import os
import time

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('MAP_EDITOR_TEST_PLANNER') != '1',
                                reason='Set MAP_EDITOR_TEST_PLANNER=1 in a built ROS workspace.')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from map_editor.app import EditorWindow
from map_editor.document import MapDocument
from map_editor.overlays import plan_layers
from map_editor.parameters import METADATA_KEY, PlannerParameters


@pytest.fixture(autouse=True)
def dismiss_readiness_report(monkeypatch):
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: None)


def wait_for_job(window):
    deadline = time.monotonic() + 90
    while window.process is not None and time.monotonic() < deadline:
        QTest.qWait(50)
    assert window.process is None, 'Planner timeout'


def test_full_planning_edit_preview_save_loop(tmp_path):
    from global_planner.readwrite_global_waypoints import read_global_waypoints
    application = QApplication.instance() or QApplication([])
    window = EditorWindow(tmp_path)
    errors = []
    window.error = errors.append
    document = MapDocument.new(520, 520, .05, origin=(-4., 2., .4))
    pixels = document.pixels.copy()
    cv2.circle(pixels, (260, 260), 210, 255, -1)
    cv2.circle(pixels, (260, 260), 150, 0, -1)
    document.commit(pixels)
    window.set_document(document)
    try:
        # Start/cancel must leave document contents and history intact.
        fingerprint = document.fingerprint
        window.start_planner()
        window.cancel_planner()
        wait_for_job(window)
        assert document.fingerprint == fingerprint and document.plan is None
        # A map edit during planning must prevent accepting the old snapshot's result.
        window.start_planner()
        pixels[5, 5] = 255
        document.commit(pixels)
        wait_for_job(window)
        assert not errors, '\n'.join(errors) + window.logs.toPlainText()
        assert document.plan is None
        assert 'discarded outdated result' in window.logs.toPlainText()
        # Regeneration matches the edited map and all four overlay types.
        window.reverse.setChecked(True)
        window.start_planner()
        wait_for_job(window)
        assert not errors, '\n'.join(errors) + window.logs.toPlainText()
        assert document.plan_current and window.plan_pending
        assert len(window.canvas.overlay_items) == 4
        for name, points in plan_layers(document.plan).items():
            coords = np.rint(document.world_to_pixel(points)).astype(int)
            assert np.all((coords >= 0) & (coords < 520)), name
            if name != 'Track Boundaries':
                assert document.free_mask()[coords[:, 1], coords[:, 0]].all(), name
        # Configure both independent partitions using the freshly generated data.
        window.control_tabs.setCurrentIndex(1)
        panel = window.sector_panel
        count = len(document.plan['global_traj_wpnts_iqp']['wpnts'])
        panel.add_boundary(count // 3)
        panel.scaling.setValue(.3)
        panel.kind.setCurrentIndex(1)
        panel.add_boundary(2 * count // 3)
        panel.spline.setValue(10)
        panel.allowed.setChecked(True)
        assert document.sectors['speed'].sections[1]['start'] == count // 3
        assert document.sectors['overtaking'].sections[1]['start'] == 2 * count // 3
        document.path = tmp_path / 'loop/loop.yaml'
        assert window.save()
        assert not window.plan_pending and not document.dirty
        reloaded = MapDocument.load(document.path)
        assert reloaded.plan_current
        assert not reloaded.sector_issues()
        assert reloaded.sectors['speed'].sections[1]['scaling'] == .3
        assert reloaded.sectors['overtaking'].sections[1]['ot_flag']
        from map_editor.readiness import validate_saved_map
        assert validate_saved_map(reloaded).sim_ready
        messages = read_global_waypoints(str(document.path.parent))
        assert len(messages) == 9
        assert len(messages[5].wpnts) > 100
        assert not list(tmp_path.rglob('map_centerline*.csv'))
    finally:
        window.document.saved_fingerprint = window.document.fingerprint
        window.document.saved_sector_fingerprint = window.document.sector_fingerprint
        window.plan_pending = False
        window.close()
        application.processEvents()


def test_parameter_snapshot_invalidation_and_effective_speed_limit(tmp_path):
    application = QApplication.instance() or QApplication([])
    window = EditorWindow(tmp_path)
    errors = []
    window.error = errors.append
    document = MapDocument.new(520, 520, .05)
    pixels = document.pixels.copy()
    cv2.circle(pixels, (260, 260), 210, 255, -1)
    cv2.circle(pixels, (260, 260), 150, 0, -1)
    document.commit(pixels)
    window.set_document(document)
    try:
        window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('BLUE'))
        vehicle = window.planner_parameters.vehicle
        assert vehicle['name'] == 'BLUE'
        values = dict(window.planner_parameters.values)
        values.update({'veh_params.v_max': 3., 'stepsize_opts.stepsize_interp_after_opt': .2,
                       'optim_opts_mincurv.iqp_iters_min': 6})
        window.apply_planner_parameters(PlannerParameters(values, vehicle))
        window.start_planner()
        # Editing only a planner parameter must invalidate the running job.
        values['veh_params.v_max'] = 4.
        window.apply_planner_parameters(PlannerParameters(values, vehicle))
        wait_for_job(window)
        assert not errors, '\n'.join(errors) + window.logs.toPlainText()
        assert document.plan is None
        assert 'discarded outdated result' in window.logs.toPlainText()
        window.start_planner()
        wait_for_job(window)
        assert not errors, '\n'.join(errors) + window.logs.toPlainText()
        assert document.plan_current
        for key in ('centerline_waypoints', 'global_traj_wpnts_iqp', 'global_traj_wpnts_sp'):
            waypoints = document.plan[key]['wpnts']
            max_speed = max(point['vx_mps'] for point in waypoints)
            assert 3.99 < max_speed <= 4.000001, key
            # The 0.2 m output sampling should produce about half as many waypoints as default.
            assert 200 < len(waypoints) < 350, key
        document.path = tmp_path / 'configured/configured.yaml'
        assert window.save()
        loaded = MapDocument.load(document.path)
        assert loaded.metadata[METADATA_KEY]['parameters'] == window.planner_parameters.values
        assert loaded.metadata[METADATA_KEY]['vehicle'] == vehicle
        assert loaded.plan_current
        window.set_document(loaded)
        assert window.planner_parameters.values['veh_params.v_max'] == 4.
        assert window.planner_parameters.values['stepsize_opts.stepsize_interp_after_opt'] == .2
    finally:
        window.document.saved_fingerprint = window.document.fingerprint
        window.plan_pending = False
        window.close()
        application.processEvents()
