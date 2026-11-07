"""Exercise sector interactions on the shared Qt canvas, without ROS."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from map_editor.app import EditorWindow
from map_editor.document import MapDocument

EXAMPLE = Path(__file__).resolve().parents[4] / 'stack_master/maps/GLC_smile_small/GLC_smile_small.yaml'


@pytest.fixture
def window():
    application = QApplication.instance() or QApplication([])
    window = EditorWindow()
    window.set_document(MapDocument.load(EXAMPLE))
    window.resize(1500, 1000)
    window.show()
    window.control_tabs.setCurrentIndex(1)
    application.processEvents()
    errors = []
    window.error = errors.append
    window.test_errors = errors
    yield window
    window.document.saved_fingerprint = window.document.fingerprint
    window.document.saved_sector_fingerprint = window.document.sector_fingerprint
    window.plan_pending = False
    window.close()
    application.processEvents()


def waypoint(canvas, index):
    p = canvas.document.plan['global_traj_wpnts_iqp']['wpnts'][index]
    pixels = canvas.document.world_to_pixel([[p['x_m'], p['y_m']]])[0]
    return canvas.mapFromScene(QPointF(*pixels))


def click(window, index, button=Qt.MouseButton.LeftButton):
    QTest.mouseClick(window.canvas.viewport(), button, pos=waypoint(window.canvas, index))


def test_add_select_drag_delete_and_independent_overtaking(window):
    canvas, panel, doc = window.canvas, window.sector_panel, window.document
    assert canvas.tool == 'Sectors'
    assert panel.editors.isEnabled()
    panel.accept_indices(True)
    old_ot = deepcopy(doc.sectors['overtaking'].raw)
    panel.action.setCurrentIndex(1)
    click(window, 70)
    assert len(doc.sectors['speed'].sections) == 2
    boundary = doc.sectors['speed'].sections[1]['start']
    assert abs(boundary - 70) <= 1
    panel.action.setCurrentIndex(0)
    click(window, 100)
    assert panel.selected == 1
    panel.scaling.setValue(.3)
    assert doc.sectors['speed'].sections[1]['scaling'] == .3
    panel.limit.setValue(.25)
    assert all(section['scaling'] <= .25 for section in doc.sectors['speed'].sections)
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=waypoint(canvas, boundary))
    QTest.mouseMove(canvas.viewport(), waypoint(canvas, 90))
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=waypoint(canvas, 90))
    assert abs(doc.sectors['speed'].sections[1]['start'] - 90) <= 1
    assert doc.sectors['overtaking'].raw == old_ot
    boundary = doc.sectors['speed'].sections[1]['start']
    click(window, boundary, Qt.MouseButton.RightButton)
    assert len(doc.sectors['speed'].sections) == 1
    window.undo()
    assert len(doc.sectors['speed'].sections) == 2
    panel.kind.setCurrentIndex(1)
    assert canvas.sector_kind == 'overtaking'
    panel.accept_indices(True)
    panel.spline.setValue(10)
    panel.allowed.setChecked(True)
    assert doc.sectors['overtaking'].sections[0]['ot_flag']
    assert len(doc.sectors['speed'].sections) == 2
    assert not window.test_errors


def test_invalidation_pauses_controls_and_replan_needs_review(window):
    panel, doc = window.sector_panel, window.document
    pixels = doc.pixels.copy()
    pixels[0, 0] = 255 - pixels[0, 0]
    doc.commit(pixels)
    window.canvas.refresh()
    window._changed()
    assert not panel.editors.isEnabled() and not window.canvas.overlay_items
    window.undo()
    assert panel.editors.isEnabled()
    plan = deepcopy(doc.plan)
    plan['global_traj_wpnts_iqp']['wpnts'][30]['x_m'] += .1
    doc.attach_plan(plan)
    window.canvas.refresh_overlays()
    window._changed()
    assert not panel.editors.isEnabled() and panel.rebind.isVisible()
    assert len(window.canvas.overlay_items) > 4  # unconfirmed colored preview
    panel.accept_indices(False)
    assert panel.editors.isEnabled()
    window.process = object()
    window._changed()
    assert not panel.editors.isEnabled() and window.canvas.sectors_paused
    window.process = None
    window._changed()
    assert panel.editors.isEnabled()


def test_save_displays_readiness_and_preserves_ros_yaml(window, tmp_path, monkeypatch):
    reports = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: reports.append(args[-1]))
    window.document.path = tmp_path / 'track/track.yaml'
    window.sector_panel.scaling.setValue(.3)
    assert window.save()
    assert reports and 'SIM: READY' in reports[0] and 'PBSTREAM' in reports[0]
    loaded = MapDocument.load(window.document.path)
    assert loaded.sectors['speed'].sections[0]['scaling'] == .3
    assert not loaded.sector_issues()
