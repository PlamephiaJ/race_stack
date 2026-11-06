"""Offscreen interactions exercise the actual canvas and background process UI."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from map_editor.app import EditorWindow
from map_editor.document import MapDocument
from map_editor.parameter_dialog import PlannerParameterDialog
from map_editor.parameters import METADATA_KEY, PlannerParameters


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    window = EditorWindow()
    window.set_document(MapDocument.new(200, 200, .05))
    window.resize(1400, 900)
    window.show()
    app.processEvents()
    yield window
    window.document.saved_fingerprint = window.document.fingerprint
    window.plan_pending = False
    window.close()


def point(canvas, x, y):
    from PySide6.QtCore import QPointF
    return canvas.mapFromScene(QPointF(x, y))


def drag(canvas, start, end):
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(canvas, *start))
    QTest.mouseMove(canvas.viewport(), point(canvas, *end))
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(canvas, *end))


def test_drawing_tools_undo_and_polygon(window):
    canvas = window.canvas
    canvas.material = 255
    drag(canvas, (20, 20), (100, 20))
    assert window.document.pixels[20, 60] == 255
    assert len(window.document.undo_stack) == 1
    window.undo()
    assert window.document.pixels[20, 60] == 0
    window.redo()
    assert window.document.pixels[20, 60] == 255
    canvas.set_tool('Rectangle')
    drag(canvas, (40, 40), (80, 80))
    assert window.document.pixels[60, 60] == 255
    canvas.set_tool('Polygon')
    for p in ((120, 100), (180, 100), (180, 180)):
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(canvas, *p))
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    assert window.document.pixels[120, 170] == 255
    canvas.set_tool('Line')
    canvas.material = 0
    drag(canvas, (40, 60), (80, 60))
    assert window.document.pixels[60, 60] == 0
    canvas.set_tool('Eraser')
    drag(canvas, (40, 60), (80, 60))
    assert window.document.pixels[60, 60] == 255


def test_cancel_preview_and_initial_pose(window):
    canvas = window.canvas
    canvas.material = 255
    # Clicking the gray margin must not draw a clamped stroke on the image border.
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(canvas, -5, 100))
    assert not window.document.pixels.any()
    assert not window.document.undo_stack
    canvas.set_tool('Rectangle')
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(canvas, 40, 40))
    QTest.mouseMove(canvas.viewport(), point(canvas, 80, 80))
    QTest.keyClick(canvas, Qt.Key.Key_Escape)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point(canvas, 80, 80))
    assert not window.document.pixels.any()
    canvas.set_tool('Initial pose')
    drag(canvas, (100, 100), (140, 100))
    pose = window.document.metadata['initial_pose']
    np.testing.assert_allclose(pose, [5., 4.95, 0.], atol=.06)
    window.undo()
    assert 'initial_pose' not in window.document.metadata


def test_existing_preview_toggles_and_stale_hiding(window):
    example = Path(__file__).resolve().parents[4] / 'stack_master/maps/GLC_smile_small/GLC_smile_small.yaml'
    window.set_document(MapDocument.load(example))
    assert len(window.canvas.overlay_items) == 4
    # Opening and accepting an unchanged parameter dialog keeps the old preview valid.
    dialog = PlannerParameterDialog(window.planner_parameters, window.project_parameters, window)
    dialog.accept()
    window.apply_planner_parameters(dialog.result_parameters)
    assert window.document.plan_current and not window.document.dirty
    window.toggle_layer('Centerline', False)
    assert len(window.canvas.overlay_items) == 3
    window.document.commit(255 - window.document.pixels)
    window.canvas.refresh()
    assert len(window.canvas.overlay_items) == 0
    window.undo()
    assert len(window.canvas.overlay_items) == 3


def test_parameter_edit_invalidates_preview_undo_restores_controls_and_save(window, tmp_path):
    example = Path(__file__).resolve().parents[4] / 'stack_master/maps/GLC_smile_small/GLC_smile_small.yaml'
    window.set_document(MapDocument.load(example))
    before = dict(window.planner_parameters.values)
    parameters = PlannerParameters(dict(before, **{'veh_params.v_max': 3.5, 'safety_width': .800123}))
    dialog = PlannerParameterDialog(parameters, window.project_parameters, window)
    dialog.editors['veh_params.v_max'].setValue(4.)
    dialog.editors['vel_calc_opts.vel_profile_conv_filt_window'].setValue(5)
    dialog.accept()
    window.apply_planner_parameters(dialog.result_parameters)
    assert not window.document.plan_current and not window.canvas.overlay_items
    assert window.safety.value() == .800123
    assert window.document.metadata[METADATA_KEY]['parameters']['veh_params.v_max'] == 4.
    window.undo()
    assert window.document.plan_current
    assert window.planner_parameters.values == before
    assert window.safety.value() == before['safety_width']
    window.redo()
    path = window.document.save(tmp_path / 'map/map.yaml')
    window.set_document(MapDocument.load(path))
    assert window.planner_parameters.values['veh_params.v_max'] == 4.
    assert window.planner_parameters.values['vel_calc_opts.vel_profile_conv_filt_window'] == 5
    assert window.safety.value() == .800123


def test_dialog_cancel_defaults_and_invalid_filter(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    original = dict(window.planner_parameters.values)
    dialog = PlannerParameterDialog(window.planner_parameters, window.project_parameters, window)
    dialog.editors['veh_params.v_max'].setValue(2.)
    dialog.reject()
    assert window.planner_parameters.values == original
    dialog.set_values(window.project_parameters)
    assert dialog.parameters().values == window.project_parameters.values
    errors = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args[-1]))
    dialog.editors['vel_calc_opts.vel_profile_conv_filt_window'].setValue(4)
    dialog.accept()
    assert errors and dialog.result_parameters is None


def test_quick_parameters_are_undoable(window):
    before = window.safety.value()
    window.safety.setValue(.9)
    assert window.planner_parameters.values['safety_width'] == .9
    window.reverse.setChecked(True)
    assert window.planner_parameters.values['reverse_mapping'] is True
    window.undo()
    assert not window.reverse.isChecked()
    window.undo()
    assert window.safety.value() == before
