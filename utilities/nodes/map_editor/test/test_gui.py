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
def window(app, monkeypatch):
    monkeypatch.setenv('RACECAR_VERSION', 'SIM')
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


def test_vehicle_selection_pauses_sectors_and_survives_dialog_undo_save(window, tmp_path):
    example = Path(__file__).resolve().parents[4] / 'stack_master/maps/GLC_smile_small/GLC_smile_small.yaml'
    window.set_document(MapDocument.load(example))
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('BLUE'))
    parameters = window.planner_parameters
    assert parameters.vehicle['name'] == 'BLUE'
    from map_editor.paths import planner_config
    import yaml
    model = yaml.safe_load((planner_config().parent / 'BLUE/car_model.yaml').read_text())
    assert parameters.values['veh_params.mass'] == model['m']
    assert parameters.values['veh_params.v_max'] == pytest.approx(23250 / 4176.3)
    assert not window.document.plan_current and not window.sector_panel.editors.isEnabled()
    dialog = PlannerParameterDialog(parameters, window.project_parameters, window)
    dialog.accept()
    assert dialog.result_parameters.profile() == parameters.profile()  # Preserve full conversion precision.
    window.apply_planner_parameters(dialog.result_parameters)
    assert len(window.document.undo_stack) == 1  # Unchanged dialog adds no history entry.
    old_safety = window.safety.value()
    window.safety.setValue(old_safety + .1)
    assert window.planner_parameters.vehicle == parameters.vehicle
    window.undo()
    window.undo()
    assert window.planner_parameters.vehicle is None and window.document.plan_current
    window.redo()
    assert window.planner_parameters.vehicle['name'] == 'BLUE'
    path = window.document.save(tmp_path / 'blue_track/blue_track.yaml')
    window.set_document(MapDocument.load(path))
    assert window.vehicle_choice.currentData() == 'BLUE'
    assert window.planner_parameters.vehicle == parameters.vehicle
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('NUC5'))
    assert window.planner_parameters.values['veh_params.mass'] == 2.783
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('SIM'))
    assert window.planner_parameters.vehicle is None
    assert window.planner_parameters.values['veh_params.mass'] == 3.518
    assert window.planner_parameters.config_vehicle == 'SIM'


def test_vehicle_reload_error_keeps_previous_selection(window, monkeypatch):
    import map_editor.vehicles as vehicles
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('BLUE'))
    before = window.planner_parameters.profile()
    errors = []
    window.error = errors.append

    def fail(_directory):
        raise ValueError('invalid wheelbase')

    monkeypatch.setattr(vehicles, 'vehicle_settings', fail)
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('NUC6'))
    assert errors and 'wheelbase' in errors[-1]
    assert window.planner_parameters.profile() == before
    assert window.vehicle_choice.currentData() == 'BLUE'


def test_vehicle_reload_updates_snapshot_only_when_requested(window, monkeypatch):
    import map_editor.vehicles as vehicles
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('BLUE'))
    old_mass = window.planner_parameters.values['veh_params.mass']
    values, limits = vehicles.vehicle_settings(Path(__file__).resolve().parents[4] / 'stack_master/config/BLUE')
    values['veh_params.mass'] = old_mass + .1
    monkeypatch.setattr(vehicles, 'vehicle_settings', lambda _directory: (values, limits))
    window._changed()
    assert window.planner_parameters.values['veh_params.mass'] == old_mass
    QTest.mouseClick(window.reload_vehicle, Qt.MouseButton.LeftButton)
    assert window.planner_parameters.values['veh_params.mass'] == pytest.approx(old_mass + .1)
    window.undo()
    assert window.planner_parameters.values['veh_params.mass'] == old_mass


def test_vehicle_ini_length_reload_dialog_defaults_and_job_source(window, tmp_path, monkeypatch):
    import configparser
    import json
    import shutil
    import map_editor.app as editor_app
    import map_editor.paths as paths
    source = paths.config_root()
    for name, length in (('BLUE', .61), ('NUC5', .49)):
        directory = tmp_path / name
        shutil.copytree(source / name, directory)
        path = directory / 'racecar_f110.ini'
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path)
        values = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
        values.update(length=length, g=9.7)
        parser.set('GENERAL_OPTIONS', 'veh_params', json.dumps(values))
        with path.open('w') as stream:
            parser.write(stream)
    resolve = lambda name=None: tmp_path / (name or 'BLUE')
    monkeypatch.setattr(editor_app, 'planner_config', resolve)
    monkeypatch.setattr(paths, 'planner_config', resolve)
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('NUC5'))
    assert window.planner_parameters.values['veh_params.length'] == .49
    window.vehicle_choice.setCurrentIndex(window.vehicle_choice.findData('BLUE'))
    assert window.planner_parameters.values['veh_params.length'] == .61
    dialog = PlannerParameterDialog(window.planner_parameters, window.project_parameters, window)
    dialog.editors['veh_params.length'].setValue(.65)
    dialog.load_vehicle_defaults()
    assert dialog.parameters().values['veh_params.length'] == .61
    assert dialog.parameters().config_vehicle == 'BLUE'
    window.error = lambda error: pytest.fail(error)
    window.start_planner()
    try:
        assert window.process is not None
        parser.read(Path(window.job_directory.name) / 'config/racecar_f110.ini')
        actual = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
        assert actual['length'] == .61 and actual['g'] == 9.7
    finally:
        window.cancel_planner()
        for _ in range(100):
            if window.process is None:
                break
            QTest.qWait(50)
        assert window.process is None
