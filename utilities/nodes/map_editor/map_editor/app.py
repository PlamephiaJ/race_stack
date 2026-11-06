"""Standalone Qt application entry point (no ROS executor in the GUI)."""
import argparse
import configparser
from copy import deepcopy
import faulthandler
import json
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QProcess, QSignalBlocker, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QPlainTextEdit, QVBoxLayout, QWidget,
)
import yaml

from .canvas import MapCanvas
from .document import MapDocument
from .overlays import LAYERS, plan_layers
from .parameter_dialog import PlannerParameterDialog
from .parameters import METADATA_KEY, PlannerParameters
from .paths import default_maps_root, planner_config
from .qt_runtime import check_pyside_compatibility, check_xcb_dependencies, prepare_qt_environment


def real_spin(value, minimum, maximum, decimals=3):
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


class NewMapDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('New map')
        form = QFormLayout(self)
        self.width, self.height = QSpinBox(), QSpinBox()
        for spin in (self.width, self.height):
            spin.setRange(16, 8000)
            spin.setValue(800)
        self.resolution = real_spin(.05, .001, 10., 4)
        self.origin_x = real_spin(0., -1e6, 1e6)
        self.origin_y = real_spin(0., -1e6, 1e6)
        self.origin_yaw = real_spin(0., -3.141593, 3.141593, 6)
        self.background = QComboBox()
        self.background.addItem('Occupied (black) — draw a free track', 0)
        self.background.addItem('Free (white) — draw obstacles', 255)
        form.addRow('Width (pixels)', self.width)
        form.addRow('Height (pixels)', self.height)
        form.addRow('Resolution (m/pixel)', self.resolution)
        form.addRow('Origin x (m)', self.origin_x)
        form.addRow('Origin y (m)', self.origin_y)
        form.addRow('Origin yaw (rad)', self.origin_yaw)
        form.addRow('Background', self.background)
        self.size_label = QLabel()
        form.addRow('Physical size', self.size_label)
        for spin in (self.width, self.height, self.resolution):
            spin.valueChanged.connect(self.update_size)
        self.update_size()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def update_size(self):
        self.size_label.setText(f'{self.width.value()*self.resolution.value():.2f} × '
                                f'{self.height.value()*self.resolution.value():.2f} m')

    def document(self):
        return MapDocument.new(self.width.value(), self.height.value(), self.resolution.value(),
                               (self.origin_x.value(), self.origin_y.value(), self.origin_yaw.value()),
                               self.background.currentData())


class EditorWindow(QMainWindow):
    def __init__(self, maps_root=None):
        super().__init__()
        self.resize(1380, 900)
        self.maps_root = Path(maps_root) if maps_root else default_maps_root()
        self.canvas = MapCanvas()
        self.setCentralWidget(self.canvas)
        self.document = None
        self.process = None
        self.job_directory = None
        self.job_document = None
        self.job_fingerprint = None
        self.cancelled = False
        self.plan_pending = False
        self.parameter_warning = None
        try:
            self.project_parameters = PlannerParameters.from_config(planner_config())
        except (OSError, ValueError, KeyError, TypeError, ImportError, LookupError,
                configparser.Error, yaml.YAMLError) as error:
            # Editing maps remains available before the ROS planner is installed.
            self.project_parameters = PlannerParameters()
            self.parameter_warning = f'Could not load project planner defaults: {error}'
        self.planner_parameters = PlannerParameters(self.project_parameters.values)
        self.actions = {}
        self._menus()
        self._library()
        self._controls()
        self._logs()
        if self.parameter_warning:
            self.logs.appendPlainText(self.parameter_warning)
        self.canvas.changed.connect(self._changed)
        self.canvas.position.connect(lambda text: self.statusBar().showMessage(text))
        self.set_document(MapDocument.new(800, 800, .05))

    def _action(self, menu, text, callback, shortcut=None, key=None):
        action = QAction(text, self)
        action.triggered.connect(lambda _checked=False: callback())
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)
        if key:
            self.actions[key] = action
        return action

    def _menus(self):
        file_menu = self.menuBar().addMenu('File')
        self._action(file_menu, 'New map…', self.new_map, QKeySequence.StandardKey.New)
        self._action(file_menu, 'Open YAML…', self.open_dialog, QKeySequence.StandardKey.Open)
        self._action(file_menu, 'Save map + current trajectory', self.save, QKeySequence.StandardKey.Save)
        self._action(file_menu, 'Save as new map…', lambda: self.save(save_as=True),
                     QKeySequence.StandardKey.SaveAs)
        self._action(file_menu, 'Quit', self.close, QKeySequence.StandardKey.Quit)
        edit = self.menuBar().addMenu('Edit')
        self._action(edit, 'Undo', self.undo, QKeySequence.StandardKey.Undo, 'undo')
        self._action(edit, 'Redo', self.redo, QKeySequence.StandardKey.Redo, 'redo')
        self._action(edit, 'Global Planner parameters…', self.edit_planner_parameters)
        view = self.menuBar().addMenu('View')
        self._action(view, 'Fit map', self.canvas.fit_map, 'F')
        self._action(view, 'Actual pixels', self.canvas.resetTransform, '1')
        help_menu = self.menuBar().addMenu('Help')
        self._action(help_menu, 'Workflow / controls', self.help)

    def _dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _library(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.root_label = QLabel(str(self.maps_root))
        self.root_label.setWordWrap(True)
        layout.addWidget(self.root_label)
        row = QHBoxLayout()
        for text, callback in (('Choose maps folder…', self.choose_root), ('Refresh', self.refresh_library)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.map_list = QListWidget()
        self.map_list.itemDoubleClicked.connect(lambda item: self.open_map(item.data(Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.map_list)
        self._dock('Map library', widget, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.refresh_library()

    def choose_root(self):
        path = QFileDialog.getExistingDirectory(self, 'Select stack_master/maps', str(self.maps_root))
        if path:
            self.maps_root = Path(path)
            self.refresh_library()

    def refresh_library(self):
        self.root_label.setText(str(self.maps_root))
        self.map_list.clear()
        for path in sorted([*self.maps_root.glob('**/*.yaml'), *self.maps_root.glob('**/*.yml')]):
            try:
                metadata = yaml.safe_load(path.read_text())
                if not isinstance(metadata, dict) or not {'image', 'resolution', 'origin'} <= metadata.keys():
                    continue
            except (OSError, ValueError, yaml.YAMLError):
                continue
            item = QListWidgetItem(str(path.relative_to(self.maps_root)))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.map_list.addItem(item)

    def _controls(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self.tool = QComboBox()
        self.tool.addItems(['Brush', 'Eraser', 'Line', 'Rectangle', 'Polygon',
                            'Remove component', 'Initial pose', 'Pan'])
        self.tool.currentTextChanged.connect(self.canvas.set_tool)
        form.addRow('Tool', self.tool)
        self.material = QComboBox()
        for label, value in (('Obstacle (black)', 0), ('Free track (white)', 255), ('Unknown (gray)', 205)):
            self.material.addItem(label, value)
        self.material.currentIndexChanged.connect(
            lambda _: setattr(self.canvas, 'material', self.material.currentData()))
        form.addRow('Paint', self.material)
        self.brush = QSpinBox()
        self.brush.setRange(1, 1000)
        self.brush.setValue(self.canvas.brush_size)
        self.brush.valueChanged.connect(lambda value: setattr(self.canvas, 'brush_size', value))
        form.addRow('Brush / line width (px)', self.brush)
        self.filled = QCheckBox('Fill rectangles and polygons')
        self.filled.setChecked(True)
        self.filled.toggled.connect(lambda checked: setattr(self.canvas, 'filled', checked))
        form.addRow(self.filled)
        self.noise_area = QSpinBox()
        self.noise_area.setRange(1, 100000)
        self.noise_area.setValue(20)
        form.addRow('Noise max area (px²)', self.noise_area)
        clean = QPushButton('Remove small obstacle components')
        clean.clicked.connect(self.clean_noise)
        form.addRow(clean)
        layout.addLayout(form)
        hint = QLabel('Wheel: zoom · Middle drag: pan\nPolygon: click vertices, Enter / right click to finish\nEsc: cancel stroke / shape\nInitial pose: drag from start along travel direction')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(QLabel('Global Planner — offline'))
        planning = QFormLayout()
        self.safety = real_spin(.7, .05, 20., 6)
        self.safety_sp = real_spin(.7, .05, 20., 6)
        planning.addRow('Safety width (m)', self.safety)
        planning.addRow('Shortest path width (m)', self.safety_sp)
        self.reverse = QCheckBox('Reverse travel direction')
        planning.addRow(self.reverse)
        layout.addLayout(planning)
        for spin in (self.safety, self.safety_sp):
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._basic_parameters_changed)
        self.reverse.toggled.connect(self._basic_parameters_changed)
        parameters_button = QPushButton('Global Planner parameters…')
        parameters_button.clicked.connect(self.edit_planner_parameters)
        layout.addWidget(parameters_button)
        self.generate = QPushButton('Generate trajectories')
        self.generate.clicked.connect(self.start_planner)
        layout.addWidget(self.generate)
        self.cancel = QPushButton('Cancel generation')
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_planner)
        layout.addWidget(self.cancel)
        self.plan_status = QLabel()
        self.plan_status.setWordWrap(True)
        layout.addWidget(self.plan_status)
        for label, color in LAYERS.items():
            check = QCheckBox(label)
            check.setChecked(True)
            check.setStyleSheet(f'color: {color};')
            check.toggled.connect(lambda checked, name=label: self.toggle_layer(name, checked))
            layout.addWidget(check)
        layout.addStretch()
        self._dock('Drawing / trajectory preview', widget, Qt.DockWidgetArea.RightDockWidgetArea)

    def _logs(self):
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(5000)
        dock = self._dock('Planner log', self.logs, Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setMaximumHeight(220)

    def toggle_layer(self, label, checked):
        if checked:
            self.canvas.visible_layers.add(label)
        else:
            self.canvas.visible_layers.discard(label)
        self.canvas.refresh_overlays()

    def set_document(self, document):
        # Validate a saved profile before replacing the current document.
        self._document_parameters(document)
        self.document = document
        self.plan_pending = False
        self.canvas.set_document(document)
        self._changed()

    def _changed(self):
        doc = self.document
        if doc is None:
            return
        self.planner_parameters = self._document_parameters(doc)
        self._sync_parameter_controls()
        name = doc.path.stem if doc.path else 'Untitled'
        unsaved = doc.dirty or (self.plan_pending and doc.plan_current)
        self.setWindowTitle(f'{name}{" *" if unsaved else ""} — Race Stack Map Editor')
        self.actions['undo'].setEnabled(bool(doc.undo_stack))
        self.actions['redo'].setEnabled(bool(doc.redo_stack))
        if self.process is not None:
            self.plan_status.setText('Generating snapshot in background…')
        elif doc.plan_current:
            info = doc.plan.get('map_info_str', {}).get('data', '')
            prefix = 'Trajectory matches map. ' + ('Save to export.\n' if self.plan_pending else '\n')
            self.plan_status.setText(prefix + info)
        elif doc.plan is not None:
            self.plan_status.setText('Map or planner parameters changed. Old trajectory hidden; generate again.')
        else:
            self.plan_status.setText('Draw a closed, sufficiently wide loop, then generate.')

    def _document_parameters(self, document):
        if METADATA_KEY in document.metadata:
            return PlannerParameters.from_profile(document.metadata[METADATA_KEY], self.project_parameters)
        return PlannerParameters(self.project_parameters.values)

    def _sync_parameter_controls(self):
        values = self.planner_parameters.values
        with QSignalBlocker(self.safety), QSignalBlocker(self.safety_sp), QSignalBlocker(self.reverse):
            self.safety.setValue(values['safety_width'])
            self.safety_sp.setValue(values['safety_width_sp'])
            self.reverse.setChecked(values['reverse_mapping'])

    def _basic_parameters_changed(self, _value=None):
        if self.document is None:
            return
        values = dict(self.planner_parameters.values, safety_width=self.safety.value(),
                      safety_width_sp=self.safety_sp.value(), reverse_mapping=self.reverse.isChecked())
        try:
            self.apply_planner_parameters(PlannerParameters(values))
        except ValueError as error:
            self._sync_parameter_controls()
            self.error(str(error))

    def apply_planner_parameters(self, parameters, record=False):
        parameters.validate()
        if (not record and METADATA_KEY not in self.document.metadata
                and parameters.values == self._document_parameters(self.document).values):
            return
        self.canvas.cancel_gesture()
        metadata = deepcopy(self.document.metadata)
        metadata[METADATA_KEY] = parameters.profile()
        self.document.commit(metadata=metadata)
        self.canvas.refresh_overlays()
        self._changed()

    def edit_planner_parameters(self):
        dialog = PlannerParameterDialog(self.planner_parameters, self.project_parameters, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.apply_planner_parameters(dialog.result_parameters)
        finally:
            dialog.deleteLater()

    def _confirm_save(self):
        if not self.document.dirty and not (self.plan_pending and self.document.plan_current):
            return True
        answer = QMessageBox.question(self, 'Unsaved work', 'Save map and current trajectory before continuing?',
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                      | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def new_map(self):
        if not self._confirm_save():
            return
        dialog = NewMapDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.set_document(dialog.document())
            except (ValueError, MemoryError) as error:
                self.error(str(error))

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open occupancy map', str(self.maps_root), 'Maps (*.yaml *.yml)')
        if path:
            self.open_map(path)

    def open_map(self, path):
        if not self._confirm_save():
            return
        try:
            self.set_document(MapDocument.load(path))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            self.error(f'Could not open map:\n{error}')

    def save(self, save_as=False):
        self.canvas.cancel_gesture()
        path = self.document.path
        if path is None or save_as:
            # Choosing a map name creates maps/name/name.{png,yaml} automatically.
            chosen, _ = QFileDialog.getSaveFileName(
                self, 'Save map (creates a map-name subfolder)',
                str(self.maps_root / (path.stem if path else 'custom_map')), 'Map YAML (*.yaml)')
            if not chosen:
                return False
            selected = Path(chosen).with_suffix('.yaml')
            path = selected if selected.parent.name == selected.stem else selected.parent / selected.stem / selected.name
            if path.exists() and path != self.document.path:
                answer = QMessageBox.question(self, 'Replace map', f'Replace {path}?',
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    return False
        try:
            self.document.save(path)
            self.plan_pending = False
            self.refresh_library()
            self._changed()
            self.statusBar().showMessage(f'Saved {path}', 6000)
            return True
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.error(f'Could not save map:\n{error}')
            return False

    def undo(self):
        self.canvas.cancel_gesture()
        self.document.undo()
        self.canvas.refresh()
        self._changed()

    def redo(self):
        self.canvas.cancel_gesture()
        self.document.redo()
        self.canvas.refresh()
        self._changed()

    def clean_noise(self):
        self.canvas.cancel_gesture()
        self.document.clean_noise(self.noise_area.value())
        self.canvas.refresh()
        self._changed()

    def start_planner(self):
        if self.process is not None:
            return
        self.canvas.cancel_gesture()
        try:
            config = planner_config()
            self.job_directory = tempfile.TemporaryDirectory(prefix='race-map-editor-')
            root = Path(self.job_directory.name)
            config = self.planner_parameters.snapshot(config, root / 'config')
            # A separate document prevents saving or changing the user's map during planning.
            snapshot = MapDocument(self.document.pixels, self.document.metadata)
            snapshot.metadata[METADATA_KEY] = self.planner_parameters.profile()
            snapshot.save(root / 'input/map.yaml')
            self.job_document = self.document
            self.job_fingerprint = self.document.fingerprint
            arguments = ['-u', '-m', 'map_editor.planner_worker', '--map', str(root / 'input/map.yaml'),
                         '--config', str(config), '--work-dir', str(root / 'output'),
                         '--safety-width', str(self.planner_parameters.values['safety_width']),
                         '--safety-width-sp', str(self.planner_parameters.values['safety_width_sp'])]
            if self.planner_parameters.values['reverse_mapping']:
                arguments.append('--reverse')
            self.logs.clear()
            self.logs.appendPlainText('Starting existing Global Planner on an isolated map snapshot…')
            self.logs.appendPlainText('Planner parameters:\n' + yaml.safe_dump(
                self.planner_parameters.profile(), sort_keys=False))
            self.cancelled = False
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self._read_log)
            self.process.finished.connect(self._planner_finished)
            self.process.errorOccurred.connect(self._planner_error)
            self.generate.setEnabled(False)
            self.cancel.setEnabled(True)
            self._changed()
            self.process.start(sys.executable, arguments)
        except (OSError, ValueError, KeyError, TypeError, ImportError, LookupError,
                configparser.Error, yaml.YAMLError) as error:
            self._cleanup_job()
            self.error(f'Could not start planner:\n{error}')

    def _read_log(self):
        if self.process is not None:
            text = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
            cursor = self.logs.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self.logs.setTextCursor(cursor)
            self.logs.ensureCursorVisible()

    def _planner_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            message = self.process.errorString()
            self._cleanup_job()
            self.error('Failed to start Python planner worker: ' + message)

    def _planner_finished(self, exit_code, exit_status):
        self._read_log()
        try:
            if self.cancelled:
                self.logs.appendPlainText('\nGeneration cancelled; map unchanged.')
            elif exit_code != 0 or exit_status != QProcess.ExitStatus.NormalExit:
                self.error('Global Planner failed. See Planner log for dependency or track geometry errors.')
            elif self.document is not self.job_document or self.document.fingerprint != self.job_fingerprint:
                self.logs.appendPlainText('\nMap or parameters changed during generation; '
                                          'discarded outdated result. Generate again.')
            else:
                output = Path(self.job_directory.name) / 'output/global_waypoints.json'
                plan = json.loads(output.read_text())
                plan_layers(plan)
                # Record the effective defaults too, so a saved trajectory is reproducible.
                self.apply_planner_parameters(self.planner_parameters, record=True)
                self.document.attach_plan(plan)
                self.plan_pending = True
                self.canvas.refresh_overlays()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.error(f'Could not load planner result:\n{error}')
        finally:
            self._cleanup_job()

    def cancel_planner(self):
        if self.process is not None:
            self.cancelled = True
            self.process.kill()

    def _cleanup_job(self):
        if self.process is not None:
            self.process.deleteLater()
        self.process = None
        if self.job_directory is not None:
            self.job_directory.cleanup()
        self.job_directory = None
        self.job_document = None
        self.generate.setEnabled(True)
        self.cancel.setEnabled(False)
        self._changed()

    def error(self, text):
        QMessageBox.critical(self, 'Map Editor', text)

    def help(self):
        message = (
            '1. Create a map or open a YAML from the map library.\n'
            '2. Draw a closed free-space loop with occupied inner and outer walls.\n'
            '3. Optionally set Initial pose by dragging along the travel direction.\n'
            '4. Set Global Planner parameters, generate, toggle layers and edit again.\n'
            '5. Save exports PNG + YAML + matching global_waypoints.json.\n\n'
            'Brush width and noise area are in pixels. Eraser paints free space.\n'
            'Remove component deletes the entire clicked connected obstacle.\n'
            'Map / parameter changes hide outdated trajectories; undo restores matching preview.\n'
            'Planner profiles are saved in map YAML and can be imported / exported.\n'
            'Real-car localization still requires corresponding localization data (e.g. PBSTREAM).')
        QMessageBox.information(self, 'Map Editor workflow', message)

    def closeEvent(self, event):
        if not self._confirm_save():
            event.ignore()
            return
        if self.process is not None:
            self.process.blockSignals(True)
            self.process.kill()
            self.process.waitForFinished(3000)
            self._cleanup_job()
        event.accept()


def main(argv=None):
    # Native Qt/OpenCV aborts bypass Python exception handlers. Preserve a Python
    # stack in the terminal so a desktop-only crash can be located.
    faulthandler.enable()
    parser = argparse.ArgumentParser(description='Independent Race Stack occupancy map editor')
    parser.add_argument('--maps-dir', type=Path, help='Override stack_master/maps directory')
    parser.add_argument('--map', type=Path, help='Open a specific map YAML at startup')
    args = parser.parse_args(argv)
    try:
        check_pyside_compatibility()
        plugins = prepare_qt_environment()
        check_xcb_dependencies(plugins)
    except RuntimeError as error:
        print(f'Map Editor: {error}', file=sys.stderr)
        return 1
    application = QApplication([sys.argv[0]])
    application.setApplicationName('Race Stack Map Editor')
    window = EditorWindow(args.maps_dir)
    if args.map:
        # No prompt for the initial empty document.
        try:
            window.set_document(MapDocument.load(args.map))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            window.error(str(error))
    window.show()
    return application.exec()


if __name__ == '__main__':
    sys.exit(main())
