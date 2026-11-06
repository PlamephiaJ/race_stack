"""Typed Qt controls for the parameters consumed by the offline planner."""
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)
import yaml

from .parameters import PARAMETERS, PlannerParameters


class PlannerParameterDialog(QDialog):
    def __init__(self, parameters, defaults, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Global Planner parameters')
        self.resize(690, 490)
        self.defaults = defaults
        self.result_parameters = None
        self.editors = {}
        layout = QVBoxLayout(self)
        note = QLabel('Apply changes, then Generate trajectories. Settings are saved with the map.\n'
                      'Dynamics CSV names are relative to the project configuration’s veh_dyn_info folder.')
        note.setWordWrap(True)
        layout.addWidget(note)
        tabs = QTabWidget()
        forms = {}
        for parameter in PARAMETERS:
            if parameter.tab not in forms:
                page = QWidget()
                forms[parameter.tab] = QFormLayout(page)
                tabs.addTab(page, parameter.tab)
            if parameter.kind == 'bool':
                editor = QCheckBox()
            elif parameter.kind == 'file':
                editor = QLineEdit()
            elif parameter.kind in ('int', 'optional_int'):
                editor = QSpinBox()
                editor.setRange(0 if parameter.kind == 'optional_int' else int(parameter.minimum),
                                int(parameter.maximum))
                if parameter.kind == 'optional_int':
                    editor.setSpecialValueText('Disabled')
                editor.setKeyboardTracking(False)
            else:
                editor = QDoubleSpinBox()
                editor.setDecimals(6)
                editor.setRange(parameter.minimum, parameter.maximum)
                editor.setKeyboardTracking(False)
            editor.setToolTip(parameter.key)
            editor.setObjectName(parameter.key)
            self.editors[parameter.key] = editor
            forms[parameter.tab].addRow(parameter.label, editor)
        layout.addWidget(tabs)
        row = QHBoxLayout()
        for title, callback in (('Load project defaults', lambda: self.set_values(self.defaults)),
                                ('Import profile…', self.import_profile), ('Export profile…', self.export_profile)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.set_values(parameters)

    def set_values(self, parameters):
        for parameter in PARAMETERS:
            value = parameters.values[parameter.key]
            editor = self.editors[parameter.key]
            if parameter.kind == 'bool':
                editor.setChecked(value)
            elif parameter.kind == 'file':
                editor.setText(value)
            else:
                editor.setValue(0 if value is None else value)

    def parameters(self):
        values = {}
        for parameter in PARAMETERS:
            editor = self.editors[parameter.key]
            if parameter.kind == 'bool':
                value = editor.isChecked()
            elif parameter.kind == 'file':
                value = editor.text().strip()
            else:
                value = editor.value()
                if parameter.kind == 'optional_int' and value == 0:
                    value = None
            values[parameter.key] = value
        return PlannerParameters(values)

    def accept(self):
        try:
            self.result_parameters = self.parameters()
        except ValueError as error:
            QMessageBox.warning(self, 'Invalid planner parameters', str(error))
            return
        super().accept()

    def import_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import planner profile', '', 'Planner profile (*.yaml *.yml)')
        if path:
            try:
                self.set_values(PlannerParameters.load(path, self.defaults))
            except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
                QMessageBox.warning(self, 'Could not import profile', str(error))

    def export_profile(self):
        try:
            parameters = self.parameters()
            path, _ = QFileDialog.getSaveFileName(self, 'Export planner profile', 'planner_profile.yaml',
                                                  'Planner profile (*.yaml)')
            if path:
                parameters.export(Path(path).with_suffix('.yaml'))
        except (OSError, ValueError, yaml.YAMLError) as error:
            QMessageBox.warning(self, 'Could not export profile', str(error))
