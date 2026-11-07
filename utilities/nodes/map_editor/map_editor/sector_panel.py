"""Qt controls for the two independent, offline sector partitions."""
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QListWidget,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


def number(minimum, maximum, decimals=3):
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minimum, maximum)
    spin.setKeyboardTracking(False)
    return spin


class SectorPanel(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.selected = 0
        self.syncing = False
        layout = QVBoxLayout(self)
        self.kind = QComboBox()
        self.kind.addItem('Speed Sectors', 'speed')
        self.kind.addItem('Overtaking Sectors', 'overtaking')
        self.kind.currentIndexChanged.connect(self.kind_changed)
        layout.addWidget(self.kind)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.rebind = QPushButton('Use existing indices on current trajectory')
        self.rebind.setToolTip('Review the colored sections on the current trajectory before applying.')
        self.rebind.clicked.connect(lambda: self.accept_indices(False))
        layout.addWidget(self.rebind)
        self.reset = QPushButton('Reset to one section on current trajectory')
        self.reset.clicked.connect(lambda: self.accept_indices(True))
        layout.addWidget(self.reset)
        self.editors = QWidget()
        form = QFormLayout(self.editors)
        self.action = QComboBox()
        self.action.addItems(['Select / move boundaries', 'Add boundary', 'Delete boundary'])
        self.action.currentIndexChanged.connect(self.action_changed)
        form.addRow('Canvas action', self.action)
        self.limit = number(0, 1)
        self.yeet = number(0, 1000)
        self.spline = QSpinBox()
        self.spline.setRange(1, 1000000)
        self.spline.setKeyboardTracking(False)
        self.begin = QLabel()
        self.begin.setToolTip('Preserved in YAML. The current ot_interpolator does not read this parameter.')
        self.globals = [('global_limit', self.limit), ('yeet_factor', self.yeet),
                        ('spline_len', self.spline)]
        for key, spin in self.globals:
            form.addRow(key, spin)
            spin.valueChanged.connect(lambda value, name=key: self.global_changed(name, value))
        form.addRow('ot_sector_begin (preserved)', self.begin)
        self.sections = QListWidget()
        self.sections.setMaximumHeight(160)
        self.sections.currentRowChanged.connect(self.select)
        form.addRow('Sections', self.sections)
        self.indices = QLabel()
        form.addRow('Global Waypoint indices', self.indices)
        self.boundary = QSpinBox()
        self.boundary.setKeyboardTracking(False)
        self.boundary.valueChanged.connect(self.move_boundary)
        form.addRow('Start boundary index', self.boundary)
        self.scaling = number(0, 1)
        self.scaling.valueChanged.connect(self.scaling_changed)
        form.addRow('scaling', self.scaling)
        self.allowed = QCheckBox('Allow overtaking (ot_flag)')
        self.allowed.toggled.connect(self.allowed_changed)
        form.addRow(self.allowed)
        self.delete = QPushButton('Delete selected start boundary')
        self.delete.clicked.connect(lambda: self.delete_boundary(self.selected))
        form.addRow(self.delete)
        self.preview = QCheckBox('Preview scaled speed on raceline')
        self.preview.setChecked(True)
        self.preview.toggled.connect(self.preview_changed)
        form.addRow(self.preview)
        self.speed_info = QLabel()
        self.speed_info.setWordWrap(True)
        form.addRow(self.speed_info)
        layout.addWidget(self.editors)
        hint = QLabel('Select: click a section; drag a boundary circle to move it.\n'
                      'Add / Delete: click the raceline / boundary circle.\n'
                      'Right click: delete boundary. Waypoint 0 stays fixed.\n'
                      'Speed: section colors + blue → red speed preview (m/s).\n'
                      'Overtaking: green allowed / gray disabled. Ctrl+Z: undo.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()
        canvas = window.canvas
        canvas.sector_selected.connect(self.select)
        canvas.sector_add.connect(self.add_boundary)
        canvas.sector_delete.connect(self.delete_boundary)
        canvas.sector_move.connect(self.move_from_canvas)

    @property
    def current_kind(self):
        return self.kind.currentData()

    def kind_changed(self, _index):
        self.selected = 0
        self.window.canvas.sector_kind = self.current_kind
        self.refresh()
        self.window.canvas.refresh_overlays()

    def action_changed(self, index):
        self.window.canvas.sector_action = index

    def preview_changed(self, enabled):
        self.window.canvas.sector_speed_preview = enabled
        self.window.canvas.refresh_overlays()

    def refresh(self):
        doc = self.window.document
        if doc is None:
            return
        self.syncing = True
        try:
            kind = self.current_kind
            issues = doc.sector_issues(kind)
            structural = doc.sector_issues(kind, check_binding=False)
            current = doc.plan_current and self.window.process is None
            self.editors.setEnabled(current and not issues)
            self.reset.setVisible(bool(issues))
            self.reset.setEnabled(current)
            self.rebind.setVisible(bool(issues) and not structural)
            self.rebind.setEnabled(current)
            if not current:
                self.status.setText('Sector editing paused. Generate a trajectory matching the map and planner parameters.')
            elif issues:
                self.status.setText('\n'.join(issues) + '\nReview the colored preview before using existing indices; or reset.')
            else:
                self.status.setText('Sections match the current Global Waypoints. Indices include the lap closure point.')
            config = doc.sectors.get(kind)
            self.sections.clear()
            if config is None or structural:
                self.indices.setText('Invalid / unavailable configuration')
                return
            speed = kind == 'speed'
            for key, spin in self.globals:
                visible = (key == 'global_limit') == speed
                spin.setVisible(visible)
                self.editors.layout().labelForField(spin).setVisible(visible)
                if visible:
                    spin.setValue(config.params[key])
            self.begin.setVisible(not speed)
            self.editors.layout().labelForField(self.begin).setVisible(not speed)
            self.begin.setText(str(config.params.get('ot_sector_begin', '(not set)')))
            self.scaling.setVisible(speed)
            self.editors.layout().labelForField(self.scaling).setVisible(speed)
            self.allowed.setVisible(not speed)
            self.preview.setVisible(speed)
            self.speed_info.setVisible(speed)
            for i, section in enumerate(config.sections):
                value = f"×{section['scaling']:.3f}" if speed else ('allowed' if section['ot_flag'] else 'disabled')
                self.sections.addItem(f"{i}: {section['start']}…{section['end']}  {value}")
            self.selected = max(0, min(self.selected, len(config.sections) - 1))
            self.sections.setCurrentRow(self.selected)
            section = config.sections[self.selected]
            count = len(doc.plan['global_traj_wpnts_iqp']['wpnts'])
            self.indices.setText(f"{section['start']} … {section['end']} (inclusive; lap sentinel {count} supported)")
            lower = config.sections[self.selected - 1]['start'] + 1 if self.selected else 0
            self.boundary.setRange(lower, min(section['end'], count - 1))
            self.boundary.setValue(section['start'])
            self.boundary.setEnabled(self.selected > 0)
            self.delete.setEnabled(self.selected > 0)
            if speed:
                self.scaling.setValue(section['scaling'])
                factors = config.speed_factors(count)
                velocities = [p['vx_mps'] * factors[i] for i, p in
                              enumerate(doc.plan['global_traj_wpnts_iqp']['wpnts'])]
                self.speed_info.setText(f'Preview: {min(velocities):.2f}–{max(velocities):.2f} m/s.\n'
                                        'Runtime interpolation: 10 waypoints on each side of a boundary.\n'
                                        'Hover the raceline for waypoint speed.')
            else:
                self.allowed.setChecked(section['ot_flag'])
            self.window.canvas.sector_selected_index = self.selected
        finally:
            self.syncing = False

    def select(self, index):
        if self.syncing or index < 0:
            return
        self.selected = index
        self.refresh()
        self.window.canvas.refresh_overlays()

    def edit(self, callback):
        if self.syncing:
            return
        try:
            result = self.window.document.edit_sectors(self.current_kind, callback)
            if type(result) is int:
                self.selected = result
        except (ValueError, KeyError, TypeError) as error:
            self.window.error(str(error))
        self.window.canvas.refresh_overlays()
        self.window._changed()

    def global_changed(self, key, value):
        def update(config):
            config.params[key] = value
            if key == 'global_limit':
                for section in config.sections:
                    section['scaling'] = min(section['scaling'], value)
        self.edit(update)

    def scaling_changed(self, value):
        def update(config):
            config.sections[self.selected]['scaling'] = value
            for section in config.sections:
                section['scaling'] = min(section['scaling'], config.params['global_limit'])
        self.edit(update)

    def allowed_changed(self, value):
        self.edit(lambda config: config.sections[self.selected].update(ot_flag=value))

    def add_boundary(self, index):
        self.edit(lambda config: config.split(index))

    def delete_boundary(self, index):
        self.edit(lambda config: config.delete_boundary(index))

    def move_from_canvas(self, section, index):
        self.selected = section
        self.edit(lambda config: config.move_boundary(section, index,
                  len(self.window.document.plan['global_traj_wpnts_iqp']['wpnts'])))

    def move_boundary(self, index):
        if not self.syncing:
            self.move_from_canvas(self.selected, index)

    def accept_indices(self, reset):
        try:
            self.window.document.accept_sector_indices(self.current_kind, reset=reset)
            self.selected = 0
        except (ValueError, KeyError, TypeError) as error:
            self.window.error(str(error))
        self.window.canvas.refresh_overlays()
        self.window._changed()
