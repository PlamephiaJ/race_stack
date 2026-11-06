"""Sustained native interaction tests run in children to contain native aborts."""
import os
import subprocess
import sys

import pytest

pytest.importorskip('PySide6')


def run_child(code, platform='offscreen', timeout=45):
    environment = dict(os.environ, QT_QPA_PLATFORM=platform,
                       PYTHONMALLOC='debug', PYTHONFAULTHANDLER='1')
    result = subprocess.run([sys.executable, '-u', '-c', code], env=environment,
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return result.stdout


def test_repeated_signals_preserve_boolean_reference_counts():
    output = run_child('''
import sys
from map_editor.qt_runtime import check_pyside_compatibility
check_pyside_compatibility()
from PySide6.QtCore import QObject, Signal
class Sender(QObject):
    position = Signal(str)
    changed = Signal()
    enabled = Signal(bool)
sender = Sender()
sender.position.connect(lambda text: None)
sender.changed.connect(lambda: None)
sender.enabled.connect(lambda value: None)
def emit(index):
    sender.position.emit(f'Pixel {index}, {index+1} | Map {index*.05:.3f}, {index*.1:.3f} m')
    sender.changed.emit()
    sender.enabled.emit(index % 2 == 0)
for index in range(20):
    emit(index)
before = (sys.getrefcount(True), sys.getrefcount(False))
for index in range(10000):
    emit(index)
after = (sys.getrefcount(True), sys.getrefcount(False))
assert before == after, (before, after)
print('30000 signal emissions: stable bool reference counts', before, after)
''')
    assert 'stable bool reference counts' in output


def test_sustained_drawing_undo_and_modal_parameter_dialogs():
    platform = 'xcb' if os.environ.get('MAP_EDITOR_TEST_DESKTOP') == '1' else 'offscreen'
    output = run_child('''
import gc
from map_editor.qt_runtime import check_pyside_compatibility, prepare_qt_environment, check_xcb_dependencies
check_pyside_compatibility()
check_xcb_dependencies(prepare_qt_environment())
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QTabWidget
from map_editor.app import EditorWindow
from map_editor.document import MapDocument
application = QApplication([])
print('platform=' + application.platformName(), flush=True)
window = EditorWindow()
window.show()
application.processEvents()
for width in (800, 801, 1501):
    window.set_document(MapDocument.new(width, 503, .05))
    canvas = window.canvas
    canvas.material = 255
    for index in range(70):
        canvas.set_tool(('Brush', 'Eraser', 'Line', 'Rectangle')[index % 4])
        start, end = (20 + index, 30 + index), (400 + index, 300 + index)
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton,
                         pos=canvas.mapFromScene(QPointF(*start)))
        for step in range(1, 9):
            x = start[0] + (end[0] - start[0]) * step / 8
            y = start[1] + (end[1] - start[1]) * step / 8
            QTest.mouseMove(canvas.viewport(), canvas.mapFromScene(QPointF(x, y)))
            application.processEvents()
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton,
                           pos=canvas.mapFromScene(QPointF(*end)))
        if index % 5 == 0:
            window.undo()
            window.redo()
        if index % 20 == 0:
            gc.collect()
    print('drawing complete', width, flush=True)
for index in range(30):
    def close_dialog(index=index):
        dialog = application.activeModalWidget()
        assert isinstance(dialog, QDialog)
        tabs = dialog.findChild(QTabWidget)
        for tab in range(tabs.count()):
            tabs.setCurrentIndex(tab)
        dialog.editors['veh_params.v_max'].setValue(3. + index % 5)
        dialog.accept() if index % 2 else dialog.reject()
    QTimer.singleShot(20, close_dialog)
    window.edit_planner_parameters()
    application.processEvents()
QTest.qWait(1000)
window.document.saved_fingerprint = window.document.fingerprint
window.plan_pending = False
window.close()
application.processEvents()
print('210 strokes, 1680 mouse moves, 30 modal dialogs: stable', flush=True)
''', platform=platform)
    assert '30 modal dialogs: stable' in output
    assert 'platform=' + platform in output
