"""Regressions for OpenCV wheel pollution and readable X11 dependency failures."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('PySide6')

from map_editor.qt_runtime import check_pyside_compatibility, check_xcb_dependencies, prepare_qt_environment


def test_known_signal_refcount_bug_is_rejected_before_creating_widgets(monkeypatch):
    import PySide6
    from collections import namedtuple
    import map_editor.qt_runtime as runtime
    version = namedtuple('VersionInfo', 'major minor micro releaselevel serial')
    monkeypatch.setattr(PySide6, '__version__', '6.12.0')
    monkeypatch.setattr(runtime.sys, 'version_info', version(3, 10, 12, 'final', 0))
    with pytest.raises(RuntimeError, match='PYSIDE-3474'):
        check_pyside_compatibility()


def test_verified_pyside_release_is_accepted(monkeypatch):
    import PySide6
    monkeypatch.setattr(PySide6, '__version__', '6.11.2')
    check_pyside_compatibility()


def test_opencv_paths_removed_without_losing_user_settings(monkeypatch):
    from PySide6.QtCore import QCoreApplication, QLibraryInfo
    # Trigger cv2's own environment mutation before simulating user environment.
    import cv2  # noqa: F401
    monkeypatch.setenv('QT_QPA_PLATFORM_PLUGIN_PATH', '/usr/local/lib/python3.10/dist-packages/cv2/qt/plugins')
    monkeypatch.setenv('QT_QPA_FONTDIR', '/usr/local/lib/python3.10/dist-packages/cv2/qt/fonts')
    monkeypatch.setenv('QT_PLUGIN_PATH', '/opt/custom-qt/plugins:/old/cv2/qt/plugins')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    before = QCoreApplication.libraryPaths()
    try:
        plugins = prepare_qt_environment()
        assert 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ
        assert 'QT_QPA_FONTDIR' not in os.environ
        assert os.environ['QT_PLUGIN_PATH'] == '/opt/custom-qt/plugins'
        assert os.environ['QT_QPA_PLATFORM'] == 'offscreen'
        assert plugins == Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
        assert str(plugins) in QCoreApplication.libraryPaths()
        assert not any('/cv2/qt/' in path for path in QCoreApplication.libraryPaths())
    finally:
        QCoreApplication.setLibraryPaths(before)


def test_explicit_non_opencv_paths_preserved(monkeypatch):
    import cv2  # noqa: F401
    monkeypatch.setenv('QT_QPA_PLATFORM_PLUGIN_PATH', '/opt/custom/platforms')
    monkeypatch.setenv('QT_QPA_FONTDIR', '/opt/custom/fonts')
    prepare_qt_environment()
    assert os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] == '/opt/custom/platforms'
    assert os.environ['QT_QPA_FONTDIR'] == '/opt/custom/fonts'


def test_missing_cursor_has_actionable_error(monkeypatch):
    import map_editor.qt_runtime as runtime
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'xcb')

    def missing_library(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, '\tlibxcb-cursor.so.0 => not found\n', '')

    monkeypatch.setattr(runtime.subprocess, 'run', missing_library)
    with pytest.raises(RuntimeError, match='sudo apt install libxcb-cursor0'):
        check_xcb_dependencies(Path('/qt/plugins'))


@pytest.mark.parametrize('platform', ['offscreen', 'wayland', 'minimal'])
def test_non_xcb_platforms_do_not_require_xcb_cursor(monkeypatch, platform):
    import map_editor.qt_runtime as runtime
    monkeypatch.setenv('QT_QPA_PLATFORM', platform)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('Must not inspect xcb'))
    check_xcb_dependencies(Path('/qt/plugins'))


def test_dependency_check_runs_outside_the_gui_process(monkeypatch):
    import map_editor.qt_runtime as runtime
    monkeypatch.setattr(runtime.sys, 'platform', 'linux')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'xcb')
    calls = []

    def linked_libraries(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '\tlibxcb-cursor.so.0 => /lib/libxcb-cursor.so.0\n', '')

    monkeypatch.setattr(runtime.subprocess, 'run', linked_libraries)
    check_xcb_dependencies(Path('/qt/plugins'))
    assert calls == [['ldd', '/qt/plugins/platforms/libqxcb.so']]


def test_real_entrypoint_after_cv2_import_starts_offscreen(tmp_path):
    # Exercise the same import order as ros2 run, not just helper mocks.
    code = '''
import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from map_editor import app
original = app.QApplication
def timed_application(arguments):
    application = original(arguments)
    QTimer.singleShot(50, lambda: application.exit(0))
    return application
app.QApplication = timed_application
raise SystemExit(app.main([]))
'''
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c', code], env=environment,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'Could not load the Qt platform plugin' not in result.stderr
