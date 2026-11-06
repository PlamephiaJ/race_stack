"""Keep OpenCV's Qt5 wheel paths out of the PySide6 application."""
import os
from pathlib import Path
import subprocess
import sys


def check_pyside_compatibility():
    import PySide6
    if PySide6.__version__.split('.')[:3] == ['6', '12', '0'] and sys.version_info[:2] < (3, 12):
        raise RuntimeError(
            f'PySide6 {PySide6.__version__} has a Signal.emit() reference-count bug on Python '
            f'{sys.version_info.major}.{sys.version_info.minor} (PYSIDE-3474). '
            'Repeated mouse moves or parameter edits can abort the process.\n'
            'Install the supported version before starting the editor:\n'
            '  python3 -m pip install --user --upgrade "PySide6==6.11.2"')


def _opencv_qt_path(value):
    parts = Path(value).parts
    return any(parts[index:index + 2] == ('cv2', 'qt') for index in range(len(parts) - 1))


def prepare_qt_environment():
    # Import first: non-headless opencv-python wheels set these variables on import.
    import cv2  # noqa: F401
    from PySide6.QtCore import QCoreApplication, QLibraryInfo

    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR', 'QT_PLUGIN_PATH'):
        value = os.environ.get(key)
        if value is None:
            continue
        remaining = [entry for entry in value.split(os.pathsep) if entry and not _opencv_qt_path(entry)]
        if len(remaining) != len(value.split(os.pathsep)):
            if remaining:
                os.environ[key] = os.pathsep.join(remaining)
            else:
                os.environ.pop(key, None)
    plugins = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    # Also clear any paths QtCore cached before the environment was cleaned.
    paths = [path for path in QCoreApplication.libraryPaths() if not _opencv_qt_path(path)]
    QCoreApplication.setLibraryPaths(list(dict.fromkeys([plugins, *paths])))
    return Path(plugins)


def check_xcb_dependencies(plugins):
    """Report missing shared libraries before QApplication can abort the process."""
    if not sys.platform.startswith('linux'):
        return
    platform = os.environ.get('QT_QPA_PLATFORM', '').split(';')[0].split(':')[0]
    if platform and platform != 'xcb':
        return
    if not platform and (os.environ.get('WAYLAND_DISPLAY') or not os.environ.get('DISPLAY')):
        return
    plugin = Path(plugins) / 'platforms/libqxcb.so'
    # Never dlopen a Qt platform plugin here. Qt must own its initialization and
    # lifetime; inspect dependencies in a separate loader process instead.
    try:
        result = subprocess.run(['ldd', str(plugin)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return  # Let Qt report its own loader error if inspection is unavailable.
    missing = [line.strip() for line in result.stdout.splitlines() if '=> not found' in line]
    if missing:
        raise RuntimeError(
            'PySide6 X11 (xcb) plugin has missing dependencies: ' + '; '.join(dict.fromkeys(missing)) + '\n'
            'On Ubuntu / Debian install the Qt X11 runtime dependencies:\n'
            '  sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 '
            'libxcb-image0 libxcb-keysyms1 libxcb-render-util0\n'
            'Then run ros2 run map_editor map_editor again.')
