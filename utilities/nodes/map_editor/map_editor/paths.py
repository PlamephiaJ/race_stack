"""Resolve source maps first, including colcon symlink and installed layouts."""
import os
from pathlib import Path


def source_stack_master():
    candidates = [Path(__file__).resolve(), Path.cwd()]
    candidates.extend(Path(p) for p in os.environ.get('AMENT_PREFIX_PATH', '').split(':') if p)
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            for relative in ('stack_master', 'src/race_stack/stack_master'):
                root = parent / relative
                if (root / 'maps').is_dir() and (root / 'package.xml').is_file():
                    return root
    return None


def stack_share():
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('stack_master'))


def default_maps_root():
    explicit = os.environ.get('RACE_STACK_MAPS_DIR')
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = source_stack_master()
    if root:
        return root / 'maps'
    try:
        return stack_share() / 'maps'
    except (ImportError, LookupError):
        return Path.cwd() / 'maps'


def planner_config():
    root = source_stack_master()
    if root:
        return root / 'config/global_planner'
    return stack_share() / 'config/global_planner'
