"""Incremental builds must discover configuration from the current source tree."""
from pathlib import Path
import runpy
from unittest.mock import patch


def test_config_inventory_ignores_removed_files_in_symlink_build(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    stale = tmp_path / 'config/global_planner'
    stale.mkdir(parents=True)
    (stale / 'racecar_f110.ini').symlink_to(source / 'config/global_planner/racecar_f110.ini')
    (stale / 'retired.ini').write_text('stale build artifact')
    (tmp_path / 'maps').mkdir()
    monkeypatch.chdir(tmp_path)
    captured = {}
    with patch('setuptools.setup', side_effect=lambda **kwargs: captured.update(kwargs)):
        runpy.run_path(str(source / 'setup.py'), run_name='__main__')
    config_files = {
        path
        for destination, files in captured['data_files']
        if destination == 'share/stack_master/config' or destination.startswith('share/stack_master/config/')
        for path in files
    }
    expected = {
        str(path.relative_to(source))
        for path in (source / 'config').rglob('*')
        if path.is_file()
    }
    assert config_files == expected
    assert 'config/BLUE/racecar_f110.ini' in config_files
    assert 'config/global_planner/racecar_f110.ini' not in config_files
    assert 'config/global_planner/retired.ini' not in config_files
