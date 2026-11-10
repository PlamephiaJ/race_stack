"""The state machine must stop using a spline when its publisher goes quiet."""

import ast
from pathlib import Path
from types import SimpleNamespace


def availability_check(monotonic_now):
    """Load the actual property body without starting a ROS node."""
    source = Path(__file__).resolve().parents[1] / 'state_machine' / 'state_machine.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'StateMachine')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == '_check_availability_splini_wpts')
    method.decorator_list = []
    namespace = {
        'time': SimpleNamespace(monotonic=lambda: monotonic_now),
        'time_to_float': float,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[method.name]


def test_stale_spline_is_rejected_and_cleared():
    spline = SimpleNamespace(wpnts=[1, 2],
                             header=SimpleNamespace(stamp=10.0),
                             last_switch_time=9.0)
    node = SimpleNamespace(
        avoidance_wpnts=spline,
        avoidance_received_at=100.0,
        last_valid_avoidance_wpnts=None,
        params=SimpleNamespace(splini_ttl=2.0, splini_hyst_timer_sec=0.2),
    )
    assert availability_check(100.1)(node)
    assert node.last_valid_avoidance_wpnts == [1, 2]
    assert not availability_check(102.1)(node)
    assert node.avoidance_wpnts is None
    assert node.last_valid_avoidance_wpnts is None
