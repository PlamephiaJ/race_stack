"""A blind trailing floor must respect the raceline speed limit."""

from types import SimpleNamespace

from controller.map import MAP_Controller
from controller.pp import PP_Controller


def trailing_stub():
    return SimpleNamespace(
        opponent=[19.07, -0.67, 0.0, True, False],
        position_in_map_frenet=[12.5, 0.0, 0.77, 0.0],
        track_length=27.7, trailing_gap=1.5,
        trailing_p_gain=1.0, trailing_i_gain=0.0,
        trailing_d_gain=0.2, i_gap=0.0,
        loop_rate=40, blind_trailing_speed=1.5)


def test_blind_trailing_never_exceeds_raceline_speed():
    for controller in (MAP_Controller, PP_Controller):
        stub = trailing_stub()
        assert controller.trailing_controller(stub, 0.76) == 0.76


def test_blind_trailing_can_still_stop_for_close_obstacle():
    for controller in (MAP_Controller, PP_Controller):
        stub = trailing_stub()
        stub.position_in_map_frenet[0] = 18.92
        assert controller.trailing_controller(stub, 0.76) == 0.0
