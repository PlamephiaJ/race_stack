"""A wall-side obstacle must not block a clear raceline."""

from types import SimpleNamespace

from state_machine.state_machine import StateMachine


def test_side_obstacle_from_recorded_run_does_not_block_raceline():
    side = SimpleNamespace(
        s_center=19.0705, d_right=-0.7945, d_left=-0.5512)
    node = SimpleNamespace(
        obstacles=[side], cur_s=18.0, track_length=27.7,
        params=SimpleNamespace(gb_horizon_m=6.9, lateral_width_gb_m=0.4))
    assert StateMachine._check_gbfree.fget(node)


def test_obstacle_edge_entering_driving_corridor_blocks_raceline():
    obstacle = SimpleNamespace(
        s_center=19.0705, d_right=-0.55, d_left=-0.35)
    node = SimpleNamespace(
        obstacles=[obstacle], cur_s=18.0, track_length=27.7,
        params=SimpleNamespace(gb_horizon_m=6.9, lateral_width_gb_m=0.4))
    assert not StateMachine._check_gbfree.fget(node)
