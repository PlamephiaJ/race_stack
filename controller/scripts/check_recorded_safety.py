"""Evaluate new controller decisions against recorded inputs without ROS output.

This is an open-loop check: recorded pose/speed/applied commands stay unchanged.
It does not simulate the changed trajectory or prove collision avoidance.
Source ROS and the workspace; put the source controller package on PYTHONPATH.
"""

import argparse
import csv
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml

from controller.controller_manager import Controller
from controller.map import MAP_Controller
from controller.tracking_safety import RecoverySpeedLimiter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/tmp/recorded_safety.csv'))
    parser.add_argument('--lookup-table', default='NUC2_hangar_pacejka')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    params = yaml.safe_load((root / 'stack_master/config/BLUE/l1_params.yaml').read_text())['controller']['ros__parameters']
    map_names = ('t_clip_min', 't_clip_max', 'm_l1', 'q_l1', 'speed_lookahead',
                 'lat_err_coeff', 'acc_scaler_for_steer', 'dec_scaler_for_steer',
                 'start_scale_speed', 'end_scale_speed', 'downscale_factor',
                 'speed_lookahead_for_steer', 'prioritize_dyn', 'trailing_gap',
                 'trailing_p_gain', 'trailing_i_gain', 'trailing_d_gain', 'blind_trailing_speed')
    model = MAP_Controller(**{k: params[k] for k in map_names}, loop_rate=40,
                           LUT_name=args.lookup_table, logger_info=lambda *a: None,
                           logger_warn=lambda *a: None)
    c = sqlite3.connect(f'file:{args.database.resolve()}?mode=ro', uri=True)
    types = {i: (n, get_message(typ)) for i, n, typ in c.execute('select id,name,type from topics')}
    start = c.execute('select min(timestamp) from messages').fetchone()[0]
    clock = SimpleNamespace(ns=start, seconds=0.0)
    noop = lambda *a, **kw: None
    published = []
    stub = SimpleNamespace(
        mode='MAP', state='GB_TRACK', last_safe_steer=0.0, speed_now=None,
        position_in_map=None, position_in_map_frenet=None, waypoint_array_in_map=None,
        opponent=None, acc_now=np.zeros(10), track_length=None,
        last_odom_time=None, last_frenet_time=None, last_waypoints_time=None,
        last_odom_stamp_ns=None, last_frenet_stamp_ns=None, last_waypoints_stamp_ns=None,
        last_scan_time=None, last_scan_stamp_ns=None, scan=None,
        applied_speed=None, last_applied_time=None, last_applied_stamp_ns=None,
        max_input_age_sec=.5, max_odom_age_sec=.3, max_path_start_rear_m=.75,
        require_applied_feedback=True, front_guard_enabled=True,
        front_stop_latched=False, front_speed_cap=0.0, last_control_time=0.0,
        recovery_limiter=RecoverySpeedLimiter(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=clock.ns, to_msg=lambda: Time(sec=clock.ns // 10**9, nanosec=clock.ns % 10**9))),
        get_logger=lambda: SimpleNamespace(warning=noop, info=noop),
        drive_pub=SimpleNamespace(publish=published.append))
    safety = yaml.safe_load((root / 'controller/config/tracking_safety.yaml').read_text())['/**']['ros__parameters']
    for name in ('front_clearance_m', 'front_decel_mps2', 'front_ttc_stop_sec', 'front_ttc_release_sec', 'front_speed_margin'):
        setattr(stub, name, safety[name])
    stub.inputs_ready = lambda now, **kw: Controller.inputs_ready(stub, now, **kw)
    stub.path_start_too_far_behind = lambda: Controller.path_start_too_far_behind(stub)
    stub.front_obstacle_stop = lambda now: Controller.front_obstacle_stop(stub, now)

    def cycle(braking=False):
        result = model.main_loop(stub.state, stub.position_in_map,
                                 stub.waypoint_array_in_map, stub.speed_now, None,
                                 stub.position_in_map_frenet, stub.acc_now,
                                 stub.track_length, steering_speed=stub.speed_now if braking else None)
        return result[0], result[3]

    stub.map_cycle = cycle
    callbacks = {'/state_estimation/odom': Controller.odom_cb,
                 '/car_state/frenet/odom': Controller.car_state_frenet_cb,
                 '/local_waypoints': Controller.local_waypoint_cb,
                 '/scan': Controller.scan_cb,
                 '/ackermann_cmd_applied': Controller.applied_command_cb,
                 '/state': Controller.state_cb}
    rows = []
    navigation_time = 0.0
    with patch('controller.controller_manager.time.monotonic', lambda: clock.seconds):
        for topic, ns, data in c.execute('select topic_id,timestamp,data from messages order by timestamp,id'):
            name, cls = types[topic]
            if name not in callbacks and name not in ('/drive', '/global_waypoints_scaled', '/rosout'):
                continue
            clock.ns, clock.seconds = ns, (ns - start) * 1e-9
            m = deserialize_message(data, cls)
            if name in callbacks:
                callbacks[name](stub, m)
            elif name == '/rosout':
                if not navigation_time and m.name == 'actuation_manager' and m.msg == 'Actuation source: navigation':
                    navigation_time = clock.seconds
            elif name == '/global_waypoints_scaled':
                stub.track_length = m.wpnts[-1].s_m
            elif name == '/drive':
                Controller.control_loop(stub)
                command = published[-1].drive
                rows.append(dict(t=clock.seconds, recorded_speed=m.drive.speed,
                                 new_speed=command.speed,
                                 recorded_steer_deg=math.degrees(m.drive.steering_angle),
                                 new_steer_deg=math.degrees(command.steering_angle),
                                 front_stop=stub.front_stop_latched,
                                 front_speed_cap=stub.front_speed_cap,
                                 front_ttc=stub.front_ttc if hasattr(stub, 'front_ttc') else float('inf'),
                                 recovery=stub.recovery_limiter.recovering))
    with args.output.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first_guard = next((r for r in rows if r['front_stop']), None)
    at_enable = next(r for r in rows if r['t'] >= navigation_time)
    print('Decision at navigation enable:', at_enable)
    print('First new front guard on recorded inputs:', first_guard)
    for key in ('recorded_speed', 'new_speed'):
        stops = sum(a[key] > 0 and b[key] == 0 for a,b in zip(rows,rows[1:])
                    if navigation_time < b['t'] < 20.0)
        print('Stop entries before 20s:', key, stops)
    print('CSV:', args.output)
    print('Open loop; recorded actuation and trajectory were not replaced. MAP table:', args.lookup_table)


if __name__ == '__main__':
    main()
