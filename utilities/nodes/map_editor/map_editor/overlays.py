"""Decode the existing global_waypoints.json format without importing ROS."""
import math

import numpy as np

LAYERS = {
    'Centerline': '#00b8ff',
    'Global Raceline': '#f04040',
    'Shortest Path': '#ffa500',
    'Track Boundaries': '#00ba64',
}


def plan_layers(plan):
    layers = {}
    for label, key in (
        ('Centerline', 'centerline_waypoints'),
        ('Global Raceline', 'global_traj_wpnts_iqp'),
        ('Shortest Path', 'global_traj_wpnts_sp'),
    ):
        points = np.asarray([[p['x_m'], p['y_m']] for p in plan[key]['wpnts']], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f'Invalid or empty {label} waypoints.')
        layers[label] = points
    markers = plan['trackbounds_markers']['markers']
    points = [[m['pose']['position']['x'], m['pose']['position']['y']] for m in markers]
    bounds = np.asarray(points, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all():
        raise ValueError('Invalid or empty track bounds.')
    layers['Track Boundaries'] = bounds
    return layers


def transform_plan(plan, origin):
    """Transform the planner's local metric result into the YAML world frame."""
    ox, oy, yaw = origin
    c, s = math.cos(yaw), math.sin(yaw)

    def xy(item, xkey, ykey):
        x, y = item[xkey], item[ykey]
        item[xkey], item[ykey] = ox + c * x - s * y, oy + s * x + c * y

    for group in plan.values():
        if not isinstance(group, dict):
            continue
        for waypoint in group.get('wpnts', []):
            xy(waypoint, 'x_m', 'y_m')
            if 'psi_rad' in waypoint:
                heading = waypoint['psi_rad'] + yaw
                waypoint['psi_rad'] = math.atan2(math.sin(heading), math.cos(heading))
        for marker in group.get('markers', []):
            # Existing Race Stack markers use pose positions, with identity orientation.
            xy(marker['pose']['position'], 'x', 'y')
            orientation = marker['pose']['orientation']
            qx, qy, qz, qw = (orientation.get(k, 0.) for k in ('x', 'y', 'z', 'w'))
            sh, ch = math.sin(yaw / 2), math.cos(yaw / 2)
            orientation.update(x=ch * qx - sh * qy, y=sh * qx + ch * qy,
                               z=ch * qz + sh * qw, w=ch * qw - sh * qz)
    return plan
