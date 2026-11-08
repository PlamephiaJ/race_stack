"""Translate repository car models into the subset used by Global Planner."""
import math
from pathlib import Path

import yaml


def positive(data, key, source):
    value = data.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{source}: {key} must be positive and finite.')
    return float(value)


def ros_parameters(data, node):
    """Merge the shared VESC block with the concrete node's overrides."""
    if not isinstance(data, dict):
        raise ValueError('VESC YAML must contain ROS parameter mappings.')
    result = {}
    for name in ('/**', node, '/' + node, '/**/' + node):
        if name in data:
            block = data[name]
            if not isinstance(block, dict) or not isinstance(block.get('ros__parameters'), dict):
                raise ValueError(f'Invalid {name}.ros__parameters in VESC configuration.')
            result.update(block['ros__parameters'])
    return result


def vehicle_settings(directory):
    """Read one named car; keep physical limits as a reproducible snapshot.

    GGV lateral grip and aerodynamic drag have no equivalent car_model fields.
    Longitudinal model bounds are applied to copies of the existing CSV tables.
    """
    directory = Path(directory)
    model = yaml.safe_load((directory / 'car_model.yaml').read_text())
    if not isinstance(model, dict):
        raise ValueError('car_model.yaml must be a mapping.')
    source = f'{directory.name}/car_model.yaml'
    mass = positive(model, 'm', source)
    width = positive(model, 'width_m', source)
    model_speed = positive(model, 'v_max', source)
    wheelbase = positive(model, 'wheelbase', source)
    steering = positive(model, 'max_steering_angle', source)
    if steering >= math.pi / 2:
        raise ValueError(f'{source}: max_steering_angle must be below pi/2.')
    curvature = min(1 / positive(model, 'min_radius_curv_m', source), math.tan(steering) / wheelbase)
    acceleration = positive(model, 'a_max', source)
    braking = -model.get('a_min', 0) if type(model.get('a_min')) in (int, float) else 0
    if not math.isfinite(braking) or braking <= 0:
        raise ValueError(f'{source}: a_min must be negative and finite.')
    speed_limits = [model_speed]
    vesc_speed = None
    path = directory / 'vesc.yaml'
    if path.is_file():
        vesc = yaml.safe_load(path.read_text())
        conversion = ros_parameters(vesc, 'ackermann_to_vesc_node')
        driver = ros_parameters(vesc, 'vesc_driver_node')
        gain = positive(conversion, 'speed_to_erpm_gain', path)
        offset = conversion.get('speed_to_erpm_offset', 0.)
        if type(offset) not in (int, float) or not math.isfinite(offset):
            raise ValueError(f'{path}: invalid speed_to_erpm_offset.')
        erpm = positive(driver, 'speed_max', path)
        vesc_speed = (erpm - offset) / gain
        if vesc_speed <= 0:
            raise ValueError(f'{path}: VESC has no positive forward speed range.')
        speed_limits.append(vesc_speed)
        actuator = ros_parameters(vesc, 'actuation_manager')
        # Do not mistake the shared ERPM speed_max for the manager's m/s value.
        specific = vesc.get('actuation_manager', vesc.get('/actuation_manager',
                            vesc.get('/**/actuation_manager', {}))).get('ros__parameters', {})
        if 'speed_max' in specific:
            speed_limits.append(positive(specific, 'speed_max', path))
        if 'steering_min' in actuator or 'steering_max' in actuator:
            left = positive(actuator, 'steering_max', path)
            right = -actuator.get('steering_min', 0) if type(actuator.get('steering_min')) in (int, float) else 0
            if not math.isfinite(right) or not 0 < min(left, right) < math.pi / 2:
                raise ValueError(f'{path}: expected negative steering_min and positive steering_max.')
            curvature = min(curvature, math.tan(min(left, right)) / wheelbase)
        if 'max_acceleration' in actuator:
            acceleration = min(acceleration, positive(actuator, 'max_acceleration', path))
        if 'max_deceleration' in actuator:
            braking = min(braking, positive(actuator, 'max_deceleration', path))
    speed = min(speed_limits)
    values = {'veh_params.mass': mass, 'veh_params.width': width,
              'veh_params.v_max': speed, 'veh_params.curvlim': curvature}
    vehicle = {'name': directory.name, 'speed_limit': speed, 'curvature_limit': curvature,
               'acceleration_limit': acceleration, 'deceleration_limit': braking,
               'model_speed': model_speed, 'vesc_speed': vesc_speed}
    return values, vehicle
