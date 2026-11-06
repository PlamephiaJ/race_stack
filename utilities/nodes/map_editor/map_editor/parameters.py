"""Validated editor profiles and isolated copies of the existing planner config."""
import configparser
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import yaml

from .document import atomic_bytes

METADATA_KEY = 'map_editor_planner'


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    tab: str
    default: object
    minimum: float = 0.
    maximum: float = 1e6
    kind: str = 'float'


PARAMETERS = (
    Parameter('safety_width', 'Raceline safety width (m)', 'Planner', .7, .05, 20.),
    Parameter('safety_width_sp', 'Shortest path safety width (m)', 'Planner', .7, .05, 20.),
    Parameter('reverse_mapping', 'Reverse travel direction', 'Planner', False, kind='bool'),
    Parameter('veh_params.v_max', 'Maximum speed (m/s)', 'Vehicle', 15., .01, 100.),
    Parameter('veh_params.length', 'Vehicle length (m)', 'Vehicle', .535, .01, 20.),
    Parameter('veh_params.width', 'Vehicle width (m)', 'Vehicle', .3, .01, 20.),
    Parameter('veh_params.mass', 'Vehicle mass (kg)', 'Vehicle', 3.518, .001, 1e5),
    Parameter('veh_params.dragcoeff', 'Drag coefficient', 'Vehicle', .0136, 0., 1e4),
    Parameter('veh_params.curvlim', 'Curvature limit (rad/m)', 'Vehicle', 1., .000001, 100.),
    Parameter('stepsize_opts.stepsize_prep', 'Preparation step (m)', 'Sampling / smoothing', .05, .0001, 10.),
    Parameter('stepsize_opts.stepsize_reg', 'Optimization step (m)', 'Sampling / smoothing', .2, .0001, 10.),
    Parameter('stepsize_opts.stepsize_interp_after_opt', 'Output waypoint step (m)',
              'Sampling / smoothing', .1, .0001, 10.),
    Parameter('reg_smooth_opts.k_reg', 'B-spline order', 'Sampling / smoothing', 3, 1, 5, 'int'),
    Parameter('reg_smooth_opts.s_reg', 'Spline smoothing factor', 'Sampling / smoothing', 1., 0., 1e4),
    Parameter('optim_opts_mincurv.iqp_iters_min', 'Minimum IQP iterations', 'Optimization / speed', 5, 1, 1000, 'int'),
    Parameter('optim_opts_mincurv.iqp_curverror_allowed', 'Allowed IQP curvature error (rad/m)',
              'Optimization / speed', .1, .000001, 100.),
    Parameter('vel_calc_opts.dyn_model_exp', 'Dynamics model exponent', 'Optimization / speed', 1., 1., 2.),
    Parameter('vel_calc_opts.vel_profile_conv_filt_window', 'Speed filter window (0 = disabled)',
              'Optimization / speed', None, 1, 10001, 'optional_int'),
    Parameter('ggv_file', 'GGV CSV file', 'Dynamics files', 'ggv.csv', kind='file'),
    Parameter('ax_max_machines_file', 'Acceleration limits CSV file', 'Dynamics files',
              'ax_max_machines.csv', kind='file'),
)
BY_KEY = {parameter.key: parameter for parameter in PARAMETERS}


def section_for(group):
    return 'OPTIMIZATION_OPTIONS' if group.startswith('optim_opts_') else 'GENERAL_OPTIONS'


class PlannerParameters:
    def __init__(self, values=None):
        self.values = {p.key: p.default for p in PARAMETERS}
        if values is not None:
            unknown = set(values) - BY_KEY.keys()
            if unknown:
                raise ValueError('Unknown planner parameters: ' + ', '.join(sorted(unknown)))
            self.values.update(deepcopy(values))
        self.validate()

    @classmethod
    def from_config(cls, directory):
        directory = Path(directory)
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(directory / 'racecar_f110.ini'):
            raise ValueError(f'Missing planner configuration: {directory}/racecar_f110.ini')
        ros_config = yaml.safe_load((directory / 'global_planner_params.yaml').read_text())
        node = ros_config['global_planner']['ros__parameters']
        values = {}
        for parameter in PARAMETERS:
            key = parameter.key
            if key in ('safety_width', 'safety_width_sp', 'reverse_mapping'):
                values[key] = node[key]
            else:
                group, _, item = key.partition('.')
                raw = json.loads(parser.get(section_for(group), group))
                values[key] = raw[item] if item else raw
        return cls(values)

    @classmethod
    def from_profile(cls, profile, defaults=None):
        if not isinstance(profile, dict) or set(profile) != {'version', 'parameters'}:
            raise ValueError('Expected planner profile with version and parameters fields.')
        if type(profile['version']) is not int or profile['version'] != 1:
            raise ValueError('Unsupported planner profile version (expected 1).')
        if not isinstance(profile['parameters'], dict):
            raise ValueError('Planner profile parameters must be a mapping.')
        values = deepcopy(defaults.values) if defaults is not None else {}
        values.update(profile['parameters'])
        return cls(values)

    def profile(self):
        self.validate()
        return {'version': 1, 'parameters': deepcopy(self.values)}

    def validate(self):
        for parameter in PARAMETERS:
            value = self.values[parameter.key]
            if parameter.kind == 'bool':
                valid = type(value) is bool
            elif parameter.kind == 'file':
                # All dynamics inputs live inside the copied veh_dyn_info directory.
                valid = (isinstance(value, str) and bool(value.strip()) and not Path(value).is_absolute()
                         and '..' not in Path(value).parts and value.lower().endswith('.csv'))
            elif parameter.kind == 'optional_int' and value is None:
                valid = True
            else:
                valid = (type(value) in (int, float) and math.isfinite(value)
                         and parameter.minimum <= value <= parameter.maximum)
                if parameter.kind in ('int', 'optional_int'):
                    valid = valid and type(value) is int
                if parameter.kind == 'optional_int':
                    valid = valid and value % 2 == 1
            if not valid:
                raise ValueError(f'Invalid {parameter.label} ({parameter.key}): {value!r}')
        width = self.values['veh_params.width']
        for key in ('safety_width', 'safety_width_sp'):
            if self.values[key] < width:
                raise ValueError(f'{key} must be at least the vehicle width ({width:g} m).')

    def export(self, path):
        atomic_bytes(path, yaml.safe_dump(self.profile(), sort_keys=False).encode())

    @classmethod
    def load(cls, path, defaults=None):
        return cls.from_profile(yaml.safe_load(Path(path).read_text()), defaults)

    def snapshot(self, source, destination):
        """Copy source configuration and write overrides only into the job's copy."""
        self.validate()
        destination = Path(destination)
        shutil.copytree(source, destination)
        ini = destination / 'racecar_f110.ini'
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(ini):
            raise ValueError(f'Missing planner configuration: {ini}')
        groups = {}
        for parameter in PARAMETERS:
            if parameter.tab == 'Planner':
                continue
            group, _, item = parameter.key.partition('.')
            section = section_for(group)
            value = self.values[parameter.key]
            if item:
                if (section, group) not in groups:
                    groups[section, group] = json.loads(parser.get(section, group))
                groups[section, group][item] = value
            else:
                groups[section, group] = value
        for (section, group), value in groups.items():
            parser.set(section, group, json.dumps(value, allow_nan=False))
        with ini.open('w') as stream:
            parser.write(stream)
        for key in ('ggv_file', 'ax_max_machines_file'):
            if not (destination / 'veh_dyn_info' / self.values[key]).is_file():
                raise ValueError(f'Missing dynamics CSV: {self.values[key]} (in veh_dyn_info)')
        ros_path = destination / 'global_planner_params.yaml'
        ros_config = yaml.safe_load(ros_path.read_text())
        ros_config['global_planner']['ros__parameters'].update(
            {key: self.values[key] for key in ('safety_width', 'safety_width_sp', 'reverse_mapping')})
        atomic_bytes(ros_path, yaml.safe_dump(ros_config, sort_keys=False).encode())
        return destination
