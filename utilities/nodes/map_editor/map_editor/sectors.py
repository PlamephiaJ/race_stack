"""Offline editing of the existing Sector Tuner ROS parameter files."""
from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np
import yaml

FILES = {'speed': 'speed_scaling.yaml', 'overtaking': 'ot_sectors.yaml'}
COLORS = ('#ff795e', '#53bfff', '#ffd166', '#be8cff', '#55ddaa', '#ee86c3')


def trajectory_signature(plan):
    """Bind indices to ordered geometry, including the closure point, not speed."""
    points = [[p['x_m'], p['y_m'], p['s_m']]
              for p in plan['global_traj_wpnts_iqp']['wpnts']]
    return hashlib.sha256(json.dumps(points, allow_nan=False).encode()).hexdigest()


def finite_number(value, name, minimum, maximum=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number.')
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be within [{minimum}, {maximum}].')


class SectorConfig:
    def __init__(self, kind, raw):
        self.kind = kind
        self.prefix = 'Sector' if kind == 'speed' else 'Overtaking_sector'
        self.raw = deepcopy(raw)
        node = 'sector_tuner' if kind == 'speed' else 'ot_interpolator'
        if not isinstance(raw, dict):
            raise ValueError('Sector YAML must be a ROS parameter mapping.')
        candidates = [key for key, value in self.raw.items()
                      if isinstance(key, str) and key.rstrip('/').split('/')[-1] == node
                      and isinstance(value, dict) and 'ros__parameters' in value]
        if len(candidates) != 1:
            raise ValueError(f'Expected one {node} ROS parameter block.')
        self.node_key = candidates[0]
        if not isinstance(self.params, dict):
            raise ValueError('ros__parameters must be a mapping.')
        count = self.params.get('n_sectors')
        if type(count) is not int or count < 1:
            raise ValueError('n_sectors must be a positive integer.')
        for i in range(count):
            if not isinstance(self.params.get(f'{self.prefix}{i}'), dict):
                raise ValueError(f'Missing {self.prefix}{i}.')

    @property
    def params(self):
        return self.raw[self.node_key]['ros__parameters']

    @property
    def sections(self):
        return [self.params[f'{self.prefix}{i}'] for i in range(self.params['n_sectors'])]

    @classmethod
    def default(cls, kind, count):
        if count < 2:
            raise ValueError('Need at least two global waypoints.')
        if kind == 'speed':
            return cls(kind, {'/**/sector_tuner': {'ros__parameters': {
                'global_limit': .5, 'n_sectors': 1, 'Sector0': {
                    'start': 0, 'end': count - 1, 'scaling': .5,
                    'only_FTG': False, 'no_FTG': False}}}})
        return cls(kind, {'ot_interpolator': {'ros__parameters': {
            'n_sectors': 1, 'yeet_factor': 1.25, 'spline_len': 30,
            'ot_sector_begin': .5, 'Overtaking_sector0': {
                'start': 0, 'end': count - 1, 'ot_flag': False}}}})

    def validate(self, count):
        sections = self.sections
        previous = -1
        for i, section in enumerate(sections):
            start, end = section.get('start'), section.get('end')
            if type(start) is not int or type(end) is not int:
                raise ValueError(f'{self.prefix}{i}: start/end must be integer waypoint indices.')
            # The original slicers wrote len(wpnts) as the last end sentinel.
            # Preserve this legacy convention; all other ends are inclusive.
            maximum = count if i == len(sections) - 1 else count - 1
            if start != previous + 1 or not 0 <= start < count or not start <= end <= maximum:
                raise ValueError(f'{self.prefix}{i}: invalid/gapped indices {start}…{end} for {count} waypoints.')
            previous = end
            if self.kind == 'speed':
                finite_number(section.get('scaling'), f'{self.prefix}{i}.scaling', 0, 1)
            elif type(section.get('ot_flag')) is not bool:
                raise ValueError(f'{self.prefix}{i}.ot_flag must be boolean.')
        if previous not in (count - 1, count):
            raise ValueError('Sections must cover the whole global raceline.')
        if self.kind == 'speed':
            finite_number(self.params.get('global_limit'), 'global_limit', 0, 1)
        else:
            finite_number(self.params.get('yeet_factor'), 'yeet_factor', 0)
            length = self.params.get('spline_len')
            if type(length) is not int or length < 1:
                raise ValueError('spline_len must be a positive integer.')
            # The runtime uses explicit indexing in both transition loops.
            for i, section in enumerate(sections):
                previous_enabled = sections[i - 1]['ot_flag']
                next_enabled = sections[(i + 1) % len(sections)]['ot_flag']
                if section['ot_flag'] and (
                        (not previous_enabled and section['start'] + length > count)
                        or (not next_enabled and section['end'] - length < 0)):
                    raise ValueError('Overtaking transition exceeds waypoint array; reduce spline_len.')

    def _replace_sections(self, sections):
        for key in list(self.params):
            if re.fullmatch(re.escape(self.prefix) + r'\d+', key):
                del self.params[key]
        self.params['n_sectors'] = len(sections)
        for i, section in enumerate(sections):
            self.params[f'{self.prefix}{i}'] = section

    def section_at(self, index):
        return max(i for i, section in enumerate(self.sections) if section['start'] <= index)

    def split(self, index):
        i = self.section_at(index)
        sections = deepcopy(self.sections)
        section = sections[i]
        if index == section['start'] or index > section['end']:
            return i
        following = deepcopy(section)
        section['end'] = index - 1
        following['start'] = index
        sections.insert(i + 1, following)
        self._replace_sections(sections)
        return i + 1

    def delete_boundary(self, section_index):
        if section_index <= 0:
            raise ValueError('Waypoint 0 is the fixed start of the lap.')
        sections = deepcopy(self.sections)
        sections[section_index - 1]['end'] = sections[section_index]['end']
        del sections[section_index]
        self._replace_sections(sections)
        return section_index - 1

    def move_boundary(self, section_index, index, count):
        if section_index <= 0:
            raise ValueError('Waypoint 0 is the fixed start of the lap.')
        sections = self.sections
        lower = sections[section_index - 1]['start'] + 1
        upper = min(sections[section_index]['end'], count - 1)
        if not lower <= index <= upper:
            raise ValueError(f'Boundary must stay between waypoints {lower} and {upper}.')
        sections[section_index - 1]['end'] = index - 1
        sections[section_index]['start'] = index

    def dumps(self):
        return yaml.safe_dump(self.raw, sort_keys=False).encode()

    def speed_factors(self, count):
        """Match SectorTuner.get_vel_scaling, including its lap-end convention.

        Keep the runtime's branch order for short adjacent sections. In particular,
        end is used directly at the lap seam (legacy files may use count there).
        """
        self.validate(count)
        sections = self.sections
        if len(sections) == 1:
            return np.full(count, sections[0]['scaling'], dtype=float)
        result = np.empty(count)
        for s in range(count):
            for i, section in enumerate(sections):
                start, value = section['start'], section['scaling']
                last = i == len(sections) - 1
                end = section['end'] if last else sections[i + 1]['start']
                if start <= s < start + 10:
                    result[s] = np.interp(s, [start - 10, start + 10],
                                          [sections[i - 1]['scaling'], value])
                elif start + 10 <= s < end - 10:
                    result[s] = value
                elif (s >= end - 10) if last else (end - 10 <= s < end):
                    result[s] = np.interp(s, [end - 10, end + 10],
                                          [value, sections[(i + 1) % len(sections)]['scaling']])
        return result
