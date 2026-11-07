"""Validate the on-disk map bundle without ROS, simulation or localization."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .sectors import FILES


def validate_runtime_plan(plan):
    """Check the arrays consumed by the trajectory publisher and interpolator."""
    for key in ('map_info_str', 'est_lap_time'):
        if 'data' not in plan[key]:
            raise ValueError(f'Missing {key}.data.')
    for key in ('centerline_markers', 'global_traj_markers_iqp',
                'global_traj_markers_sp', 'trackbounds_markers'):
        if not isinstance(plan[key]['markers'], list):
            raise ValueError(f'Invalid {key}.markers.')
    for key in ('centerline_waypoints', 'global_traj_wpnts_iqp', 'global_traj_wpnts_sp'):
        points = plan[key]['wpnts']
        if len(points) < 4:
            raise ValueError(f'{key}: at least four waypoints required for runtime splines.')
        values = np.asarray([[p[field] for field in (
            's_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2', 'd_left', 'd_right')]
                             for p in points], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'{key}: nonfinite waypoint data.')
        if np.any(np.diff(values[:, 0]) <= 0) or values[0, 0] != 0:
            raise ValueError(f'{key}: s_m must start at zero and increase strictly.')
        if np.any(values[:, 5] < 0):
            raise ValueError(f'{key}: negative speed.')
        if any(type(p['id']) is not int or p['id'] != i for i, p in enumerate(points)):
            raise ValueError(f'{key}: waypoint IDs must match array indices.')


@dataclass
class ReadinessReport:
    sim_issues: list
    localization: str
    localization_status: str

    @property
    def sim_ready(self):
        return not self.sim_issues

    def text(self):
        sim = ('SIM: READY — required map files, trajectory and Sector indices verified.'
               if self.sim_ready else 'SIM: NOT READY\n' + '\n'.join(f'• {issue}' for issue in self.sim_issues))
        return (sim + '\n\nReal car / PBSTREAM: ' + self.localization
                + '\n\nThis checks the map bundle. Runtime dependencies and vehicle setup are not tested.')


def validate_saved_map(document):
    """Read back every required file; do not infer readiness from a preview."""
    from .document import MapDocument, file_digest
    issues = []
    path = document.path
    if path is None:
        return ReadinessReport(['Map has not been saved.'], 'Map has not been saved.', 'missing')
    path = Path(path)
    required = [path, path.with_suffix('.png'), path.parent / 'global_waypoints.json',
                *(path.parent / filename for filename in FILES.values())]
    for source in required:
        if not source.is_file() or not source.stat().st_size:
            issues.append(f'Missing or empty: {source.name}')
    if path.parent.name != path.stem or path.suffix != '.yaml':
        issues.append('Launch expects maps/<map_name>/<map_name>.yaml and matching PNG.')
    if not document.plan_current:
        issues.append('No trajectory matching the current map / planner parameters. Generate again.')
    issues.extend(document.sector_issues())
    try:
        loaded = MapDocument.load(path)
        if loaded.fingerprint != document.fingerprint:
            issues.append('Saved map differs from the current document.')
        if not loaded.plan_current:
            issues.append('Saved trajectory is missing, damaged, or does not match saved validation data.')
        else:
            validate_runtime_plan(loaded.plan)
            if document.plan_current and loaded.plan_digest(loaded.plan) != document.plan_digest(document.plan):
                issues.append('Saved trajectory differs from the current preview.')
            issues.extend(loaded.sector_issues())
        for kind, config in document.sectors.items():
            if kind in loaded.sectors and loaded.sectors[kind].raw != config.raw:
                issues.append(f'{FILES[kind]} differs from the current configuration.')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        issues.append(f'Could not validate saved bundle: {error}')

    pbstream = path.with_suffix('.pbstream')
    status = 'unverified'
    if not pbstream.is_file() or not pbstream.stat().st_size:
        alternatives = sorted(p.name for p in path.parent.glob('*.pbstream') if p.is_file())
        suffix = f" Other filenames found: {', '.join(alternatives)}." if alternatives else ''
        localization = (f'MISSING usable {pbstream.name}.{suffix} '
                        'A hand-drawn / non-scanned map without matching scan-built PBSTREAM '
                        'cannot be used for the current real-car Cartographer localization flow. '
                        'Scan the real track and use the corresponding localization state.')
        status = 'missing'
    else:
        baseline = document.localization_baseline
        digest = file_digest(pbstream)
        if baseline and baseline.get('geometry') != document.geometry_fingerprint:
            localization = (f'{pbstream.name} exists, but map pixels, resolution, origin or occupancy '
                            'thresholds changed since the baseline. PBSTREAM was not updated. '
                            'Treat it as incompatible until checked against a real scan.')
            status = 'changed'
        elif baseline and baseline.get('pbstream') != digest:
            localization = (f'{pbstream.name} was replaced since the baseline; compatibility is unknown. '
                            'Verify scan provenance, coordinate frame, scale and real-track alignment.')
            status = 'changed'
        else:
            localization = (f'{pbstream.name} exists and is nonempty. '
                            'Spatial compatibility is UNVERIFIED: file presence cannot prove scan '
                            'provenance, frame, scale or alignment with the physical track. '
                            'Verify these before using real-car localization.')
    return ReadinessReport(list(dict.fromkeys(issues)), localization, status)
