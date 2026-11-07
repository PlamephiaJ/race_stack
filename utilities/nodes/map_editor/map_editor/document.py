"""Map data, geometry, bounded undo history and Race Stack persistence."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import zlib

import cv2
import numpy as np
import yaml

from .sectors import FILES, SectorConfig, trajectory_signature

MAX_PIXELS = 64_000_000
VALIDATION_KEY = 'map_editor_validation'


def validate_metadata(metadata):
    resolution = float(metadata['resolution'])
    origin = np.asarray(metadata['origin'], dtype=float)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('Resolution must be positive and finite.')
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Origin must contain finite x, y, yaw values.')
    free, occupied = float(metadata['free_thresh']), float(metadata['occupied_thresh'])
    if not 0 <= free < occupied <= 1:
        raise ValueError('Require 0 <= free_thresh < occupied_thresh <= 1.')
    if 'initial_pose' in metadata:
        pose = np.asarray(metadata['initial_pose'], dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError('Initial pose must contain finite x, y, yaw values.')


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(content)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class MapDocument:
    def __init__(self, pixels, metadata, path=None):
        pixels = np.asarray(pixels)
        if pixels.ndim != 2 or pixels.dtype != np.uint8 or pixels.size > MAX_PIXELS or not pixels.size:
            raise ValueError('Map must be a nonempty uint8 grayscale image, up to 64 million pixels.')
        validate_metadata(metadata)
        self.pixels = pixels.copy()
        self.metadata = deepcopy(metadata)
        self.path = Path(path) if path else None
        self.undo_stack = []
        self.redo_stack = []
        self.saved_fingerprint = self.fingerprint if path else None
        self.plan = None
        self.plan_fingerprint = None
        self.plan_load_error = None
        self.sectors = {}
        self.sector_load_errors = {}
        self.sector_bindings = deepcopy(self.metadata.get(VALIDATION_KEY, {}).get('sector_bindings', {}))
        self.saved_sector_fingerprint = self.sector_fingerprint
        self.localization_baseline = self.metadata.get(VALIDATION_KEY, {}).get('localization_baseline')

    @classmethod
    def new(cls, width, height, resolution, origin=(0., 0., 0.), background=0):
        if width < 1 or height < 1 or width * height > MAX_PIXELS:
            raise ValueError('Map size must be positive and at most 64 million pixels.')
        return cls(np.full((height, width), background, np.uint8), {
            'resolution': float(resolution), 'origin': list(origin), 'negate': 0,
            'occupied_thresh': 0.65, 'free_thresh': 0.196,
        })

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        metadata = yaml.safe_load(path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError('Map YAML must be a mapping.')
        # The Race Stack planner consumes trinary maps; preserve extra YAML fields.
        if metadata.get('mode', 'trinary') != 'trinary':
            raise ValueError('Only trinary occupancy maps are supported. Convert scale/raw maps first.')
        image_path = Path(metadata['image']).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        pixels = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError(f'Cannot read map image: {image_path}')
        negate = metadata.get('negate', 0)
        if negate not in (0, 1):
            raise ValueError('negate must be 0 or 1.')
        if negate:
            pixels = 255 - pixels
        metadata['negate'] = 0
        document = cls(pixels, metadata, path)
        for kind, filename in FILES.items():
            source = path.parent / filename
            if source.is_file():
                try:
                    document.sectors[kind] = SectorConfig(kind, yaml.safe_load(source.read_text()))
                    document.sector_bindings.setdefault(kind, 'unverified')
                except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
                    document.sector_load_errors[kind] = str(error)
        result = path.parent / 'global_waypoints.json'
        if result.is_file():
            try:
                plan = json.loads(result.read_text())
                evidence = metadata.get(VALIDATION_KEY, {})
                if evidence and (evidence.get('map_fingerprint') != document.fingerprint
                                 or evidence.get('plan_digest') != document.plan_digest(plan)):
                    raise ValueError('Saved map/trajectory validation does not match.')
                # Legacy maps have no recorded binding; their colocated files are
                # the baseline. Subsequent replans must match ordered geometry.
                if not evidence:
                    document.sector_bindings = {kind: trajectory_signature(plan)
                                                for kind in document.sectors}
                document.attach_plan(plan)
            except (OSError, ValueError, KeyError, TypeError) as error:
                document.plan_load_error = str(error)
        pbstream = path.with_suffix('.pbstream')
        if document.localization_baseline is None and pbstream.is_file() and pbstream.stat().st_size:
            document.localization_baseline = {
                'geometry': document.geometry_fingerprint,
                'pbstream': file_digest(pbstream),
            }
        document.saved_sector_fingerprint = document.sector_fingerprint
        return document

    @property
    def fingerprint(self):
        metadata = {k: v for k, v in self.metadata.items() if k not in ('image', VALIDATION_KEY)}
        digest = hashlib.sha256(self.pixels.tobytes())
        digest.update(str(self.pixels.shape).encode())
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        return digest.hexdigest()

    @property
    def dirty(self):
        return (self.fingerprint != self.saved_fingerprint
                or self.sector_fingerprint != self.saved_sector_fingerprint)

    @property
    def geometry_fingerprint(self):
        metadata = {key: self.metadata.get(key) for key in
                    ('resolution', 'origin', 'negate', 'occupied_thresh', 'free_thresh', 'mode')}
        digest = hashlib.sha256(self.pixels.tobytes())
        digest.update(str(self.pixels.shape).encode())
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        return digest.hexdigest()

    @property
    def sector_fingerprint(self):
        return json.dumps({'configs': {k: v.raw for k, v in self.sectors.items()},
                           'bindings': self.sector_bindings,
                           'errors': self.sector_load_errors}, sort_keys=True)

    @staticmethod
    def plan_digest(plan):
        return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @property
    def plan_current(self):
        return self.plan is not None and self.plan_fingerprint == self.fingerprint

    def attach_plan(self, plan):
        # Validate before displaying or exporting a planner result.
        from .overlays import plan_layers
        plan_layers(plan)
        signature = trajectory_signature(plan)
        points = plan['global_traj_wpnts_iqp']['wpnts']
        velocities = np.asarray([point['vx_mps'] for point in points], dtype=float)
        if len(points) < 2 or not np.isfinite(velocities).all() or np.any(velocities < 0):
            raise ValueError('Global raceline must have at least two points with finite, nonnegative speeds.')
        self.plan = plan
        self.plan_fingerprint = self.fingerprint
        self.plan_load_error = None
        count = len(plan['global_traj_wpnts_iqp']['wpnts'])
        for kind in FILES:
            if kind not in self.sectors and kind not in self.sector_load_errors:
                self.sectors[kind] = SectorConfig.default(kind, count)
                self.sector_bindings[kind] = signature

    def sector_issues(self, kind=None, check_binding=True):
        issues = []
        for name in ([kind] if kind else FILES):
            if name in self.sector_load_errors:
                issues.append(f'{FILES[name]}: {self.sector_load_errors[name]}')
                continue
            if name not in self.sectors:
                issues.append(f'{FILES[name]}: no configuration; generate a trajectory first.')
                continue
            if self.plan is None:
                issues.append('No global trajectory available.')
                continue
            try:
                self.sectors[name].validate(len(self.plan['global_traj_wpnts_iqp']['wpnts']))
            except (ValueError, KeyError, TypeError) as error:
                issues.append(f'{FILES[name]}: {error}')
            if check_binding and self.sector_bindings.get(name) != trajectory_signature(self.plan):
                issues.append(f'{FILES[name]}: trajectory changed; review indices or reset sections.')
        return issues

    def edit_sectors(self, kind, edit):
        if not self.plan_current:
            raise ValueError('Generate a current trajectory before editing sectors.')
        if self.sector_issues(kind):
            raise ValueError('\n'.join(self.sector_issues(kind)))
        config = deepcopy(self.sectors[kind])
        result = edit(config)
        config.validate(len(self.plan['global_traj_wpnts_iqp']['wpnts']))
        if config.raw != self.sectors[kind].raw:
            self.undo_stack.append(self._snapshot())
            self._bound(self.undo_stack)
            self.redo_stack.clear()
            self.sectors[kind] = config
        return result

    def accept_sector_indices(self, kind, reset=False):
        if not self.plan_current:
            raise ValueError('Generate a current trajectory first.')
        count = len(self.plan['global_traj_wpnts_iqp']['wpnts'])
        if not reset:
            issues = self.sector_issues(kind, check_binding=False)
            if issues:
                raise ValueError('\n'.join(issues))
        self.undo_stack.append(self._snapshot())
        self._bound(self.undo_stack)
        self.redo_stack.clear()
        if reset:
            default = SectorConfig.default(kind, count)
            if kind in self.sectors:
                # Preserve global/unknown parameters and the first section's fields.
                candidate = deepcopy(self.sectors[kind])
                section = deepcopy(candidate.sections[0])
                section.update(start=0, end=count - 1)
                if kind == 'overtaking':
                    section['ot_flag'] = False
                candidate._replace_sections([section])
                try:
                    candidate.validate(count)
                except (ValueError, KeyError, TypeError):
                    # A syntactically readable but invalid file must also be
                    # repairable. Reset required fields, retaining extra fields.
                    for key, value in default.params.items():
                        if key not in ('n_sectors', f'{default.prefix}0', 'ot_sector_begin'):
                            candidate.params[key] = deepcopy(value)
                    for key in ('start', 'end', 'scaling' if kind == 'speed' else 'ot_flag'):
                        section[key] = default.sections[0][key]
                    candidate._replace_sections([section])
                default = candidate
            default.validate(count)
            self.sectors[kind] = default
            self.sector_load_errors.pop(kind, None)
        self.sector_bindings[kind] = trajectory_signature(self.plan)

    def _snapshot(self):
        return (zlib.compress(self.pixels.tobytes(), 1), self.pixels.shape, deepcopy(self.metadata),
                deepcopy(self.sectors), deepcopy(self.sector_bindings), deepcopy(self.sector_load_errors))

    def _restore(self, snapshot):
        data, shape, metadata, sectors, bindings, errors = snapshot
        self.pixels = np.frombuffer(zlib.decompress(data), np.uint8).reshape(shape).copy()
        self.metadata = deepcopy(metadata)
        self.sectors = deepcopy(sectors)
        self.sector_bindings = deepcopy(bindings)
        self.sector_load_errors = deepcopy(errors)

    @staticmethod
    def _bound(stack):
        size = sum(len(s[0]) for s in stack)
        while len(stack) > 1 and (size > 128 * 1024**2 or len(stack) > 100):
            size -= len(stack.pop(0)[0])

    def commit(self, pixels=None, metadata=None):
        pixels = self.pixels if pixels is None else pixels
        metadata = self.metadata if metadata is None else metadata
        validate_metadata(metadata)
        if np.array_equal(pixels, self.pixels) and metadata == self.metadata:
            return False
        self.undo_stack.append(self._snapshot())
        self._bound(self.undo_stack)
        self.redo_stack.clear()
        self.pixels = np.ascontiguousarray(pixels, dtype=np.uint8).copy()
        self.metadata = deepcopy(metadata)
        return True

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self._snapshot())
            self._restore(self.undo_stack.pop())
            self._bound(self.redo_stack)

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._snapshot())
            self._restore(self.redo_stack.pop())
            self._bound(self.undo_stack)

    def free_mask(self):
        occupancy = (255 - self.pixels.astype(np.float32)) / 255.
        return occupancy < float(self.metadata['free_thresh'])

    def clean_noise(self, max_area):
        occupancy = (255 - self.pixels.astype(np.float32)) / 255.
        occupied = (occupancy > float(self.metadata['occupied_thresh'])).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, connectivity=8)
        remove = np.zeros(count, dtype=bool)
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            # Preserve outer walls even when they happen to be small.
            if (area <= max_area and x > 0 and y > 0
                    and x + width < self.pixels.shape[1] and y + height < self.pixels.shape[0]):
                remove[label] = True
        result = self.pixels.copy()
        result[remove[labels]] = 255
        return self.commit(result)

    def pixel_to_world(self, points):
        points = np.asarray(points, dtype=float)
        local = points.copy()
        local[..., 1] = self.pixels.shape[0] - 1 - local[..., 1]
        local *= float(self.metadata['resolution'])
        ox, oy, yaw = self.metadata['origin']
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        return local @ rotation.T + [ox, oy]

    def world_to_pixel(self, points):
        ox, oy, yaw = self.metadata['origin']
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        local = (np.asarray(points, dtype=float) - [ox, oy]) @ rotation
        local /= float(self.metadata['resolution'])
        local[..., 1] = self.pixels.shape[0] - 1 - local[..., 1]
        return local

    def save(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() not in ('.yaml', '.yml'):
            path = path.with_suffix('.yaml')
        metadata = deepcopy(self.metadata)
        metadata['image'] = path.with_suffix('.png').name
        if self.plan_current:
            issues = self.sector_issues()
            if issues:
                raise ValueError('Resolve Sector configuration before exporting this trajectory:\n'
                                 + '\n'.join(issues))
        metadata[VALIDATION_KEY] = {
            'map_fingerprint': self.fingerprint,
            'plan_digest': self.plan_digest(self.plan) if self.plan_current else None,
            'sector_bindings': deepcopy(self.sector_bindings),
            'localization_baseline': deepcopy(self.localization_baseline),
        }
        ok, png = cv2.imencode('.png', self.pixels)
        if not ok:
            raise ValueError('Could not encode map PNG.')
        result = path.parent / 'global_waypoints.json'
        # Never leave an old trajectory looking current after saving edited geometry.
        if result.exists() and not self.plan_current:
            result.rename(result.with_name(f'global_waypoints.stale-{time.time_ns()}.json'))
        atomic_bytes(path.with_suffix('.png'), png.tobytes())
        atomic_bytes(path, yaml.safe_dump(metadata, sort_keys=False).encode())
        if self.plan_current:
            atomic_bytes(result, json.dumps(self.plan, allow_nan=False).encode())
        for kind, config in self.sectors.items():
            atomic_bytes(path.parent / FILES[kind], config.dumps())
        self.metadata = metadata
        self.path = path
        self.saved_fingerprint = self.fingerprint
        self.saved_sector_fingerprint = self.sector_fingerprint
        return path
