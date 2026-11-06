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

MAX_PIXELS = 64_000_000


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
        result = path.parent / 'global_waypoints.json'
        if result.is_file():
            try:
                document.attach_plan(json.loads(result.read_text()))
            except (ValueError, KeyError, TypeError):
                pass  # A damaged optional trajectory must not prevent map editing.
        return document

    @property
    def fingerprint(self):
        metadata = {k: v for k, v in self.metadata.items() if k != 'image'}
        digest = hashlib.sha256(self.pixels.tobytes())
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        return digest.hexdigest()

    @property
    def dirty(self):
        return self.fingerprint != self.saved_fingerprint

    @property
    def plan_current(self):
        return self.plan is not None and self.plan_fingerprint == self.fingerprint

    def attach_plan(self, plan):
        # Validate before displaying or exporting a planner result.
        from .overlays import plan_layers
        plan_layers(plan)
        self.plan = plan
        self.plan_fingerprint = self.fingerprint

    def _snapshot(self):
        return (zlib.compress(self.pixels.tobytes(), 1), self.pixels.shape, deepcopy(self.metadata))

    def _restore(self, snapshot):
        data, shape, metadata = snapshot
        self.pixels = np.frombuffer(zlib.decompress(data), np.uint8).reshape(shape).copy()
        self.metadata = deepcopy(metadata)

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
        self.metadata = metadata
        self.path = path
        self.saved_fingerprint = self.fingerprint
        return path
