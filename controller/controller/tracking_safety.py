"""Geometry and speed envelopes shared by MAP and PP supervision."""

import math

import numpy as np


def path_tracking_errors(pose, waypoints):
    """Return distance, absolute tangent error, and motion toward the path.

    Use transformed x/y rather than Wpnt.psi_rad, which may still be in map.
    The third value is the heading's unit normal component toward the path:
    positive converges, zero runs parallel, negative diverges. This distinguishes
    opposite sides of the path with the same absolute heading error.
    Degenerate or non-finite geometry cannot authorize motion.
    """
    pose = np.asarray(pose, dtype=float)
    points = np.asarray(waypoints, dtype=float)
    if (pose.shape != (3,) or points.ndim != 2 or len(points) < 2
            or points.shape[1] < 2 or not np.isfinite(pose).all()
            or not np.isfinite(points).all()):
        return None
    segments = np.diff(points[:, :2], axis=0)
    lengths2 = np.sum(segments * segments, axis=1)
    valid = lengths2 > 1e-8
    if not valid.any():
        return None
    starts, segments, lengths2 = points[:-1, :2][valid], segments[valid], lengths2[valid]
    fraction = np.clip(np.sum((pose[:2] - starts) * segments, axis=1) / lengths2, 0, 1)
    distances = np.linalg.norm(starts + fraction[:, None] * segments - pose[:2], axis=1)
    nearest = int(np.argmin(distances))
    tangent = math.atan2(segments[nearest, 1], segments[nearest, 0])
    heading = math.remainder(pose[2] - tangent, 2 * math.pi)
    offset = pose[:2] - starts[nearest]
    side = segments[nearest, 0] * offset[1] - segments[nearest, 1] * offset[0]
    approach = -math.copysign(1.0, side) * math.sin(heading) if abs(side) > 1e-8 else 0.0
    return float(distances[nearest]), abs(heading), approach


class RecoverySpeedLimiter:
    """Interpolate the speed cap when aligned/converging; crawl when diverging."""

    def __init__(self, speed=0.5, full_speed_distance=0.35, min_speed_distance=1.5,
                 full_speed_heading=math.radians(25), min_speed_heading=math.radians(90),
                 direction_tolerance=math.radians(10), release_acceleration=1.0):
        values = (speed, full_speed_distance, min_speed_distance, full_speed_heading,
                  min_speed_heading, direction_tolerance, release_acceleration)
        if (not all(math.isfinite(v) and v > 0 for v in values)
                or full_speed_distance >= min_speed_distance
                or not full_speed_heading < min_speed_heading <= math.pi / 2
                or direction_tolerance >= math.pi / 2):
            raise ValueError('Invalid path recovery limits')
        self.speed = speed
        self.full_speed_distance, self.min_speed_distance = full_speed_distance, min_speed_distance
        self.full_speed_heading, self.min_speed_heading = full_speed_heading, min_speed_heading
        self.direction_tolerance = direction_tolerance
        self.release_acceleration = release_acceleration
        self.recovering = False
        self.cap = float('inf')

    def limit(self, requested_speed, errors, dt):
        if (errors is None or len(errors) != 3 or not all(math.isfinite(v) for v in errors)
                or not math.isfinite(requested_speed) or not math.isfinite(dt)):
            self.recovering = True
            self.cap = self.speed
            return 0.0
        distance, heading, approach = errors
        # Parallel motion and a little heading noise remain acceptable. Near the
        # path, its side changes frequently, so use heading alignment alone.
        correct_direction = (heading < self.min_speed_heading
                             and (distance <= self.full_speed_distance
                                  or approach >= -math.sin(self.direction_tolerance)))
        distance_weight = np.clip((distance - self.full_speed_distance)
                                  / (self.min_speed_distance - self.full_speed_distance), 0, 1)
        heading_weight = np.clip((heading - self.full_speed_heading)
                                 / (self.min_speed_heading - self.full_speed_heading), 0, 1)
        weight = max(distance_weight, heading_weight) if correct_direction else 1.0
        target_cap = self.speed + (max(requested_speed, self.speed) - self.speed) * (1.0 - weight)
        if not correct_direction:
            # A clearly wrong direction gets the configured crawl cap directly.
            self.cap = self.speed
        elif not math.isfinite(self.cap):
            self.cap = target_cap
        else:
            # Smooth geometry/TF changes instead of switching in/out of a latch.
            change = self.release_acceleration * min(max(dt, 0.0), 0.1)
            self.cap += float(np.clip(target_cap - self.cap, -change, change))
        result = min(max(requested_speed, 0.0), self.cap)
        self.recovering = result < max(requested_speed, 0.0) - 1e-6
        return result


def collision_speed_limit(distance, speed, scan_age, clearance=0.3, gain=0.8):
    """Continuous linear speed cap from current forward free distance."""
    values = (distance, speed, scan_age, clearance, gain)
    if (not all(math.isfinite(v) for v in values) or distance < 0
            or scan_age < 0 or clearance < 0 or gain <= 0):
        raise ValueError('Invalid collision speed input')
    current_distance = max(0.0, distance - max(speed, 0.0) * scan_age)
    return gain * max(current_distance - clearance, 0.0)


def freshness_scale(age, limit):
    """Fade linearly from full speed at half the age limit to zero at the limit."""
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Invalid freshness limit')
    if not math.isfinite(age):
        return 0.0
    return float(np.clip(2.0 * (limit - max(0.0, age)) / limit, 0, 1))


class SpeedSmoother:
    """Bound autonomous reference changes, including zero/invalid-input targets."""

    def __init__(self, acceleration=1.0, deceleration=1.0):
        if not all(math.isfinite(v) and v > 0 for v in (acceleration, deceleration)):
            raise ValueError('Invalid speed slew limits')
        self.acceleration, self.deceleration = acceleration, deceleration
        self.speed = None

    def update(self, target, dt, applied_speed=None):
        if not math.isfinite(target):
            target = 0.0
        if self.speed is None:
            self.speed = max(0.0, applied_speed) if applied_speed is not None and math.isfinite(applied_speed) else 0.0
        dt = min(max(dt, 0.0), 0.1) if math.isfinite(dt) else 0.0
        # Avoid winding the reference up while the physical deadman gate is off.
        if applied_speed is not None and math.isfinite(applied_speed):
            target = min(target, max(0.0, applied_speed) + self.acceleration * .1)
        delta = max(target, 0.0) - self.speed
        self.speed += float(np.clip(delta, -self.deceleration * dt, self.acceleration * dt))
        return self.speed
