"""FTG's forward-distance speed law reused by path supervision."""

import math


def ftg_speed_target(distance, max_speed, deceleration, clearance):
    """Preserve the standalone FTG distance law and braking-speed envelope."""
    if (not math.isfinite(distance) or distance < 0 or math.isnan(max_speed)
            or max_speed <= 0 or not math.isfinite(deceleration) or deceleration <= 0
            or not math.isfinite(clearance) or clearance < 0):
        raise ValueError('Invalid FTG speed input')
    if distance < 1.2:
        target = 0.96
    elif distance < 4.0:
        target = 0.9 * distance + 0.2
    else:
        target = max_speed
    braking_speed = math.sqrt(2.0 * deceleration * max(distance - clearance, 0.0))
    return min(target, max_speed, braking_speed)
