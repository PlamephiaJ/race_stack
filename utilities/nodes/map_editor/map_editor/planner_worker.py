"""One-shot offline adapter around the existing GlobalPlannerNode.plan algorithm.

No rclpy.init(), ROS node, publishers, vehicle, SLAM, or DDS traffic is needed.
The imports still require the built planner and its ROS message dependencies.
"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
from types import FunctionType

import cv2
import numpy as np

from .document import MapDocument, atomic_bytes
from .overlays import plan_layers, transform_plan


def run(map_yaml, config_dir, work_dir, safety_width, safety_width_sp, reverse=False):
    # Must precede the planner import, which otherwise selects the interactive Tk backend.
    os.environ['RACE_STACK_PLANNER_HEADLESS'] = '1'
    from geometry_msgs.msg import Point
    from skimage.morphology import skeletonize
    from global_planner.global_planner_node import GlobalPlannerNode
    from global_racetrajectory_optimization.trajectory_optimizer import trajectory_optimizer

    document = MapDocument.load(map_yaml)
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / 'global_waypoints.json'
    output.unlink(missing_ok=True)
    if not all(math.isfinite(w) and w > 0 for w in (safety_width, safety_width_sp)):
        raise ValueError('Safety widths must be positive and finite.')
    if not (Path(config_dir) / 'racecar_f110.ini').is_file():
        raise ValueError(f'Missing planner configuration: {config_dir}/racecar_f110.ini')

    class OfflinePlanner:
        # Reuse the complete existing algorithm, including all waypoint serialization.
        plan = GlobalPlannerNode.plan

        def get_logger(self):
            return logging.getLogger('map_editor.planner')

        def publish_plan(self, **_messages):
            pass

    planner = OfflinePlanner()
    planner.work_dir = work_dir
    # The vendored optimizer uses its module __file__ only to locate export folders.
    # Keep the exact callable bytecode, with private globals for this worker's paths.
    # This avoids editing the optimizer submodule or writing to its installation.
    optimizer_globals = dict(trajectory_optimizer.__globals__, __file__=str(work_dir / 'optimizer.py'))
    planner.trajectory_optimizer = FunctionType(
        trajectory_optimizer.__code__, optimizer_globals, trajectory_optimizer.__name__,
        trajectory_optimizer.__defaults__, trajectory_optimizer.__closure__)
    planner.map_dir = str(work_dir)
    planner.input_path = str(Path(config_dir).resolve())
    planner.map_resolution = float(document.metadata['resolution'])
    # Unknown cells are excluded from free space before skeletonization.
    planner.filtered_map = cv2.flip(document.free_mask().astype(np.uint8) * 255, 0)
    planner.skeleton = np.asarray(skeletonize(planner.filtered_map, method='lee'), dtype=np.uint8)
    planner.map_origin = Point()  # Compute in local axes; transform the result for rotated maps.
    planner.initial_pose_missing = 'initial_pose' not in document.metadata
    px, py, heading = document.metadata.get('initial_pose', [0., 0., 0.])
    ox, oy, yaw = document.metadata['origin']
    c, s = math.cos(yaw), math.sin(yaw)
    planner.initial_position = [c * (px - ox) + s * (py - oy), -s * (px - ox) + c * (py - oy), heading - yaw]
    planner.reverse_mapping = bool(reverse)
    planner.show_plots = False
    planner.map_editor_mode = True
    planner.safety_width = float(safety_width)
    planner.safety_width_sp = float(safety_width_sp)
    planner.watershed = True
    planner.map_info_str = ''
    result = planner.plan()
    if result is False or not output.is_file():
        raise RuntimeError('Planner did not produce a trajectory. Check track closure and safety widths.')
    plan = transform_plan(json.loads(output.read_text()), document.metadata['origin'])
    plan_layers(plan)
    atomic_bytes(output, json.dumps(plan, allow_nan=False).encode())
    print(f'Planner complete: {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--safety-width', type=float, default=0.7)
    parser.add_argument('--safety-width-sp', type=float, default=0.7)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
    run(args.map, args.config, args.work_dir, args.safety_width, args.safety_width_sp, args.reverse)


if __name__ == '__main__':
    main()
