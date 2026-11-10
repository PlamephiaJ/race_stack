# GPU particle-filter localization

The default is `synpf`; `pf` selects the earlier `vtc26_ws` backend, and
`cartographer` selects Cartographer. All are executables/backends of the existing
state-estimation launch; no second ROS package or EKF is introduced.

| Backend | Config in stack_master/config | Particle pose | Default model | Scan selection |
| --- | --- | --- | --- | --- |
| `synpf` (default) | `synpf.yaml` | base_link | TUM, motion-dependent noise | 21 boxed beams |
| `pf` | `particle_filter.yaml` | laser | fixed Gaussian noise | uniform angle step |

SynPF source: [ForzaETH/particle_filter](https://github.com/ForzaETH/particle_filter),
pinned at `85ecc1865ac687ad8cf31afadbcb826acb23262c`. See
[porting notes](third_party/synpf/SOURCE.md) and [MIT license](third_party/synpf/LICENSE).
The ROS 1 transport is replaced with the shared ROS 2 TF/QoS/scan-time transport.
TUM (default) and AMCL motion models preserve signed turns and reverse motion;
the upstream experimental arc model also has a finite straight-line limit.
The sensor model includes normalized Gaussian hits, exponential short readings,
max-range returns and random returns. In SynPF, `sigma_hit` is in metres; it is
in pixels in the legacy PF. Do not copy this parameter between configs unchanged.
SynPF weights base hypotheses in occupied/unknown/outside-map cells by 0.01.
It publishes base pose on `/tracked_pose` and optional covariance on
`/tracked_pose/with_covariance`, in addition to the shared car-state interfaces.

The [paper](https://arxiv.org/abs/2401.07658) reports racing/wheel-slip results
against Cartographer, not against this workspace's PF. No track-data comparison
between these two ports has been performed, and paper latency numbers should not
be assumed on this Jetson/GPU configuration.

The legacy PF details follow. Shared GPU, EKF and mapping behavior applies to
both backends; their hypotheses are expressed in the frames listed above.


Ported from `vtc26_ws/src/particle_filter` (Corey Walsh / Hongrui Zheng,
MIT license). The Monte Carlo motion/sensor model, RangeLibc ray casting,
circular pose statistics and visualization topics are retained. The implementation
is the `particle_filter` executable inside `state_estimation`; there is no separate
ROS package or second state-estimation stack.

The race-stack integration uses `/scan` and the EKF's `odom -> base_link` TF
at each scan timestamp. Each scan is evaluated once. Particles represent the
laser pose; the static mounting transform is included in motion, map-to-odom
publication and RViz `/initialpose` resets (which specify the **base** pose).
Scans and initial poses accept best-effort publishers. `/clicked_point`
reinitializes globally. Initial localization is global; use RViz's 2D Pose
Estimate to select the right part of a repetitive track.

Configuration: `stack_master/config/particle_filter.yaml`. PF requires
`range_method: rmgpu`, `rangelib_variant: 2` and a CUDA-enabled RangeLibc build.
Startup rejects CPU builds, inaccessible GPUs and unsupported methods; there is
no CPU fallback. Ray casting runs on CUDA. Resampling, the motion model, sensor
likelihood evaluation and pose statistics retain their NumPy/native CPU paths.
This does not claim the entire PF runs on the GPU.

The existing EKF is unchanged: wheel speed and IMU angular velocity Z are fused
into local odometry. PF consumes its `odom -> base_link` TF at scan time; it does
not fuse raw IMU again. The source workspace's PF used raw wheel odometry and
did not subscribe directly to IMU. Integration fixes include scan/TF time matching,
a bounded delayed-TF queue, one observation update per scan, laser mounting-arm
motion, base-to-laser initial pose conversion and finite-weight recovery.
Accuracy improvements have not been measured on track.
PF publishes `map -> odom`; the EKF remains the sole owner of
`odom -> base_link`. The stack publishes `/car_state/pose` from TF as before.
`/pf/laser_global_pose` is a laser pose observation, not an EKF input.

Install and build (ROS 2 Humble Python):

```bash
sudo apt-get install ros-humble-slam-toolbox
/usr/bin/python3 -m pip install 'cython<3' 'numpy<2' transforms3d
bash .install_utils/particle_filter_setup.sh "$PWD/range_libc"
# Omit the source argument to fetch the pinned RangeLibc revision instead.
cd ~/ws
colcon build --symlink-install --packages-select state_estimation stack_master global_planner
source install/setup.bash
```

The Jetson Docker image builds RangeLibc for CUDA sm_87 (Orin), adds the video
supplementary group and prioritizes the injected host driver over CUDA's older
compat library. `CUDA_ARCH` overrides the target for other GPUs. CPU-only Docker
images can still run Cartographer; GPU PF requires a CUDA toolkit image.
Build with the ROS Python interpreter.

Jetson GPU devices must also be accessible to the runtime user. After adding the
video group, run `newgrp video` (or recreate the container/session) to refresh
supplementary groups. A new VSCode terminal alone may inherit the old groups.
The host's DRM render device GIDs can differ from the container's `render` group.
On this host `/dev/dri/renderD128` and `/dev/dri/renderD129` belong to GID 104,
while the container's `render` group is 109. Their access must be explicitly
approved/configured on the host; membership in `video` alone is insufficient. Device ACLs may need to be reapplied when the host recreates the device nodes.
The node checks CUDA access and stops with an error if it is denied. It never launches a CPU substitute.

```bash
ros2 launch stack_master base_system_launch.xml sim:=false racecar_version:=BLUE map_name:=track_wmn loc_algorithm:=pf
ros2 launch stack_master mapping_launch.xml racecar_version:=BLUE map_name:=new_track loc_algorithm:=pf
```

PF is a localizer for saved occupancy maps. The `pf` and `synpf` mapping options select
`slam_toolbox`, as in `vtc26_ws/run_slam.sh`; it does not run PF against a changing
map. Mapping uses the existing filter and terminal `y` save operation. It writes
`<name>.png`, `<name>.yaml`, `pf_map.png` and `pf_map.yaml`, including the base's
`initial_pose`, without a `.pbstream` or a serialized slam_toolbox pose graph.
The saved map can be used directly by PF, the map editor, global planner and
simulator. Cartographer localization needs a map with a matching `.pbstream`.
