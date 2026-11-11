# Controller Manager
![Controller Architecture](./misc/controller_arch.png)

This controller package implements the following controllers:
- [MAP controller](#model--and-acceleration-based-pursuit-map-controller)
- [Pure Pursuit controller](#pure-pursuit-pp-controller)
- [Follow the Gap controller](#follow-the-gap-ftg-controller)

The `controller_manager` implemented in `controller_manager.py` initializes the needed controllers. It runs at a specified loop rate where in each cycle the next control inputs are calculated via the choosen controller.

## Input/Output Topic Signature
This nodes subscribes to:
- `/perception/obstacles`: Subscribes to the obstacle array.
- `/state_estimation/odom`: Reads the car's velocity.
- `/car_state/frenet/odom`: Reads the car's state in the Frenet frame.
- `/global_waypoints`: Subscribes to global waypoints.
- `/local_waypoints`: Subscribes to local waypoints.
- `/state`: Current state from the state machine.
- `/vesc/sensors/imu/raw`: Reads the IMU measurements.
- `/scan`: Reads the LiDAR scans.
- `/ackermann_cmd_applied`: Reads the applied speed reference on the real car.
- `/l1_param_cb`: Listens to the L1 parameters.


The node publishes to:
- `/drive`: Publishes the control commands.
- `/lookahead_point`: Publishes lookahead point marker.
- `/steering`: Publishes steering arrow marker.
- `/trailing_opponent_marker`: Published trailing opponend marker.
- `/my_waypoints`: Publishes marker array of the received waypoints.
- `/l1_distance`: Publishes the L1 distance.
- `/trailing/gap_data`: Publishes the PID data for trailing.

## Tracking recovery and stopping

MAP and PP share the supervisor in `controller_manager.py`. The launch file
loads [tracking_safety.yaml](config/tracking_safety.yaml); use the
`safety_param_file` argument when including `controller_launch.xml` to override it.

When the car is traveling along or toward the path, speed is interpolated
between the path request and a 0.5 m/s recovery cap. The interpolation starts
at 0.35 m distance or 25 degrees heading error and reaches the recovery cap
at 1.5 m or 90 degrees. The larger of the two normalized errors controls the
interpolation. There is no latched low-speed state or tight exit threshold.
The interpolated cap changes at up to 1.0 m/s² to smooth geometry/TF changes.

Direction uses both the side of the path and the heading's normal component:
travel toward the path and parallel travel are accepted, including up to
10 degrees of outward heading noise. Within 0.35 m, heading alignment alone
is used to avoid side-switch noise. Clearly diverging/reversed motion is
directly capped at 0.5 m/s; when direction improves the cap ramps up again.
For example, a 2.23 m/s request at 0.79 m offset and 36 degrees heading error
gets approximately 1.57 m/s when converging, versus 0.5 m/s when diverging.
Lower raceline/trailing limits and zero requests still take priority: 0.5 m/s
is the minimum recovery cap, not a forced minimum vehicle speed.
Geometry uses local x/y and vehicle pose in odom; it does not compare map-frame
`psi_rad` with odom yaw. Missing/expired/invalid inputs fade the reference
toward zero and restart acquisition at low speed.

The collision speed cap is now purely linear:
`v_cap = collision_speed_gain * max(current_front_distance - front_clearance_m, 0)`.
Defaults are 0.8 s⁻¹ and 0.30 m; scan age is compensated using measured forward
speed. The path/trailing/recovery target remains an upper bound. There are no
TTC stop/release thresholds, front-stop latch or rear-path-start stop test.

Every MAP/PP speed target passes through one final reference smoother, including
zero goals caused by a missing path or invalid input. Both acceleration and
deceleration of this reference default to 1.0 m/s²; long scheduler pauses cannot
skip the ramp. Fresh applied-command feedback prevents winding up a high
reference while actuation is gated off. This changes the autonomous reference,
not the physical motor's braking capability.

Input freshness also uses a continuous confidence scale: full speed until half
its configured age limit, then linearly down to zero confidence at the limit.
When no trustworthy geometry remains, retain the last valid steering and fade
the speed reference toward zero. A temporarily missing path never bypasses the
smoother with an immediate zero command. The local planner rebases localization
jumps immediately with a 0.5 m/s recovery path, without publishing a stop path
or entering a timed hold. Explicit initial-pose waiting and unavailable TF skip
path publication; the controller's input-age ramp handles the gap.

The front range is the fourth-nearest valid beam within ±12 degrees. This is a
forward range estimate, not a swept-footprint collision prediction. The recovery
0.5 m/s cap is not a minimum vehicle speed: collision clearance and valid lower
path targets can reduce it further, continuously. Manual deadman/teleop gates,
shutdown handling and the driver's command watchdog remain physical stop
mechanisms, independent of this autonomous speed law. Standalone FTG is separate.

Build the updated package before launching:

```bash
cd ~/ws
colcon build --packages-select controller local_planner --symlink-install
source install/setup.bash
```

For a read-only regression against a recorded bag, from the repository root:

```bash
PYTHONPATH="$PWD/controller:$PYTHONPATH" python3 controller/scripts/check_recorded_safety.py \
  bags/control_fault_20261111_023931_197022/rosbag/rosbag_0.db3
```

The check uses recorded inputs with BLUE L1 parameters and the selected MAP
lookup table (default `NUC2_hangar_pacejka`). It writes decisions to CSV without
publishing ROS messages. It is open-loop: it cannot validate a changed vehicle
trajectory or replace a low-speed physical stopping test.

# Model- and Acceleration-based Pursuit (MAP) controller
The MAP Controller based on [Model- and Acceleration-based Pursuit Controller for High-Performance Autonomous Racing](https://arxiv.org/pdf/2209.04346.pdf) calculates and publishes the optimal steering angle based on a vehicle model and velocity to track a given trajectory. It is managed by the `controller_manager.py` node.

## Longitudinal controller
The speed from the global optimizer is propagated to compensate for the delay and scaled according to the lateral error and the curvature of the waypoints. 


`<parameter_name>`: description [minimum, maximum, tuned]

- `speed_lookahead`: Lookahead time in seconds to account for actuation and computation delay [0, ∞, 0.25]
- `lat_err_coeff`: How much of the lateral error is taken into account to smoothly rejoin the trajectory. Higher values increase the dependence of the lateral error on the speed reduction. [0, 1, 1]


### Trailing Controller

- `trailing_gap`: reference gap to the to-be-trailed opponent in meters[0, ∞, 1.5] 
- `trailing_p_gain`: P-Gain of trailing controller [0, ∞, 1]
- `trailing_i_gain`: I-Gain of trailing controller [0, ∞, 0]
- `trailing_d_gain`: D-Gain of trailing controller [0, ∞, 0.2]
- `blind_trailing_speed`: Minimal Trailing speed when opponent not in line-of-sight [0, ∞, 1.5]
- `prioritize_dyn`: Ignore static obstacles, when a dynamic obstacle is present [False, True, True]


## Lateral controller
It uses the L1 guidance described in this paper: [Model- and Acceleration-based Pursuit Controller for High-Performance Autonomous Racing](https://arxiv.org/abs/2209.04346). 

The idea is to compute the lateral acceleration and covert this with a lookup table into steering angles.
To prevent oscillations when driving fast and to be able to track small curvatures, the L1 is scaled with delay compensated speed. For the lookup, the speed is propagated to compensate for the steering delay.
When accelerating faster then 1 m/s^2, the steering is reduced and for deacceleration higher than -1m/s^2 the steering is increased.


- `m_l1`: Proportional term for the affine mapping of the velocity to the lookahead distance
for the MAP controller [0, ∞, 0.6] 
- `q_l1`: Offset term for the affine mapping of the velocity to the lookahead distance. [−∞, ∞, -0.18]
- `t_clip_min`: clipping value for minimal L1 distance [0, ∞, 0.8]
- `t_clip_max`: clipping value for maximal L1 distance [0, ∞, 5]

![visualisation of L1 distance calculation](misc/l1_dist.png)

At high speeds the steering is downscaled by a factor to reduce tire slipping 
- `start_scale_speed`: start of downscaling speed range [0, ∞, 7]
- `end_scale_speed`: end of downscaling speed range [0, ∞, 8]
- `downscale_factor`: percent of steer downscaling [0, 100, 20]

![visualisation of speed scaling](misc/speed_scaling.png)
- `acc_scaler_for_steer`: when accelerating stronger than 1m/s^2 the steering is increased by this factor [1, ∞, 1.2]
- `dec_scaler_for_steer`: when accelerating stronger than -1m/s^2 the steering is reduced by this factor [0, 1, 0.9]
- `speed_lookahead_for_steer`: Lookahead time in seconds propagate velocity, for steering lookup. Can increase laptime by cutting more corners [0, 0.2, 0]


## Basic Use

The controller can be started by using the following command:
```
ros2 launch controller controller_launch.xml mode:=MAP
```

However we suggest to use the launch files present in the `stack_master package`.

# Pure Pursuit (PP) controller
The goal of the Control module within the ForzaETH Race Stack is to enable performant, safe, and consistent lap completions by accurately tracking a trajectory provided by the local planner.
The velocity and steering commands are managed, respectively, by the longitudinal and lateral controllers.

## Longitudinal controller
The longitudinal controller computes the desired velocity of the racecar. This velocity is either obtained directly from the velocity of the local trajectory or computed by the trailing controller if the car is trailing an opponent and therefore in the Trailing state.

The speed from the local trajectory is propagated to compensate for the delay and scaled according to the lateral error and the curvature of the waypoints.


### Parameters
- `<parameter_name>`: description [minimum, maximum, tuned]

- `speed_lookahead`: Lookahead time in seconds to account for actuation and computation delay [0, ∞, 0.25]

- `lat_err_coeff`: How much of the lateral error is taken into account to smoothly rejoin the trajectory. Higher values increase the dependence of the lateral error on the speed reduction. [0, 1, 1]


### Trailing Controller Parameters


- `trailing_gap`: reference gap to the to-be-trailed opponent in meters[0, ∞, 1.5] 

- `trailing_p_gain`: P-Gain of trailing controller [0, ∞, 1]

- `trailing_i_gain`: I-Gain of trailing controller [0, ∞, 0]

- `trailing_d_gain`: D-Gain of trailing controller [0, ∞, 0.2]

- `blind_trailing_speed`: Minimal Trailing speed when opponent not in line-of-sight [0, ∞, 1.5]

- `prioritize_dyn`: Ignore static obstacles, when a dynamic obstacle is present [False, True, True]


## Lateral controller
The Pure Pursuit controller computes the steering angle via geometric considerations.
To prevent oscillations when driving fast and to be able to track small curvatures, the L1 is scaled with delay compensated speed. For the steering angle caculations, the speed is propagated to compensate for the steering delay.
When accelerating faster then 1 m/s^2, the steering is reduced and for deacceleration higher than -1m/s^2 the steering is increased.

### Parameters

- `m_l1`: Proportional term for the affine mapping of the velocity to the lookahead distance
for the PP controller [0, ∞, 0.6] 

- `q_l1`: Offset term for the affine mapping of the velocity to the lookahead distance. [−∞, ∞, -0.18]

- `t_clip_min`: clipping value for minimal L1 distance [0, ∞, 0.8]

- `t_clip_max`: clipping value for maximal L1 distance [0, ∞, 5]

![visualisation of L1 distance calculation](misc/l1_dist.png)

At high speeds the steering is downscaled by a factor to reduce tire slipping 

- `start_scale_speed`: start of downscaling speed range [0, ∞, 7]

- `end_scale_speed`: end of downscaling speed range [0, ∞, 8]

- `downscale_factor`: percent of steer downscaling [0, 100, 20]

![visualisation of speed scaling](misc/speed_scaling.png)

- `acc_scaler_for_steer`: when accelerating stronger than 1m/s^2 the steering is increased by this factor [1, ∞, 1.2]

- `dec_scaler_for_steer`: when accelerating stronger than -1m/s^2 the steering is reduced by this factor [0, 1, 0.9]

- `speed_lookahead_for_steer`: Lookahead time in seconds propagate velocity, for steering lookup. Can increase laptime by cutting more corners [0, 0.2, 0]


## Basic Use

The controller can be started by using the following command:
```
ros2 launch controller controller_launch.xml mode:=PP
```

However we suggest to use the launch files present in the `stack_master package`.


# Follow The Gap (FTG) controller

## Description
The Follow-The-Gap (FTG) Controller is a simple reactive controller which uses directly the LiDAR scans and the car's pose to compute and publish the velocity and steering angle of the car. In our race stack, the FTG controller is mainly used for mapping, so that we do not have to drive manually around the track. However, even though never used in practice anymore, there is the possibility to set a sector in the state machine as `FTGONLY` and then the FTG controller managed by `controller_manager.py` is used during these sectors.

## Standalone FTG configuration

The current ego/opponent launch scripts run `reactive_node`
(`controller/follow_the_gap.py`) through `stack_master/launch/ftg_launch.xml`.
All startup tuning parameters are listed with units and constraints in
`stack_master/config/follow_the_gap_params.yaml`:

| Group | Parameters |
| --- | --- |
| LiDAR and clearance | `bubble_radius`, `max_distance`, `disparity_thresh`, `scan_range_offset`, `preprocess_conv_size` |
| Gap selection | `minimum_gap_angle`, `gap_weight_power` |
| Speed measurement | `front_distance_index` (540 by default; -1 uses the forward beam from scan geometry) |
| Speed curve and dead zone | `max_speed`, `speed_distance_gain`, `braking_stop_margin` |
| Command acceleration/deceleration | `max_acceleration`, `max_deceleration`, `max_command_dt` |
| Steering | `steering_angle_to_servo_gain`, `steering_angle_to_servo_offset`, `servo_min`, `servo_max`, `servo_limit_margin` |
| ROS and shutdown | `debug`, `publish_queue_depth`, `stop_publish_count`, `stop_publish_interval` |

Restart FTG after editing; these parameters are read-only while running.
The three existing launch arguments `max_speed`, `max_deceleration`, and
`minimum_gap_angle` take precedence over YAML. `param_file:=/path/to/file.yaml`
selects an alternative configuration. Use ROS remappings for topics.
Real-car steering calibration comes from the selected vehicle's `vesc.yaml`;
optional calibration overrides in the FTG file are commented to preserve it.

The speed curve uses the filtered/clipped front range before obstacle
bubbles, so steering masks cannot force the speed distance to zero.
The target is `min(speed_distance_gain * max(front_distance - braking_stop_margin, 0), max_speed)`.
The default 0.30 m dead zone gives a zero target; beyond it, speed increases
continuously with a slope of 0.9 (m/s)/m. Command acceleration is limited to
`max_acceleration` (default 1.0 m/s^2), including startup: the first scan
publishes zero speed, then subsequent scans ramp toward the distance target.
The command deceleration limiter controls how quickly a previous command
decreases, including entry into the dead zone. Both limits use elapsed monotonic
time capped by `max_command_dt`. Steering continues to update independently.
Shutdown still publishes zero-speed commands.

## Historical controller-manager FTG configuration

`controller/ftg.py` is a separate historical implementation selected by
`controller_manager` mode `FTG`. Its tuning file is `controller/config/legacy_ftg.yaml`,
loaded by `controller_launch.xml` (override with `ftg_param_file`). The file
exposes filter size, scan geometry, disparity threshold, search-radius formula,
speed scale/ratios, mapping speed, steering thresholds/limit, marker frame/sizes,
and publisher queue depth. It preserves historical values; the 30-degree corner
threshold exceeds the 0.4-rad steering clip, so that speed branch is unreachable
until the thresholds or clip are changed.

The six original parameters can also be set under `legacy_ftg` in that file:
`track_width` (m), `max_speed` (m/s), `safety_radius` (beams, integer >= 2),
`max_lidar_dist` (m), `range_offset` (beams cropped from each end), and `debug`
(publishes RViz markers). When omitted, they retain their existing source on the
`state_machine` node. This file has no effect on standalone FTG. The historical
mode still depends on the controller-manager/state-machine setup; extracting
its parameters does not enable it as a standalone node.
