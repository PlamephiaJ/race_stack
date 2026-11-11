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
`psi_rad` with odom yaw. Missing/expired/invalid inputs stop the vehicle and
restart acquisition at low speed.

Front speed is now controlled continuously before an emergency occurs.
The distance speed law matches standalone FTG: below 1.2 m it requests
0.96 m/s, between 1.2 and 4 m it requests `0.9 * distance + 0.2`, and it is
bounded by `sqrt(2 * deceleration * max(distance - clearance, 0))`.
The path/trailing/recovery request remains an upper bound. The supervisor
also caps speed to `0.8 * distance / (1.0 + scan_age)` to leave headroom above
the emergency TTC boundary rather than accelerating straight back into it.
The existing actuator slew limiter still controls the physical output rate.

Only the collision emergency gate uses the explicit time rule:
`TTC = max(0, front_distance / measured_forward_speed - scan_age)`.
It triggers strictly below **1.0 s** and releases above **1.2 s**, including
while still moving. It does not wait for a standstill. Applied speed references
are not treated as actual motion, and desired clearance is not subtracted from
the TTC range. At rest/reversing TTC is infinite. These replace the former
distance-threshold/full-stop-release loop that caused 11 stop entries in the
first 20 s of bag `022513`.

The continuous FTG braking cap uses the BLUE baseline 0.30 m clearance and
0.8 m/s² effective deceleration, below the approximately 1.0 m/s² observed in
`014101`. These do not change physical braking torque. Standalone FTG's node
and its own parameters are separate; this change borrows its speed curve.

On real cars, absent/stale `/ackermann_cmd_applied` feedback stops the controller.
Direct-drive gym simulation defaults `require_applied_feedback` to false.
During braking, fresh usable forward path geometry still updates steering;
MAP uses measured speed for that lookup. At rest or with invalid/stale inputs,
the last valid steering is retained. The actuator's steering bounds/rate limits
still apply. The front distance is the fourth-nearest valid beam within
±12 degrees. TTC is a forward-range/current-speed estimate, not a prediction
of the full turning/swept footprint. Invalid inputs can independently stop the
controller even when the collision TTC gate is not active.

Build the updated package before launching:

```bash
cd ~/ws
colcon build --packages-select controller --symlink-install
source install/setup.bash
```

For a read-only regression against a recorded bag, from the repository root:

```bash
PYTHONPATH="$PWD/controller:$PYTHONPATH" python3 controller/scripts/check_recorded_safety.py \
  bags/control_fault_20261111_014101_150285/rosbag/rosbag_0.db3
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

 ## Parameters
 - `track_width`: Approximate width of the track.
 - `max_speed`: Maximum speed allowed.
 - `safety_radius`: Safety radius to reduce cutting corners.
 - `max_lidar_dist`: Maximum possible scan distance of the LiDAR.
 - `range_offset`: Range of LiDAR points to be considered, only consider `[range_offset,-range_offset]`.
 - `debug`: If set to `True`, the best gap, best point and preprocessed LiDAR scans are published as Markers.
