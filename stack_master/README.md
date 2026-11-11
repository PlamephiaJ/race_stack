# Stack Master
Here is the `stack_master`, it is intended to be the main interface between the user and the PBL ForzaETH F110 system.

## Zenoh
We use [zenoh](https://github.com/ros2/rmw_zenoh) as ros2 middleware. The zenoh router is started as a vscode task when you open the container. You can also start it manually with: `ros2 run rmw_zenoh_cpp rmw_zenohd` or the `zenoh` alias.
If you have problems you may need to first stop the ros2 daemon: `ros2 daemon stop`

### Connect to host
If you now want to connect to this host (e.g. to stream topics to RViz on another computer) you can run the following command on the other computer: `export ZENOH_CONFIG_OVERRIDE='mode="client";connect/endpoints=["tcp/THE_CAR_IP:7447"]'`. You will need to run this command in each terminal with which you want to connect. You can also add the ips of the cars to `remote_car.sh` and run `scripz && source remote_car.sh YOUR_CAR`.



### Parameters
There are two important parameters when running the base system or mapping:
 - racecar_version: This sets car related parameters such as TFs, lookup tables or Pacejka Parameters. By default this is written to the `.env` file in the installation progress.
 Everything that differs between cars lives in `stack_master/config/RACECAR_VERSION/`, e.g. the sensor mounting transforms in `static_transforms.yaml` (published by `static_transforms_launch.xml` for base system, mapping and simulator alike).
 - use_vio: Launches the RealSense D455 camera and Basalt VIO. Defaults to the `USE_VIO` value in the `.env` file (set during installation); only works if the container was built with `USE_VIO=true`.
 - map_name: This is the name of the map that you want to use / create. Every map is saved in `stack_master/maps/MAP_NAME/`. However, by default a map called `latest` is used.

### Mapping (on the real car)
Run the mapping launch file:
```shell
ros2 launch stack_master mapping_launch.xml map_name:=<map name of choice>
```

The default backend is GPU SynPF (`loc_algorithm:=synpf`). Choose the original
GPU PF with `loc_algorithm:=pf`, or Cartographer with `loc_algorithm:=cartographer`
(`slam` is an alias). The mapping and car launch scripts
accept launch arguments, e.g. `bash launch_scripts/launch_mapping.sh loc_algorithm:=pf`.
Both particle filters use the saved PNG/YAML map; their mapping modes use slam_toolbox.
Press `y` to save through the existing map exporter: identical PNG/YAML and
`pf_map` files with `initial_pose`, with no pbstream in PF mode.
[Dependencies and details](../state_estimation/PARTICLE_FILTER.md).

  - `<map name of choice>` can be any name with no white space. Conventionally we use the location name (eg, 'hangar', 'ETZ', 'icra') followed by the day of the month followed by an incremental version number. For instance, `hangar_12_v0`. You can also omit this argument. Then a map called `latest` is created. If there already is such a map it is copied to `backup`.
  - `<NUCX>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car). If you have exported a `.env` file during the build process this is read from the env variables and does not have to be set.

If you have a display connected (we recommend [this tool](https://github.com/ETH-PBL/remote-novnc)) an image will pop up showing you the current map. In the image you can press a button to save it. If you have no display you can do that via the terminal by pressing `y`, once you are satisfied by the map.



### Global Planner (on the real car)
After mapping we can run the global planner. Before you can check the saved png and edit it with your favorite image editing software. This allows you to remove mapping artefacts or add new virtual chicanes to the map.
Then you can run the global planner:
```shell
ros2 launch stack_master planner_launch.xml map_name:=<map name of choice>
```
 - `<map_name_of_choice` must be the name of any prerecorded map. If it is ommited the map called `latest` will be used.

A GUI will popup showing you the map and the extracted centerline. If you are satisfied you can close it to start the global raceline generation. If you have no monitor it starts automatically. Then two GUIs will be shown, and within them a slider can be used to select the velocity scaling and overtaking sectors. 
Be careful as once a sector is chosen it cannot be further subdivided. 

### Base System
```shell
ros2 launch stack_master base_system_launch.xml map_name:=<name of mapped track> sim:=<true/false> racecar_version:=<NUCX used>
```
  - `<name of mapped track>` is the name of the track you want to run on. It must belong to the list of maps available in the `stack_master/maps` folder or be omitted. Then the map called `latest` is used. 
  - `<true/false>` is a boolean value that indicates if you want to run the simulation or the real car. 
  - `<NUCX>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car). You can also omit this if the `.env` file was created.

### Time trials 
```shell
ros2 launch stack_master time_trials_launch.xml racecar_version:=<NUCx used> LU_table:=<Look-Up Table name> ctrl_algo:=<control algorithm> 
```
  - `<NUCx>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car). If you omit this the one set in base_system is used. 
  - `<Look-Up Table name>` is the name of the Look-Up Table you want to use. It must belong to the list of Look-Up Tables available in the `systm_identification/steering_lookup/cfg` folder.
  - `<control algorithm>` is the control algorithm you want to use. Current possibilities are MAP / PP.

### Follow the Gap

Build and source the controller and launch configuration after adding FTG:

```shell
cd ~/ws
colcon build --packages-select controller stack_master --symlink-install
source install/setup.bash
```

Start the base system (or the simulator) as above, then run the standalone
`controller.follow_the_gap` node instead of the time trials/head-to-head controller:

```shell
# Ego car (real car or simulator)
ros2 launch stack_master ftg_launch.xml racecar_version:=BLUE sim_opp:=false

# Simulated opponent (requires num_agent: 2 in config/SIM/sim.yaml)
ros2 launch stack_master ftg_launch.xml racecar_version:=BLUE sim_opp:=true
```

`sim_opp` selects which car to control; it does not start the simulator.
FTG uses `/scan` and `/drive` for the ego car, or `/opp_scan` and `/opp_drive`
under node namespace `/opp_racecar` for the opponent. It does not require
waypoints or the state machine. Run only one autonomous controller per car.
On the real car, the existing actuation manager still requires the autonomous
deadman button (button 5 for BLUE).

Edit `config/follow_the_gap_params.yaml` for all standalone FTG tuning parameters.
It includes LiDAR cropping/filtering, obstacle bubbles, gap selection, the
distance-to-speed curve, minimum driving speed, command acceleration/deceleration, steering margin,
and shutdown publications;
each entry documents its units and constraints. The configured `max_speed` is
3.5 m/s (direct execution without YAML retains the 1.0 m/s fallback).
The same file applies to both ego and opponent. Parameters are read at startup;
restart FTG after editing. The three launch arguments `max_speed`,
`max_deceleration`, and `minimum_gap_angle` override the file, and `param_file`
can select a different ROS parameter YAML:

```shell
ros2 launch stack_master ftg_launch.xml racecar_version:=BLUE sim_opp:=false \
  max_speed:=1.5 max_deceleration:=2.5 minimum_gap_angle:=0.15
```

Steering calibration is loaded from `config/<racecar_version>/vesc.yaml` before
the FTG parameter file; the `SIM` configuration uses the node's steering defaults.
The FTG YAML lists optional calibration overrides as comments so that it does
not replace the selected vehicle's calibration. Topics use ROS remappings.

Speed uses the filtered/clipped front beam before obstacle bubbles
(default index 540); set
`front_distance_index: -1` to locate the forward beam from `LaserScan` geometry.
The target is `min(max(min_speed, speed_distance_gain * max(front_distance - braking_stop_margin, 0)), max_speed)`.
The minimum driving target is 0.7 m/s, including distances at or below 0.30 m.
The linear curve takes over above approximately 1.08 m with the default gain
of 0.9. With the current 1.5 m distance clip, the target
reaches at most 1.08 m/s. Bubble masks only affect gap selection. Commands ramp
up at `max_acceleration` (default 1.0 m/s^2), starting with zero speed on the
first scan. Steering continues to update independently. The command deceleration
limiter can temporarily retain a higher speed as the car slows down, including
entry into the minimum-speed region. Both rate limits cap elapsed time at `max_command_dt`.

The historical `controller_manager` FTG implementation has its own configuration
in `controller/config/legacy_ftg.yaml`, loaded by `controller_launch.xml` through
`ftg_param_file`. It does not affect standalone `ftg_launch.xml`.

### Head to Head
```shell
ros2 launch stack_master head_to_head_launch.xml racecar_version:=<NUCx used> LU_table:=<Look-Up Table name> ctrl_algo:=<control algorithm> overtake_mode:=spliner
```
- `<NUCx>` depends on which car you are using. Parameters are available for NUC2, NUC5, NUC6, SIM (the latter represents a dummy car).
- `<Look-Up Table name>` is the name of the Look-Up Table you want to use. It must belong to the list of Look-Up Tables available in the `systm_identification/steering_lookup/cfg` folder.
- `<control algorithm>` is the control algorithm you want to use. Current possibilities are MAP / PP.
- `<overtake_mode>` is the mode you want to use for overtaking. `spliner` is the only current possibility.

### Simulation with Opponent
It is possible to add a steerable opponent to the simulation by changing the following parameter in [stack_master/config/SIM/sim.yaml](./config/SIM/sim.yaml):
```yaml
    # opponent parameters
    num_agent: 2

    # opp starting pose on map
    sx1: 2.0
    sy1: 0.5
    stheta1: 0.0
```
and then launching the the simulator normally, eg:
```shell
ros2 launch stack_master base_system_launch.xml map_name:=glc_ot_ez sim:=true
```

Then, using the `sim_opp` flag, time trials can be rerouted to control the opponent:
```shell
# launch the ego car first!
ros2 launch stack_master time_trials_launch.xml ctrl_algo:=PP

# then you can control the opponent
ros2 launch stack_master time_trials_launch.xml ctrl_algo:=PP sim_opp:=true
```

The opponent exposes the following topics:
```
# simulator (gym_bridge)
/opp_drive                         # AckermannDriveStamped, opponent drive commands
/opp_scan                          # LaserScan, opponent lidar (sees the ego car)
/opp_racecar/odom                  # Odometry, opponent odometry
/opp_racecar/pose                  # PoseStamped, opponent pose
/opp_racecar/opp_odom              # Odometry, ego odometry as seen by the opponent
/car_state/opp_odom                # Odometry, opponent odometry as seen by the ego
/goal_pose                         # PoseStamped, resets the opponent pose (2D Goal Pose in RViz)

# time trials stack with sim_opp:=true
/opp_racecar/frenet/odom           # Odometry, opponent frenet odometry
/opp_racecar/frenet/pose           # PoseStamped, opponent frenet pose
/opp_racecar/state                 # opponent state machine state
/opp_racecar/state_marker          # opponent state machine marker
/opp_racecar/local_waypoints       # opponent local waypoints
/opp_racecar/local_waypoints/markers
/opp_racecar/perception/obstacles  # opponent obstacle input
```
