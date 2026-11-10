SynPF algorithm source: https://github.com/ForzaETH/particle_filter
Pinned commit: 85ecc1865ac687ad8cf31afadbcb826acb23262c
Reviewed files: src/synPF.py, cfg/pf2_params.yaml, src/utils/utils.py, LICENSE.

The port in state_estimation/synpf.py and synpf_models.py adapts the TUM/AMCL,
experimental arc, boxed-beam selection, normalized sensor mixture and permissible
region penalty into the existing ROS 2 GPU PF transport. The original ROS 1 node,
catkin package and dynamic_reconfigure server are not installed.

Porting corrections: preserve signed rotations separately from dispersion;
integrate rather than discard small/pure-yaw motion; use a stable zero-yaw arc
limit; choose a bounded unique boxed-index set for arbitrary scan geometry;
include planar laser mounting yaw and xy; reject outside-map hypotheses without
clipping them onto free boundary cells. Retain the shared circular statistics,
finite-weight recovery, scan-time TF synchronization and REP105 map->odom TF.

Parameters follow upstream except GPU rmgpu/2 is mandatory, initialization is
explicit/global rather than assuming (0,0,0), and scan updates occur once per scan.
The Gaussian/exponential sensor table is vectorized, with upstream's sigma/lambda
conversion and column normalization. The default motion model remains tum.
No claim is made of reproducing the paper's timings or real-track accuracy.
