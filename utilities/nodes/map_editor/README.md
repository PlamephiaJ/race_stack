# Map Editor

独立的 ROS 2 Python package，使用 PySide6、NumPy 和 OpenCV 编辑占据栅格地图。
无需启动 SLAM、车辆、仿真器、RViz 或 ROS 节点。点击 **Generate trajectories**
时在独立进程中复用现有 `GlobalPlannerNode.plan()`，生成后直接叠加预览。

## 安装与运行

先按仓库安装说明准备 ROS 2 和 Global Planner 的依赖（包括
`global_racetrajectory_optimization`、`trajectory_planning_helpers`、`quadprog`、
`scikit-image` 和编译后的 `f110_msgs`）。GUI 依赖应安装到运行 ROS Python 的同一个环境。
ROS Humble / Ubuntu 22.04 可使用：

```bash
sudo apt install python3-numpy python3-opencv python3-yaml \
  libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
  libxcb-image0 libxcb-keysyms1 libxcb-render-util0
python3 -m pip install 'PySide6==6.11.2'

cd ~/ws
source /opt/ros/humble/setup.bash
# 已有 Race Stack 工作空间先 source 其环境；首次安装先构建仓库所需依赖。
source install/setup.bash
colcon build --symlink-install --packages-select map_editor global_planner
source install/setup.bash
ros2 run map_editor map_editor
```

请同时重建 `global_planner`，使编辑器能使用新增的可选后台接口。
Python 的数值计算依赖沿用现有 Planner 环境，无需为编辑器升级 NumPy/OpenCV。
PySide6 在这里通过 pip 提供；`package.xml` 不使用 Humble 中不可用的 Qt6 rosdep 键。

```bash
ros2 run map_editor map_editor --maps-dir ~/ws/src/race_stack/stack_master/maps
ros2 run map_editor map_editor --map /path/to/my_track/my_track.yaml
```

地图目录优先级为 `--maps-dir`、`RACE_STACK_MAPS_DIR`、自动发现的源码
`stack_master/maps`、安装目录 `share/stack_master/maps`。左侧也可重新选择目录。
安装目录不可写时，应选择可写目录。

## 编辑与生成循环

1. **New map** 设置像素宽高、分辨率和原点（含 yaw），同时显示实际米制尺寸。
   默认黑色背景，选择 **Free track (white)** 绘制封闭赛道；也可选白色背景再画障碍物。
2. 在地图库双击 YAML，或 **Open YAML** 加载现有地图。支持相对/绝对图片路径；
   也可读取 OpenCV 支持的 PGM 等图像，保存时统一输出 PNG。
3. 使用 Brush、Eraser、Line、Rectangle、Polygon 绘制。Paint 可选障碍、可通行和未知；
   Eraser 始终擦为白色可通行区域。矩形和多边形可选填充/轮廓。
4. **Remove component** 删除点击位置的整个连通障碍；自动清理按最大连通域面积
   删除小障碍，保留接触图像边缘的墙和未知像素。操作可撤销。
5. 可选 **Initial pose** 在赛道上沿行驶方向拖动，设置起点与朝向，写入 YAML 的
   `initial_pose: [x, y, yaw]`。不设置时沿用 Planner 的起点投影逻辑。
6. 设置两种 safety width（包含车宽的总安全宽度），按需勾选反向，点击
   **Global Planner parameters…** 可调整车辆、采样、优化和速度参数，再点击
   **Generate trajectories**。这是对当前地图与参数的快照规划，不要求先保存。
7. 勾选以下任意图层，在图内预览；日志显示优化进度、圈速、错误。生成期间界面仍可编辑。
8. 修改地图边界/障碍物，再生成，直到满意。**Save map + current trajectory** 导出当前地图
   和匹配的轨迹；**Save as new map** 创建 `maps/<name>/<name>.png` 与同名 YAML。

| 图层 | 颜色 | 来源 |
| --- | --- | --- |
| Centerline | 蓝色 | `centerline_waypoints` |
| Global Raceline | 红色 | `global_traj_wpnts_iqp` |
| Shortest Path | 橙色 | `global_traj_wpnts_sp` |
| Track Boundaries | 绿色 | `trackbounds_markers` |

滚轮缩放，中键拖动或 Pan 工具平移，`F` 适配窗口，`1` 恢复像素比例。
多边形单击添加顶点，`Enter`、右键或双击完成；`Esc` 取消当前绘制。
`Ctrl+Z` 撤销，`Ctrl+Shift+Z` 重做，`Ctrl+S` 保存。历史最多 100 步，并限制压缩历史内存。
工具宽度单位为像素，噪点面积单位为平方像素。

## Global Planner 参数

右侧 **Global Planner parameters…** 或 **Edit → Global Planner parameters…**
打开参数窗口。初始值从项目的 `global_planner_params.yaml` 与 `racecar_f110.ini`
读取，字段悬停可查看原始参数名。

| 分类 | 可编辑参数 |
| --- | --- |
| Planner | 两种安全宽度、反向行驶 |
| Vehicle | 最高速度、车长、车宽、质量、阻力系数、曲率限制 |
| Sampling / smoothing | 预处理步长、优化步长、输出轨迹点步长、B-spline 阶数和平滑因子 |
| Optimization / speed | IQP 最少迭代次数、允许曲率误差、动力学模型指数、速度滤波窗口 |
| Dynamics files | `veh_dyn_info` 下的 GGV 与纵向加速度上限 CSV 文件名 |

点击 **OK** 应用；**Cancel** 丢弃弹窗中的修改。**Load project defaults** 恢复编辑器启动时
读取的项目值，**Import profile… / Export profile…** 导入或导出独立 YAML 参数预设。
安全宽度不得小于车宽；滤波窗口必须为正奇数，`0 / Disabled` 表示关闭。
参数修改可用主窗口 Undo / Redo 撤销、重做。

应用的参数随地图保存在 YAML 的 `map_editor_planner` 扩展字段中，重新打开会恢复；
原有地图字段、Planner 和地图加载器的读取方式保持兼容。第一次规划成功时也会记录当次有效参数。
新建地图或打开没有参数预设的地图会使用项目默认值。
每次生成把原配置和动力学 CSV 复制到临时目录，覆盖其中选定的参数，再交给原优化器使用。
原项目配置文件不自动覆盖。若要将设置用于编辑器外的规划流程，需把相应数值同步到原配置。

修改地图或任何规划参数都会隐藏过期预览；生成期间继续改参数，旧快照结果会被丢弃。
输出日志包含当次参数，便于对照实验。
安全宽度直接传给现有优化器，当前 mincurv / shortest-path 调用中优先于 INI 的 `width_opt`；
因此窗口不重复显示无效的 `width_opt`。`rate`、`show_plots` 和仅用于 mintime 的参数不用于
当前离线流程，窗口中不提供这些字段。

## 保存与一致性

```text
stack_master/maps/custom_map/
├── custom_map.png
├── custom_map.yaml
└── global_waypoints.json    # 成功生成且与地图匹配时保存
```

YAML 保留 resolution、完整 origin、阈值、initial_pose 和额外字段。
`negate: 1` 的输入会反转图片并规范化为 `negate: 0`，保持占据含义。
支持 ROS trinary 地图；scale/raw 模式需先转换。未知区域在规划输入中按不可通行处理。
图像与轨迹使用现有 Planner 的栅格坐标约定：图片 `(u,v)` 对应局部坐标
`(u * resolution, (height - 1 - v) * resolution)`，再应用 YAML 原点平移和旋转。

地图或参数改变后过期轨迹会隐藏；撤销回原状态可恢复预览。规划期间地图或参数改变、地图切换，
返回的旧快照结果会被丢弃。未保存的地图和新生成轨迹在关闭/切换时会提示保存。
保存修改后的地图而没有重新生成时，目录中的旧 `global_waypoints.json` 会改名为
`global_waypoints.stale-<timestamp>.json`，避免误供给轨迹发布器。
取消或规划失败不改用户地图，保留日志和之前仍匹配的预览。

导出的 `global_waypoints.json` 由原有序列化函数生成，可交给现有轨迹发布器和仿真流程。
新建地图在源码目录保存后，如果消费者使用安装目录的数据副本，还需要重建
`stack_master` 或使用其既有数据同步方式。此工具不生成或修改 PBSTREAM；
真实车定位仍需要与场景对应的定位地图，原有 PBSTREAM 不会因 PNG 修改而同步更新。

## 规划约束与故障排查

### 绘图或打开参数窗口后 free(): invalid pointer

PySide6 6.12.0 在 CPython 3.10 / 3.11 上存在 `Signal.emit()` 引用计数缺陷
（[PYSIDE-3474](https://qt-project.atlassian.net/browse/PYSIDE-3474)）。
持续发送鼠标坐标等信号会错误减少 Python `True` 的引用计数，最终触发原生崩溃。
此问题在不使用 OpenCV、仅发送 Qt 信号的最小程序中也能复现。

编辑器限制安装依赖为 `PySide6>=6.4,<6.12`，并在启动时阻止上述已知问题组合。
已安装 6.12.0 的环境应关闭编辑器后切换到已验证的 6.11.2：

```bash
# 6.12 新增的独立组件，6.11 已在 Addons 中包含；先删除，避免遗留版本约束。
python3 -m pip uninstall -y PySide6-WebEngine PySide6-Pdf
python3 -m pip install --user --upgrade 'PySide6==6.11.2'
python3 -c 'import PySide6, shiboken6; print(PySide6.__version__, shiboken6.__version__)'
```

`PySide6`、`PySide6-Essentials`、`PySide6-Addons` 与 `shiboken6` 会由 pip 一起切换到匹配版本。
如果仍有原生崩溃，在出错终端运行 `PYTHONFAULTHANDLER=1 ros2 run map_editor map_editor`
并保留崩溃调用栈；编辑器主入口也会开启该诊断。

### Qt xcb 启动错误

如果出现 `Could not load the Qt platform plugin "xcb"`，并且路径包含
`cv2/qt/plugins`，是 pip 安装的 OpenCV 把自己的 Qt 插件路径写进了进程环境。
编辑器启动时会清理 OpenCV 注入的插件/字体路径，使用 PySide6 的插件。
其他显式设置的 Qt 路径和 `QT_QPA_PLATFORM`（如 wayland / offscreen）会保留。

若仍提示 `libxcb-cursor.so.0` 缺失，安装上述 X11 运行库，再启动编辑器：

```bash
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 \
  libxcb-image0 libxcb-keysyms1 libxcb-render-util0
```

该依赖与 Python 包不同，重新安装 PySide6 并不能补齐系统的共享库。
编辑器会在初始化窗口前检查 xcb 共享库，缺失时输出可读错误，避免 Qt 直接 abort。
运行桌面编辑器需要可访问的图形显示；`offscreen` 仅用于自动测试。
参考：[Qt X11 依赖](https://doc.qt.io/qt-6/linux-requirements.html)、
[OpenCV 插件路径设置](https://github.com/opencv/opencv-python/blob/4.x/scripts/__init__.py)。

### 地图与优化错误

Planner 需要一条封闭、足够宽、有内外边界的赛道。开口、孤立噪点、尖角或过大的
安全宽度可能导致骨架提取失败、优化无解。调整地图、设置起点方向或安全宽度后重试。
现有 `GLC_smile_small` 的 JSON 可直接预览；本环境中用默认参数重新生成该地图时，
原有优化器返回 `constraints are inconsistent, no solution`，编辑器会显示日志，
不会用占位或近似轨迹代替失败的结果。

若日志提示缺少 ROS 消息、优化器或参数文件，先安装既有 Planner 依赖并 source
工作空间。GUI 只编辑地图时不会导入 ROS 或优化器。

## 架构与测试

- `document.py`：NumPy 地图、元数据、坐标转换、压缩撤销历史、OpenCV 清理、PNG/YAML 持久化。
- `canvas.py`：Qt 绘图工具、缩放平移、轨迹叠加。
- `app.py`：地图库、操作面板、QProcess 后台任务、取消和快照版本校验。
- `planner_worker.py`：调用原有 `GlobalPlannerNode.plan()`，不创建 ROS 节点或发布车辆话题。
  在局部坐标规划，再把轨迹、朝向和 Marker 变换到 YAML 的世界坐标。
- `overlays.py`：读取现有轨迹 JSON。
- `parameters.py` / `parameter_dialog.py`：项目默认值读取、参数窗口、校验、预设和临时配置副本。

原有 Planner 仅增加 `RACE_STACK_PLANNER_HEADLESS=1` 选择 Agg 后端，
以及中间 CSV 目录与优化器适配器的可选接口。旧调用默认行为不变，
骨架、边界、优化、速度计算算法均复用原实现。每次编辑器规划使用独立临时目录，
结束后清理，不共享 `~/.ros/map_centerline*.csv`，不写入优化器安装目录。
第三方优化器子模块保持不变；后台适配器复用其函数字节码，仅在私有 globals 中
将用于导出路径的 `__file__` 指向临时目录，避免重复导入模块并重复注册速度求解器补丁。

在已构建并 source 的工作空间中：

```bash
cd ~/ws/src/race_stack
export PYTHONPATH="$PWD/utilities/nodes/map_editor:$PYTHONPATH"
QT_QPA_PLATFORM=offscreen python3 -m pytest utilities/nodes/map_editor/test -q
# 额外运行实际优化器的完整 GUI 集成测试（约几十秒）：
QT_QPA_PLATFORM=offscreen MAP_EDITOR_TEST_PLANNER=1 \
  python3 -m pytest utilities/nodes/map_editor/test -q
# 在可访问的真实桌面上做持续交互回归（X11）：
MAP_EDITOR_TEST_DESKTOP=1 \
  python3 -m pytest utilities/nodes/map_editor/test/test_stability.py -q
```

集成测试覆盖取消、编辑期间结果过期、反向生成、旋转地图的四层预览、保存重载，
并用原有 `read_global_waypoints` 验证 ROS 消息兼容性；还验证参数修改期间结果过期，
以及真实优化器的最高速度与输出步长确实采用界面设置。
稳定性测试在独立子进程中启用 `PYTHONMALLOC=debug`，覆盖 3 万次信号发送的布尔引用计数，
以及 210 次绘图、1680 次鼠标移动、撤销重做和 30 次参数弹窗；同时测试非 4 字节对齐的地图宽度。
Qt 后台进程参考：[QProcess 官方文档](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html)。
