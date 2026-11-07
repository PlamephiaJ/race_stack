# Map Editor 现场速查

**改完点 Generate trajectories；成功后点 Save map + current trajectory。**

## 报错：直接查这张表

在底部 **Planner log** 找关键词。`Global Planner failed` / `Planner did not produce` 是汇总提示，往前看具体错误。

| 看到什么 | 去哪里 | 怎么改 |
| --- | --- | --- |
| `No closed contours` / `No closed contours found` | 地图画布 | 按实测修复断口、挡路黑线/灰带，让白色道路闭环，保留内外边界。先修图。 |
| `More than two track bounds` | 地图画布 | 清除确认是噪声的小黑点、小内岛、假支路，修复边界断口，使内外边界清晰。 |
| `track might be too small ... safety distance` | 右侧 **Safety width / Shortest path width** | 先核对地图比例。有间隙余量才将对应宽度每次减 `0.05 m` 重试；不能小于“车宽 + 两侧所需间隙”。 |
| `constraints are inconsistent, no solution` | 先查宽度，再调 **Sampling / smoothing** | 按下面三步处理。降低最高速度不能解决几何无解。 |
| `window_length` / `Singular matrix` / `not positive definite` | 地图；**Sampling / smoothing** | 检查错误小环、尖角、比例；若改过采样和平滑，恢复下表默认值重试。仍失败则保留日志排查。 |
| `ggv has to cover ...` / `ax_max_machines has to cover ...` | **Vehicle → Maximum speed** | 降到两张动力学表都覆盖的范围；仓库默认表最多支持 `15 m/s`。 |
| `Missing dynamics CSV` / CSV 列数、数值错误 | **Dynamics files** 及对应 CSV | 文件放在 `stack_master/config/global_planner/veh_dyn_info/`，填相对文件名。GGV 三列：速度、纵向、横向上限；驱动表两列：速度、加速度上限。 |
| `Curvature limit is exceeded` | 急弯预览；**Sampling / smoothing** | 试下方平滑/步长调整，检查切弯；仍超限则核对真实转弯空间，不要随意放大车辆曲率限制。 |
| `ggv acceleration limit ... exceeded` / `machine limits ... exceeded` | **Vehicle → Maximum speed**；**Optimization / speed → Speed filter window** | 降低最高速度、将滤波设为 `0 / Disabled` 后重算；仍告警则核对动力学表。 |
| `Minimum distance to boundaries ...` | **Track Boundaries** 图层 | 告警阈值固定为 `1 m`，窄赛道常见。检查实际车体间隙；太贴边就增加对应安全宽度后重算。 |
| `ModuleNotFoundError` / `ImportError` / 缺配置 | 启动终端 | source ROS 和工作空间后重启；仍缺文件按 [README](README.md#安装与运行)补依赖/配置。 |
| `discarded outdated result` / 改图后轨迹消失 | 主窗口 | 停止编辑，重新生成。 |

**无解时按顺序试，每次只改一项：**

1. 核对地图比例、最窄处和对应安全宽度。主分支改 **Safety width**；日志进入 `for overtaking` / `shortest path` 后改 **Shortest path width**。
2. 若参考线锯齿明显：**Global Planner parameters… → Sampling / smoothing → Spline smoothing factor**，试 `1 → 2`，检查是否切弯。
3. 未改善则恢复 `1`，同页 **Optimization step** 试 `0.20 → 0.10 m`。仍失败就保留地图、参数、日志，检查真实转弯空间。

若先出现 `More than two track bounds`，主轨迹成功后又报 `No closed contours found`，**修内外边界**；当前最短路径阶段仍依赖它们。

## 换场地先核对这 5 项

| 项目 | 去哪里、怎么设 |
| --- | --- |
| 地图比例和坐标 | 地图 YAML 的 `resolution`、`origin`：保留正确勘测值；有错才修正，修改后重新打开。 |
| 行驶方向 | 工具 **Initial pose**：沿目标方向拖动；右侧 **Reverse travel direction** 通常不勾选。 |
| 两种安全宽度 | 右侧 **Safety width / Shortest path width**，默认均 `0.70 m`；按实测最窄处和所需间隙设置。 |
| 试跑速度 | **Global Planner parameters… → Vehicle → Maximum speed**：默认 `15 m/s`，按现场测试要求降低。 |
| 路面/轮胎变化 | 参数窗口 **Dynamics files → GGV CSV file**：选择对应标定表；降最高速度不能代替附着标定。 |

**安全宽度包含车宽。** 车宽 `0.30 m`、安全宽度 `0.70 m`，约每侧留 `0.20 m`。不要改小真实车宽来绕过错误。

## 其余通常保持默认

以下位置均在 **Global Planner parameters…**：

| 页签 | 参数 → 仓库默认值 |
| --- | --- |
| Sampling / smoothing | Preparation step → `0.05`；Optimization step → `0.20`；Output waypoint step → `0.10`（均为米） |
| Sampling / smoothing | B-spline order → `3`；Spline smoothing factor → `1` |
| Optimization / speed | Minimum IQP iterations → `5`；Allowed IQP curvature error → `0.1`；Dynamics model exponent → `1`；Speed filter window → `0 / Disabled` |
| Vehicle | 长 `0.535 m`、宽 `0.30 m`、质量 `3.518 kg`、Drag coefficient `0.0136`、Curvature limit `1.0 rad/m`；同车已标定时沿用，换车/载荷变化时核对。 |
| Dynamics files | `ggv.csv`、`ax_max_machines.csv`；车况和地面适用时沿用。 |

地图阈值正常时保留；不用为当前离线流程调 `rate`、`show_plots`、INI 的 `width_opt` 或 mintime 参数。
参数随地图保存，**不会写回项目原配置**。日志出现 `Planner complete:` 且当前轨迹预览正常后再保存。
