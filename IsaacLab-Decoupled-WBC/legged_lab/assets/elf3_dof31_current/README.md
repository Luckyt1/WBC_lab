# 从当前部署 URDF 重新生成的 ELF3 31 自由度资产

生成日期：2026-09-13。

主入口：[usd/elf3.usd](usd/elf3.usd)。使用或复制时请保留整个 `usd/` 目录，主入口引用同目录内的 configuration 文件。

源文件：`/home/cheng/bxi_controller_ros2/resources/elf3_dof31/urdf/elf3.urdf`。
源 URDF SHA-256：`6ede896d8718593927e96953cee0ce66c2c56eacd6bec50f86604962d53a0410`。

`source_elf3.urdf` 是源文本快照；重新转换需要原资源目录的 meshes，所有输入网格的哈希记录在 [conversion_manifest.json](conversion_manifest.json)。生成的 USD 网格已包含在本目录的 USD 文件中，不依赖原 STL 路径。

## 转换设置

- 使用当前安装的 Isaac Lab UrdfConverter / Isaac Sim URDF importer；强制重新转换。
- 显式启用 `import_inertia_tensor=True`，保留 URDF 中声明的质量、质心及惯量。
- 浮动基座、31 个可动关节（包括头部 yaw/pitch）、米制、Z 向上。
- `merge_fixed_joints=True`：合并四个没有质量/惯量/碰撞声明的传感器固定连杆到所属父体，生成 32 个有惯性参数的刚体；不删除或固定头部两个可动关节。
- 碰撞使用 URDF collision 网格和 convex hull；不从 visual 补碰撞。自碰撞开启，运行任务可另行覆盖。
- 关节 drive 为 force/position，力矩及速度上限取 URDF。
- USD 中 Kp/Kd 为 0，与 URDF 不含控制增益相对应。实际控制时需要提供执行器 PD 配置，不能把本资产当作已经配置好控制器的训练任务。

## 已通过的验证

通过独立 OpenUSD 读取生成文件并与源 URDF 按名称逐项比较：

- 31 个可动关节名称、父子关系、零位位置/旋转、转轴、硬角度限位一致。
- 31 个关节的力矩/速度上限一致。
- 32 个刚体的质量、质心、完整惯量张量一致（浮点容差内）；惯量已由 USD 主轴形式转换回连杆坐标系比较。
- URDF 显式质量之和约 **43.222476 kg**；USD 总质量在浮点误差内相同。
- 无缺失 USD 层或机器人网格依赖；使用 Isaac Sim 自带 `OmniPBR.mdl` 材质，纯 OpenUSD 环境未加载 MDL 搜索路径时会报告这一内建材质未解析。
- 源 URDF/网格在转换期间未改变；旧训练 USD 和旧执行器配置未修改。

完整结果与输出哈希：[verification.json](verification.json)。

`verification.json` 记录原资产转换时的静态 USD/URDF 数值与结构检查。本基础训练备份只保留训练使用的合并固定关节版本；历史未合并版本、转换工具和仿真日志已清理。

当前 `ELF3_CFG` 引用 `usd/elf3.usd`，训练不需要源 STL 文件。整个 `usd/` 目录必须随代码一起备份。`source_elf3.urdf` 用于校验质量和关节限位；重新转换模型时仍需另行准备原始网格。

训练入口与环境恢复步骤见 [备份使用说明](../../../../README.md)。
