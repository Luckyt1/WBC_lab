# 安装与部署

## 基础环境

本项目按当前训练环境固定为 Linux x86_64、Python 3.11、CUDA PyTorch 2.7.0、Isaac Sim 5.0.0 和 Isaac Lab 2.2.0。原机器为 Ubuntu 22.04、RTX 4090。12,556 环境的在训任务使用约 19 GB 总 GPU 显存；其他显卡可先使用 `--num-envs 4096`。

准备 NVIDIA 驱动、Python 3.11 与 venv 开发头文件、C++17 / OpenMP 编译器。Ubuntu 的编译依赖可通过 `build-essential` 准备；Python 来自系统或 Conda 均可。先确认 `nvidia-smi` 正常。

Isaac Sim pip 安装要求 GLIBC 2.34 以上。PyTorch CUDA 12.8 与 Isaac Sim 5.0 的安装组合依据 [Isaac Lab 2.2 官方安装说明](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/pip_installation.html)。本项目提供自己的训练入口，只安装 Isaac Lab 核心，不安装上游 RSL-RL 覆盖仓库中的修改版。

## 新机器自动安装

先解压含资源部署包，进入 `WBC_lab`：

```bash
./deploy.sh setup --python python3.11
./deploy.sh check --verify-bundle
./deploy.sh smoke
```

`setup` 创建 `.venv-train`、`.venv-eval`、`.venv-ik` 三个独立环境，安装依赖，编译 `_arm_ruckig`，生成 `.env` 并执行资源与依赖检查。不会安装系统驱动或修改其他 Python 环境。

首次启动 Isaac Sim 时按 NVIDIA 提示接受软件许可；没有自动代替用户设置许可环境变量。无显示器可 headless 运行。首轮扩展/着色器初始化可能较慢，`smoke --timeout 900` 可延长启动验证时限。

如果 `.env` 已存在，普通 `setup` 会拒绝覆盖；重新部署使用 `--force-env`，旧配置保存在 `.env.backup`。更改机器后重新创建 Python 环境，不能直接复制旧机器的虚拟环境。

`requirements/train.txt`、`eval.txt`、`ik.txt` 分别固定主要依赖，训练额外使用 `train-constraints.txt` 固定 Torch / NumPy / Isaac 组合；它们不是所有传递依赖的完整锁文件。

## 复用已有环境

已有 Isaac 环境时可省去仿真器下载：

```bash
./deploy.sh setup --train-python /path/to/isaac/bin/python
```

有三个完整环境时：

```bash
./deploy.sh setup \
  --train-python /path/to/isaac/bin/python \
  --deployment-python /path/to/eval/bin/python \
  --ik-python /path/to/ik/bin/python
```

显式提供的环境只检查，不往其中安装包。启动时通过子进程的 `PYTHONPATH` 选择本仓库与 `rsl_rl/`，避免覆盖其他正在使用的 editable 安装。Ruckig 编译产物只写入本仓库。

训练和普通评估使用 NumPy 1.x；Pinocchio 4.1 使用 NumPy 2.x，因此保留独立 IK 环境。缺少依赖时 `check` 会报告具体解释器和错误，不会自动混装。

## 资源准备

含资源包已经准备以下目录，并保存逐文件校验值：

```text
resources/elf3_bxi_31/           控制契约、USD、源 URDF 和网格
resources/bxi_example_py_elf3/   对应离线框架、MuJoCo XML 和网格
resources/elf3_ik_upstream/      固定版本 IK Python / YAML / URDF
resources/ik_bank_v3/           manifest.json、train.npz、test.npz
resources/dataset/             本轮 AMASS 保留评估数据
```

默认无需设置资源绝对路径，代码根据仓库位置解析。需要复用仓库外资源时，在 `.env` 填写 [.env.example](../.env.example) 中的对应变量。`deploy.sh` 自动读取该文件；已经显式导出的同名环境变量优先。

`check` 验证控制参数和 URDF 哈希、IK 源码哈希、生产轨迹库版本与哈希，还会实际打开 MuJoCo 模型、构建 Pinocchio IK，并验证 CUDA、项目 RSL-RL 和 Ruckig 模块。`--resources-only` 可在未安装仿真器时只检查文件。

纯源码包或 Git clone 不带这些资源。可以从现有含资源包提取 `WBC_lab/resources/`，或者按自己的合法资源来源准备。IK 上游固定 commit 及从零生成方法见 [IK 数据说明](ik-data.md)。机器人 URDF 与控制契约要求一致，不能直接换成任意最新上游模型。

如需从匹配的 BXI 源文件转换资产，使用 Isaac 环境执行：

```bash
python tools/convert_elf3_bxi.py --source /path/to/bxi_controller_ros2 \
  --output resources/elf3_bxi_31 --headless
python tools/verify_elf3_bxi_usd.py --asset resources/elf3_bxi_31 --headless
```

原始 URDF SHA-256 为 `6ede896d8718593927e96953cee0ce66c2c56eacd6bec50f86604962d53a0410`。默认每 5,000 轮评估包含 AMASS 保留测试，所以完整训练监督程序也检查其数据；IK 训练本身不读取 AMASS。

## 启动与排查

```bash
./deploy.sh train --dry-run
./deploy.sh train --num-envs 4096 --output logs/new_run --background
./deploy.sh status --output logs/new_run
tail -f logs/new_run/train.log
```

每次启动使用新输出目录。`--max-iterations` 表示本次新增 PPO 更新数，`--num-envs` 只调整并行数量。选择 GPU 可在命令前设置 `CUDA_VISIBLE_DEVICES=1`；进程内部始终使用可见的 `cuda:0`。

后台任务在启动后立即返回监督程序 PID，训练初始化是否成功以 `state.json` 和 `train.log` 为准。前台任务可用 Ctrl+C 停止；后台用记录的监督程序 PID 发送 SIGTERM，它会清理自己创建的训练和评估子进程。

`smoke` 执行 16 环境 / 2 更新，不执行耗时的完整策略评估。首次启动失败先查输出目录的 `train.log`。默认 300 秒超时，可用 `--timeout` 调整。显存不足时先结束其他 GPU 工作，或在空闲 GPU 上运行。

## 打包迁移

```bash
./deploy.sh pack --with-resources
./deploy.sh pack
```

输出在 `dist/`，同时生成 `.sha256` 文件。归档内所有路径都位于 `WBC_lab/`，并包含 `BUNDLE_MANIFEST.json`；资源采用真实文件，不依赖旧机器软链接。包中排除 `.env`、虚拟环境、模型、日志及编译产物。包解压后用 `check --verify-bundle` 核对内容，再在目标机器执行安装与编译。
