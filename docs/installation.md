# 安装与外部资源

本项目按源码仓库使用。下面命令均从仓库根目录运行，不要将发布副本安装到仍在运行训练的环境中，以免 editable 安装切换包路径。

## 1. 训练环境

整理时使用的环境：Ubuntu 22.04、Python 3.11、PyTorch 2.7.0 + CUDA 12.8、Isaac Sim 5.0.0、Isaac Lab 2.2.0。其他主要版本记录在 [tested-environment.txt](../requirements/tested-environment.txt)，该文件是参考记录，不是完整依赖锁文件。

通过 [Isaac Lab 官方项目](https://github.com/isaac-sim/IsaacLab/tree/v2.2.0) 对应版本的安装说明准备 Isaac Sim、Isaac Lab 及其扩展；确认当前 Python 能 `import isaaclab`。本仓库不随附 NVIDIA SDK、CUDA 或 Isaac Lab。

```bash
python -m pip install -e ./rsl_rl --no-deps
python -m pip install -e '.[dev,viewer]'
```

使用仓库内的 RSL-RL 修改版，不要用 PyPI 同名包替换。`--no-deps` 用于保留已配置好的 Isaac/PyTorch 组合；Isaac 环境应已有匹配的 torch、torchvision，ONNX 可另外安装。

## 2. CPU 评估环境

另建 Python 3.11 环境，安装匹配的 CPU PyTorch，再安装本项目：

```bash
python3.11 -m venv .venv-eval
.venv-eval/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
.venv-eval/bin/python -m pip install -e '.[deployment,dev]'
```

在 `.env` 中将 `ELF3_DEPLOYMENT_PYTHON` 设为 `.venv-eval/bin/python` 的**绝对路径**。该环境用于 ONNX 导出、辅助头诊断与 MuJoCo 仿真；不需要启动 ROS 节点。真实离线适配器还依赖下一节的 BXI Python 框架。

## 3. 机器人与 BXI 框架

外部上游：[bxirobotics/bxi_controller_ros2](https://github.com/bxirobotics/bxi_controller_ros2)。本代码使用 ELF3 31 关节模型，其中腰腿 15 个、双臂 14 个、头部 2 个。需要的关节参数版本已固定在 `legged_lab/assets/elf3_bxi_31/control_parameters.json`。

两种资产准备方式：

1. 已有匹配的转换资产时，`ELF3_ASSET_DIR` 指向包含 `usd/`、`source/`、`control_parameters.json` 的目录。
2. 从匹配的 BXI 源码转换：在 Isaac 环境中执行以下命令，输出目录必须是新的目录。

```bash
python tools/convert_elf3_bxi.py \
  --source /path/to/bxi_controller_ros2 \
  --output resources/elf3_bxi_31 --headless
python tools/verify_elf3_bxi_usd.py --asset resources/elf3_bxi_31 --headless
```

默认采用部署力矩限制，并保留输入源文件快照。验证工具检查 USD 质量、惯量、关节轴、限位、驱动参数和依赖。旧 29 关节 USD 的对照检查可选，用 `--legacy-usd /path/to/old.usd` 指定。

`ELF3_BXI_FRAMEWORK_DIR` 指向 BXI checkout 下的 `src/bxi_example_py_elf3`；其内部需存在 `bxi_example_py_elf3/framework/`，并提供适配器使用的 `JointPolicy`、`ModelSpec`、`PolicyJointContract` 等接口。`ELF3_MUJOCO_XML` 可覆盖默认的 `data/mujoco_simulation/elf3.xml`。

**版本限制：**发布包未包含该外部框架和机器人模型，也未锁定一个已验证的公开 BXI commit。上游最新版本不一定具有相同框架接口；转换脚本还要求对应的 `policies/joints.py`、`bxi_example_demo.py` 与 `config/elf3_state_machine.yaml`。需要准备与本项目契约匹配的版本，不能假定任意新 checkout 都能直接复现。控制参数和 URDF 不匹配时，资源检查会明确失败。

参考 URDF SHA-256：`6ede896d8718593927e96953cee0ce66c2c56eacd6bec50f86604962d53a0410`。碰撞几何 JSON 和关节约定绑定这一版本；更换机器人模型时需重新核对。

## 4. 上肢动作数据

需要已重定向到 G1 关节顺序的 AMASS 数据，不是原始 AMASS 文件。每个 `*_jpos.npy` 为 `(T,36)`：根位置 3 维、根四元数 4 维、29 个关节位置。加载器从关节块中取出 14 个手臂关节，经 ELF3 映射与保护处理后作为参考。

`ELF3_AMASS_DIR` 可以指向以下任一目录的上层：

```text
<ELF3_AMASS_DIR>/g1/CMU/**/..._jpos.npy
<ELF3_AMASS_DIR>/dataset/g1/CMU/**/..._jpos.npy
```

按 AMASS 及所用重定向数据发布方的条款取得数据。本仓库不附带数据或自动下载脚本；相关加载代码来源见第三方说明。

## 5. 本机配置与预检

```bash
cp .env.example .env
# 编辑 .env，填写本机路径
set -a
source .env
set +a
python scripts/check_resources.py --deployment
python scripts/train_current.py --output logs/my_run --dry-run
```

预检不初始化 GPU，只检查文件、控制契约与数据目录。它不能替代实际的 Isaac、BXI API 或 MuJoCo 运行验证。资源齐备后，再按 README 启动训练。

`resources/` 和 `.venv-eval/` 都被 Git 忽略；也可将资源放在仓库外，只通过环境变量引用。
