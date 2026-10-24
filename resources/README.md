# 本地部署资源

这些目录被 Git 忽略，但 `./deploy.sh pack --with-resources` 会收集已准备的五组输入：

- `elf3_bxi_31/`：当前训练机器人 USD、URDF、网格及控制契约。
- `bxi_example_py_elf3/`：与当前评估匹配的离线框架和 MuJoCo 模型。
- `elf3_ik_upstream/`：固定哈希的数值 IK 源码。
- `ik_bank_v3/`：本轮训练使用的完整 v3 轨迹库。
- `dataset/`：AMASS 保留评估数据。

默认按仓库相对位置读取，也可用 `.env` 中的 ELF3_* 路径覆盖。只包含源码的 Git clone 需另外准备这些资源；含资源包解压后可直接运行 `./deploy.sh setup`。

输入来源和哈希见 `provenance/current_training.json` 与部署包里的 `BUNDLE_MANIFEST.json`。外部输入仍适用其原条款，源码仓库的 LICENSE 不重新授权这些输入。
