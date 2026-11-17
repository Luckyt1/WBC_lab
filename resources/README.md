# 外部资源

此目录的内容被 Git 忽略（本文件除外）。机器人 USD/URDF/网格、上游 IK 源码、
完整 IK 轨迹库和 AMASS 数据在本机准备，通过 `.env` 引用。

配置和安装见根目录 README、`docs/installation.md`、`docs/ik-data.md`。
不要将 `tests/fixtures/ik_banks` 当作完整训练库。

GL-165 使用独立目录 `resources/gl165/`，包含 `model.json`、`source/`、`usd/` 和 IK 轨迹库。
`GL165_ASSET_DIR` 可覆盖资产目录；`GL165_IK_BANK` 可显式指定轨迹库。
没有指定轨迹库时，读取资产目录的 `active_ik_bank.json`，不存在时使用 `ik_bank/`。
转换命令和资源检查见 [GL-165 使用说明](../docs/gl165.md)。
