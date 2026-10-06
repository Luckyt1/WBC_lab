# 本地资源

该目录默认仅提交此说明。机器人 USD、URDF、网格及动作数据由环境变量指定，不随源码分发。

默认约定：

```text
resources/elf3_bxi_31/usd/elf3.usd
resources/elf3_bxi_31/control_parameters.json
resources/elf3_bxi_31/source/...
resources/dataset/g1/CMU/**/..._jpos.npy
```

可通过 `ELF3_ASSET_DIR` 和 `ELF3_AMASS_DIR` 引用任意外部目录。详见 [安装说明](../docs/installation.md)。
