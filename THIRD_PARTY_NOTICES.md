# Third-party notices

This source distribution preserves upstream copyright headers and license notices. The root BSD-3-Clause license covers the project source subject to the component-specific notices below; it does not relicense external assets, datasets, SDKs or model weights.

| Component | Origin / notice |
| --- | --- |
| `rsl_rl/` | Modified [RSL-RL](https://github.com/leggedrobotics/rsl_rl). See `rsl_rl/LICENSE` and `rsl_rl/licenses/dependencies/`. |
| `legged_lab/` framework and inherited configuration | File headers credit RSL-RL, Isaac Lab, Legged Lab and TienKung-Lab under BSD-3-Clause. Headers are retained. |
| `legged_lab/envs/elf3_tang/wbc/` | Adapted from [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC). See `licenses/IsaacLab-Decoupled-WBC.txt`. The AMASS loader also retains its Intel Corporation 2024 BSD-3-Clause notice. |
| `tests/fixtures/ppo_before_auxiliary.py` | Historical PPO implementation retained solely as a regression reference, with its upstream notices. |
| Frozen control/geometry JSON | Configuration derived for ELF3 from BXI robot descriptions. The external robot URDF, meshes, USD and framework are not bundled. |

External prerequisites include [Isaac Lab](https://github.com/isaac-sim/IsaacLab), NVIDIA Isaac Sim, [BXI controller sources](https://github.com/bxirobotics/bxi_controller_ros2), and separately obtained retargeted AMASS data. Their respective terms apply when installing or redistributing them. A publicly accessible repository or downloadable dataset does not itself establish permission to redistribute its assets.

No pretrained policy, AMASS clip, NVIDIA simulator binary, or external BXI runtime is included in this distribution. Repository-local provenance hashes in `SOURCE_MANIFEST.json` identify the input source snapshot before packaging edits, not third-party ownership or licensing grants.
