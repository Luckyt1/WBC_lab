"""Build the pinned, offline-only native reference limiter; no pip changes."""
from pathlib import Path
import subprocess
import sysconfig
import torch

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'third_party/ruckig'
output=ROOT/'legged_lab/envs/elf3_tang/wbc'/('_arm_ruckig'+sysconfig.get_config_var('EXT_SUFFIX'))
files=sorted(p for p in (source/'src/ruckig').glob('*.cpp') if p.name not in ('cloud_client.cpp','python.cpp'))
argv=['c++','-O3','-DNDEBUG','-std=c++17','-shared','-fPIC','-fopenmp',
      '-I'+str(Path(torch.__file__).parent/'include'),'-I'+sysconfig.get_paths()['include'],
      '-I'+str(source/'include'),str(ROOT/'tools/native/arm_ruckig.cpp'),*[str(p) for p in files],'-o',str(output)]
subprocess.run(argv,check=True)
print(output)
