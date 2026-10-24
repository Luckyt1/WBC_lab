"""Package source, or source plus local deployment inputs, with SHA-256 checksums."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_env import ROOT

DIRECTORIES = {'configs', 'deployment', 'docs', 'legged_lab', 'licenses', 'provenance',
               'requirements', 'rsl_rl', 'scripts', 'tests', 'third_party', 'tools', '.github'}
FILES = {'deploy.sh', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml',
         '.env.example', '.gitignore', '.gitattributes', 'SOURCE_MANIFEST.json', 'EXPORT_MANIFEST.json'}
EXCLUDED = {'__pycache__', '.git', '.pytest_cache', '.venv', 'logs', 'outputs', 'output', 'checkpoints'}
RESOURCE_DIRS = {'elf3_bxi_31', 'elf3_ik_upstream', 'ik_bank_v3', 'dataset', 'bxi_example_py_elf3'}


def include(relative, with_resources):
    parts = relative.parts
    if any(p in EXCLUDED or p.startswith('.venv') or p.endswith('.egg-info') for p in parts):
        return False
    if any(p == '.env' or (p.startswith('.env.') and p != '.env.example') for p in parts):
        return False
    if relative.suffix in {'.pyc', '.so', '.o', '.pt', '.pth', '.onnx', '.rknn', '.pem', '.key'}:
        return False
    if parts[0] == 'resources':
        return relative.as_posix() == 'resources/README.md' or (
            with_resources and len(parts) > 1 and parts[1] in RESOURCE_DIRS)
    return (len(parts) == 1 and parts[0] in FILES) or parts[0] in DIRECTORIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--with-resources', action='store_true', help='Include prepared robot/IK/AMASS/BXI inputs for local transfer')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    label = 'resources' if a.with_resources else 'source'
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = (a.output or ROOT/f'dist/WBC_lab_{label}_{stamp}.tar.gz').absolute()
    if output.exists():
        p.error(f'Output already exists: {output}')
    if a.with_resources:
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts.check_resources import check
        errors = check(deployment=True, amass=True)
        if errors:
            p.error('\n'.join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with tarfile.open(output, 'w:gz', compresslevel=3) as archive:
        candidates = [ROOT/f for f in FILES if (ROOT/f).is_file()]
        for folder in sorted(DIRECTORIES | {'resources'}):
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and include(path.relative_to(ROOT), a.with_resources):
                    candidates.append(path)
        for path in sorted(candidates):
            relative = path.relative_to(ROOT)
            if not include(relative, a.with_resources):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f'Release input must be an independent local file: {relative}')
            hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname='WBC_lab/'+relative.as_posix(), recursive=False)
        data = json.dumps({'with_resources': a.with_resources, 'sha256': hashes}, indent=2).encode()+b'\n'
        info = tarfile.TarInfo('WBC_lab/BUNDLE_MANIFEST.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    output.with_name(output.name+'.sha256').write_text(f'{digest.hexdigest()}  {output.name}\n')
    print(f'{output}\n{len(hashes)} files; {output.stat().st_size/1024/1024:.1f} MiB; SHA-256 {digest.hexdigest()}')


if __name__ == '__main__':
    main()
