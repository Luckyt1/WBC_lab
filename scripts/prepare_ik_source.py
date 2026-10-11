"""Copy only the pinned numerical IK inputs from a local upstream checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legged_lab.paths import ik_source_dir


def prepare(checkout, output):
    provenance = ROOT/'legged_lab/assets/elf3_ik_upstream/PROVENANCE.json'
    manifest = json.loads(provenance.read_text())
    sources = {}
    for name, digest in manifest['sha256'].items():
        candidates = [checkout/name, checkout/'assets'/name]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} must match upstream commit {manifest["commit"]}')
        sources[name] = source
    output.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        shutil.copy2(source, output/name)
    shutil.copy2(provenance, output/'PROVENANCE.json')
    print(f'Prepared verified IK files at {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ik_source_dir())
    args = parser.parse_args()
    prepare(args.checkout.expanduser().absolute(), args.output.expanduser().absolute())
