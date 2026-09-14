"""Resolve specified HF refs to commits. This is NOT proof of the paper's exact checkpoint."""
import argparse
import json
import os
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, default=Path('configs/models.json'))
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    p.error('Output already exists; choose a new lock file')
models = json.loads(a.config.read_text(encoding='utf-8'))
headers = {'Authorization': 'Bearer ' + os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}
for name, spec in models.items():
    ref = spec['revision'] or 'main'
    req = urllib.request.Request(f"https://huggingface.co/api/models/{spec['repo']}/revision/{ref}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        info = json.load(response)
    spec['revision'] = info['sha']
    spec['revision_provenance'] = f'Resolved {ref}; paper checkpoint identity not established'
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(models, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(a.out)
