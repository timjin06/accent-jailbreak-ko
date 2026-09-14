"""Single-model resumable runs with immutable manifest/config identities."""
import importlib.metadata
import platform
import time
from pathlib import Path
from .io import read_jsonl, unique_index, read_json, write_json, append_jsonl, file_hash
from .audio import load_audio


def checkpoint(path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_suffix('.meta.json')
    if path.exists() or meta.exists():
        if not meta.exists() or read_json(meta)['identity'] != identity:
            raise ValueError('Result file belongs to a different manifest/configuration')
    else:
        versions = {}
        for package in ('numpy', 'scipy', 'torch', 'transformers', 'soundfile'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(meta, {'identity': identity, 'python': platform.python_version(), 'packages': versions})
    return unique_index(read_jsonl(path)) if path.exists() else {}


def execute(root, manifest, output, name, spec, device='cuda', limit=None, backend=None):
    from .models import AudioModel
    from .io import digest
    import random
    jobs = [j for j in read_jsonl(Path(manifest) / 'jobs.jsonl') if j['model'] == name]
    if not jobs:
        raise ValueError(f'No planned jobs for model {name}')
    assets = unique_index(read_jsonl(Path(manifest) / 'audio.jsonl'))
    provenance = read_json(Path(manifest) / 'provenance.json')
    identity = {'jobs_sha256': file_hash(Path(manifest) / 'jobs.jsonl'), 'model': name,
                'spec': spec, 'device': device, 'adapter_sha256': file_hash(Path(__file__).with_name('models.py'))}
    previous = checkpoint(output, identity)
    if set(previous) - {j['id'] for j in jobs}:
        raise ValueError('Unexpected IDs in result file')
    done = 0
    for job in jobs:
        audio_path = Path(root) / assets[job['audio_id']]['path'] if job['audio_id'] else None
        if job['id'] in previous:
            old = previous[job['id']]
            if old['status'] == 'ok' and audio_path and file_hash(audio_path) != old['audio_sha256']:
                raise ValueError(f"Audio changed since previous response: {job['id']}")
            # Errors remain explicit. Retry into a fresh output file after fixing the cause.
            continue
        if limit is not None and done >= limit:
            break
        result = {'id': job['id'], 'experiment': job['experiment'], 'model': name,
                  'audio_sha256': None}
        start = time.monotonic()
        try:
            audio = None
            if audio_path:
                meta = read_json(audio_path.with_suffix('.json'))
                if file_hash(audio_path) != meta['sha256']:
                    raise ValueError('Audio checksum mismatch')
                audio, _ = load_audio(audio_path, 16000)
                result['audio_sha256'] = meta['sha256']
            if backend is None:
                backend = AudioModel(name, spec, device)
            # Reset RNG per request, so resumed and uninterrupted runs use the same seed.
            seed = int(digest([provenance['config']['seed'], job['id']])[:8], 16)
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)
            if hasattr(backend, 'torch'):
                backend.torch.manual_seed(seed)
            response = backend.generate(job['text'] if audio is None else None, audio, job['system'])
            if not isinstance(response, str):
                raise ValueError('Model did not return text')
            result.update(status='ok', response=response, seed=seed)
        except Exception as exc:
            result.update(status='error', error=f'{type(exc).__name__}: {exc}')
            append_jsonl(output, {**result, 'seconds': time.monotonic() - start})
            # A broken backend or missing asset must not fill thousands of fake observations.
            raise
        result['seconds'] = time.monotonic() - start
        append_jsonl(output, result)
        done += 1
    return {'new_responses': done, 'previous_records': len(previous), 'planned': len(jobs)}
