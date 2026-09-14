"""Manifest construction. Missing identities are never silently substituted."""
from pathlib import Path
from .data import load_dataset, accent_subset, sqa_dataset
from .io import digest, read_json, write_json, write_jsonl

MODELS = ('qwen2', 'diva', 'meralion', 'minicpm', 'ultravox')
CONDITIONS = ('clean', 'reverb_teisco', 'reverb_room', 'reverb_railway', 'echo', 'whisper')


def counts(voices):
    out = {}
    for group, n in [('native', 520), ('natural', 400), ('synthetic', 400)]:
        speakers = sum(v['group'] == group for v in voices)
        out[group] = {'voice_slots': speakers, 'prompts_per_voice': n,
                      'clean_audio': speakers * n, 'all_audio': speakers * n * 6}
    out['total_audio'] = sum(out[g]['all_audio'] for g in ('native', 'natural', 'synthetic'))
    return out


def build(root, config, voices, suite, groups=None):
    root = Path(root)
    rows, hashes = load_dataset(root)
    selected = [v for v in voices if groups is None or v['group'] in groups]
    if suite == 'sqa':
        selected = [v for v in selected if v['group'] == 'native']
        rows = sqa_dataset(root / config['sqa_path'])
    elif suite in ('defense', 'ablation'):
        locales = ('de', 'it') if suite == 'defense' else ('de',)
        selected = [v for v in selected if v['group'] == 'native' and v['source_locale'] in locales]
        if not selected:
            raise ValueError(f'No original native voice slots selected for {suite}: {locales}')
    subset = None
    if suite == 'main' and any(v['group'] != 'native' for v in selected):
        subset = accent_subset(root / config['accent_subset_path'], rows)
    if suite == 'defense':
        defense = (root / config['defense_path']).read_text(encoding='utf-8').strip()
        if not defense:
            raise ValueError('Missing Korean defense prompt')
    else:
        defense = None
    provenance = {'dataset_sha256': hashes, 'config': config, 'voices': selected,
                  'suite': suite, 'groups': groups, 'defense': defense,
                  'subset_ids': [r['id'] for r in subset] if subset else None,
                  'sqa': rows if suite == 'sqa' else None}
    fingerprint = digest(provenance)
    jobs, audio = [], {}
    def add(r, v, condition, params, models, system=None):
        voice = v['id'] if v else None
        key = {'prompt_id': r['id'], 'voice': voice, 'condition': condition, 'params': params}
        aid = digest(key)[:24] if v else None
        if aid:
            audio[aid] = {'id': aid, **key, 'text': r['text'], 'voice_spec': v,
                          'path': f'assets/audio/{aid}.wav'}
        for m in models:
            job = {**key, 'model': m, 'suite': suite, 'group': v['group'] if v else 'text',
                   'source_locale': v['source_locale'] if v else None, 'language': 'ko',
                   'audio_id': aid, 'text': r['text'], 'system': system,
                   'reference_answer': r.get('answer'), 'experiment': fingerprint}
            job['id'] = digest(job)
            jobs.append(job)
    if suite == 'main':
        for r in rows:
            add(r, None, 'text', {}, MODELS)
    for v in selected:
        prompts = subset if v['group'] != 'native' and suite == 'main' else rows
        for r in prompts:
            if suite in ('main', 'sqa'):
                for c in CONDITIONS if suite == 'main' else ('clean',):
                    add(r, v, c, config['effects'].get(c, {}), MODELS)
            elif suite == 'defense':
                for system in (None, defense):
                    add(r, v, 'reverb_teisco', config['effects']['reverb_teisco'],
                        [m for m in MODELS if m != 'diva'], system)
            elif suite == 'ablation':
                # One shared baseline (0.3 s, 0.6), not two independent baseline runs.
                for delay, decay in [(0.1, 0.6), (0.3, 0.6), (0.6, 0.6), (0.3, 0.1), (0.3, 0.9)]:
                    add(r, v, 'echo', {'delay': delay, 'decay': decay}, MODELS)
            else:
                raise ValueError(f'Unknown suite: {suite}')
    return jobs, list(audio.values()), provenance


def save(root, out, config, voices, suite, groups=None):
    jobs, audio, provenance = build(root, config, voices, suite, groups)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Manifest output directory must be empty to protect run identity')
    write_json(out / 'provenance.json', provenance)
    write_jsonl(out / 'jobs.jsonl', jobs)
    write_jsonl(out / 'audio.jsonl', audio)
    return {'jobs': len(jobs), 'audio': len(audio), 'directory': str(out)}
