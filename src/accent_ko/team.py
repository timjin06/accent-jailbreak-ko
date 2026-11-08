"""Select team voices and prepare a scoped Korean clean-audio pilot."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

from .data import TRANSLATION
from .io import digest, file_hash, read_json, write_json
from .plan import save


OWNERS = {
    'heesu': ('natural-en-AU', 'synthetic-zh'),
    'sungjeh': ('natural-en-ZA', 'native-es'),
    'myeongjin': ('natural-en-KE', 'synthetic-ar'),
    'baseline': ('synthetic-ko',),
}
TTS_CONFIG = Path('configs/tts_team_v2.json')
PROBE_TEXT = '안녕하세요. 한국어 음성 합성 테스트입니다.'


def probe_voices(voices, config, key):
    from .audio import elevenlabs_pcm, elevenlabs_seed

    failures = 0
    for voice in voices:
        seed = elevenlabs_seed(config, {'voice': voice['id'], 'text': PROBE_TEXT})
        try:
            pcm = elevenlabs_pcm(config['tts'], voice, PROBE_TEXT, seed, key)
        except HTTPError as exc:
            failures += 1
            print(f"{voice['id']}: HTTP {exc.code}")
            if exc.code == 429:
                break
        else:
            print(f"{voice['id']}: OK, {len(pcm) // 2} samples at 24000 Hz")
    return failures


def main():
    parser = argparse.ArgumentParser(description='Team Korean TTS pilot')
    parser.add_argument('owner', type=lambda name: 'sungjeh' if name == 'seongje' else name,
                        choices=(*OWNERS, 'all'))
    parser.add_argument('action', choices=('check', 'probe', 'sample', 'cer', 'plan'))
    parser.add_argument('--speakers', type=int, choices=(1, 2))
    parser.add_argument('--voice-id', action='append', default=[], metavar='SLOT=ID')
    parser.add_argument('--listened', help='Record who verified every selected voice in Korean')
    parser.add_argument('--out', type=Path, help='New manifest directory for plan')
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / 'configs/voices_elevenlabs.json').is_file():
        parser.error('Run from the repository root')
    tts = read_json(root / TTS_CONFIG)
    speakers = args.speakers or (2 if args.owner in ('baseline', 'all') else 1)
    prefixes = sum(OWNERS.values(), ()) if args.owner == 'all' else OWNERS[args.owner]
    catalog = {voice['id']: voice for voice in read_json(root / 'configs/voices_elevenlabs.json')}
    selected = {}
    for prefix in prefixes:
        first = catalog[f'{prefix}-1']
        count = 2 if prefix in OWNERS['baseline'] else speakers
        for index in range(1, count + 1):
            slot = f'{prefix}-{index}'
            selected[slot] = dict(catalog.get(slot, {
                'id': slot, 'group': first['group'], 'source_locale': first['source_locale'],
                'speaker': index, 'voice_id': None, 'settings': {},
            }))
    overridden = set()
    for value in args.voice_id:
        slot, _, voice_id = value.partition('=')
        if slot not in selected or slot in overridden or not re.fullmatch(r'[A-Za-z0-9_-]+', voice_id):
            parser.error(f'Expected one valid SLOT=ID for a selected slot, got {value!r}')
        overridden.add(slot)
        selected[slot].update(voice_id=voice_id, accent_label=None, elevenlabs_name=None)
    voices = list(selected.values())
    for voice in voices:
        voice['settings'] = {}
    missing = [voice['id'] for voice in voices if not voice['voice_id']]
    ids = [voice['voice_id'] for voice in voices if voice['voice_id']]
    if len(ids) != len(set(ids)):
        parser.error('Each selected slot must use a distinct voice_id')
    print(json.dumps({
        'tts': tts, 'owner': args.owner, 'voices': voices, 'missing_voice_ids': missing,
        'sample_wavs': 5 * len(voices), 'comparison_wavs': 50 * len(voices),
    }, ensure_ascii=False, indent=2), flush=True)
    if args.action == 'check':
        return
    if missing:
        parser.error('Missing voice IDs. Supply --voice-id SLOT=ID: ' + ', '.join(missing))
    if args.action == 'probe':
        if args.out or args.listened:
            parser.error('--out and --listened are for plan')
        key = os.environ.get('ELEVENLABS_API_KEY')
        if not key:
            parser.error('Set ELEVENLABS_API_KEY before sending probe requests')
        config = read_json(root / 'configs/experiment.json')
        config['tts'] = tts
        raise SystemExit(int(bool(probe_voices(voices, config, key))))
    if args.action in ('sample', 'cer'):
        if args.out or args.listened:
            parser.error('--out and --listened are for plan')
        experiment = read_json(root / 'configs/experiment.json')
        fingerprint = digest([voices, tts, experiment['seed'], file_hash(root / TRANSLATION)])[:12]
        output = root / 'assets/elevenlabs' / f'team-{args.owner}-{fingerprint}'
        if args.action == 'cer':
            from .cer import score_samples
            result = root / 'results' / f'{output.name}-cer.json'
            print(json.dumps(score_samples(output, voices, result), ensure_ascii=False, indent=2))
            return
        with TemporaryDirectory() as temp:
            voice_file = Path(temp) / 'voices.json'
            write_json(voice_file, voices)
            subprocess.run([
                sys.executable, 'scripts/synthesize_elevenlabs.py',
                '--voices', str(voice_file), '--tts-config', str(TTS_CONFIG),
                '--limit', '5', '--out', str(output),
            ], check=True)
        return
    if not args.out or not args.listened or not args.listened.strip():
        parser.error('plan needs --out and --listened with a Korean listening review record')
    for voice in voices:
        voice['korean_supported'] = True
        voice['provenance'] = args.listened.strip()
    config = read_json(root / 'configs/experiment.json')
    config['tts'] = tts
    result = save(root, args.out, config, voices, 'main',
                  team_pilot={'owner': args.owner, 'prompt_count': 50})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
