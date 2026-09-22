"""ElevenLabs 다국어 음성으로 한국어 프롬프트를 합성. 원문 논문의 TTS가 아님.

TTSMaker는 비한국어 화자로 한국어 텍스트를 읽지 못한다(docs/experiment-design.md
「원문 TTS의 한국어 합성 불가」). 이 스크립트는 locale별 ElevenLabs 다국어 음성으로
같은 한국어 문항을 합성해 억양 축을 만든다. 결과는 assets/audio에 직접 쓰지 않고
`accent-ko import-clean`으로 등록한다.

  export ELEVENLABS_API_KEY=...
  python scripts/synthesize_elevenlabs.py --list-voices          # voice_id/억양 라벨 확인
  python scripts/synthesize_elevenlabs.py --limit 5              # 채워진 슬롯만 합성
"""
import argparse
import csv
import json
import os
import time
import urllib.request
import wave
from pathlib import Path

from accent_ko.io import file_hash, write_json

API = 'https://api.elevenlabs.io/v1'
RATE = 24000  # pcm_24000: 상위 요금제 없이 받을 수 있는 최고 PCM 레이트


def call(path, payload=None):
    key = os.environ.get('ELEVENLABS_API_KEY')
    if not key:
        raise ValueError('Set ELEVENLABS_API_KEY')
    headers = {'xi-api-key': key}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(f'{API}/{path}', data=data, headers=headers)
    # 유료 생성 요청은 자동 재시도하지 않는다.
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def write_wav(path, pcm):
    if not pcm or len(pcm) % 2:
        raise ValueError('TTS provider did not return 16-bit mono PCM')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.wav')
    with wave.open(str(tmp), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    tmp.replace(path)


def read_prompts(path, column, limit):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if column not in rows[0]:
        raise ValueError(f'No column {column} in {path}')
    return [(f'advbench-{i:04d}', r[column]) for i, r in enumerate(rows[:limit])]


def self_check(tmp):
    pcm = bytes(range(256)) * 4
    write_wav(tmp / 'x.wav', pcm)
    with wave.open(str(tmp / 'x.wav')) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        assert w.getnframes() == len(pcm) // 2
    prompts = read_prompts(Path('advbench_ko.csv'), 'goal_ko', 3)
    assert [pid for pid, _ in prompts] == ['advbench-0000', 'advbench-0001', 'advbench-0002']
    assert all(text.strip() for _, text in prompts)
    print('self-check ok')


p = argparse.ArgumentParser(description='ElevenLabs 다국어 음성으로 한국어 프롬프트 합성')
p.add_argument('--voices', type=Path, default=Path('configs/voices_elevenlabs.json'))
p.add_argument('--prompts', type=Path, default=Path('advbench_ko.csv'))
p.add_argument('--column', default='goal_ko', help='합성할 텍스트 열: goal_ko(한국어), goal(영어 원문)')
p.add_argument('--limit', type=int, default=5, help='앞쪽 문항 수. 억양용 400문항 선정 규칙이 아님')
p.add_argument('--model', default='eleven_v3')
p.add_argument('--out', type=Path, default=Path('assets/elevenlabs'))
p.add_argument('--list-voices', action='store_true', help='계정에서 쓸 수 있는 voice_id와 억양 라벨 출력')
p.add_argument('--self-check', action='store_true')
a = p.parse_args()
if a.limit < 1:
    p.error('--limit must be positive')

if a.self_check:
    self_check(a.out / 'self-check')
    raise SystemExit(0)

if a.list_voices:
    for v in json.loads(call('voices'))['voices']:
        labels = v.get('labels') or {}
        print(v['voice_id'], v['name'], labels.get('language', '?'), labels.get('accent', '?'), sep='\t')
    raise SystemExit(0)

slots = json.loads(a.voices.read_text(encoding='utf-8'))
unfilled = [s['id'] for s in slots if not s['voice_id']]
slots = [s for s in slots if s['voice_id']]
if not slots:
    p.error(f'{a.voices}의 모든 voice_id가 비어 있습니다. --list-voices로 확인해 채우세요')
if unfilled:
    print(f'voice_id 미지정 {len(unfilled)}개 슬롯 건너뜀: {", ".join(unfilled)}')

for slot in slots:
    for pid, text in read_prompts(a.prompts, a.column, a.limit):
        path = a.out / slot['id'] / f'{pid}.wav'
        if path.exists():
            continue
        payload = {'text': text, 'model_id': a.model, **slot.get('settings', {})}
        pcm = call(f"text-to-speech/{slot['voice_id']}?output_format=pcm_{RATE}", payload)
        write_wav(path, pcm)
        write_json(path.with_suffix('.json'), {
            'sha256': file_hash(path), 'sampling_rate': RATE, 'samples': len(pcm) // 2,
            'provenance': {'provider': 'elevenlabs', 'endpoint': f'{API}/text-to-speech',
                           'model_id': a.model, 'output_format': f'pcm_{RATE}', 'voice': slot,
                           'prompt_id': pid, 'column': a.column, 'text': text,
                           'note': '원문 논문 TTS 아님. 억양 유지 여부는 청취 확인 필요'}})
        print(path)
        time.sleep(1.05)
