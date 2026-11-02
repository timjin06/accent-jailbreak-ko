import csv
import json
from pathlib import Path
import numpy as np
import pytest
from accent_ko.data import TRANSLATION, load_dataset, accent_subset
from accent_ko.io import write_json, write_jsonl, read_jsonl, read_json, unique_index
from accent_ko.plan import build, counts
from accent_ko.audio import transform, normalize, store_audio, cached
from accent_ko.evaluate import safety_label, word_errors, summarize, judge_responses
from accent_ko.models import messages
from accent_ko.run import execute

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    for filename, fields in [('advbench.csv', ['goal', 'target']),
                             (TRANSLATION, ['goal', 'target', 'goal_ko', 'target_ko'])]:
        with (tmp_path / filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(520):
                data = dict(goal=f'question {i}', target=f'target {i}', goal_ko=f'오늘의 질문 {i}', target_ko='답변')
                w.writerow({k: data[k] for k in fields})
    config = read_json(ROOT / 'configs/experiment.json')
    write_json(tmp_path / 'configs/accent_subset.json',
               {'provenance': 'TEST FIXTURE ONLY', 'prompt_ids': [f'advbench-{i:04d}' for i in range(400)]})
    (tmp_path / 'configs/defense_ko.txt').write_text('안전하게 답하세요.', encoding='utf-8')
    voices = read_json(ROOT / 'configs/voices.json')
    voices = [next(v for v in voices if v['group'] == g) for g in ('native', 'natural', 'synthetic')]
    return tmp_path, config, voices


def test_repository_data_integrity():
    rows, hashes = load_dataset(ROOT)
    assert len(rows) == 520
    assert set(hashes) == {'advbench.csv', TRANSLATION}
    assert all('target' not in r for r in rows)
    assert rows[0]['id'] == 'advbench-0000'


def test_main_echo_parameters(project):
    root, cfg, voices = project
    assert cfg['effects']['echo'] == {'delay': 0.3, 'decay': 0.6}
    _, audio, _ = build(root, cfg, voices, 'main')
    echoes = [a for a in audio if a['condition'] == 'echo']
    assert echoes
    assert all(a['params'] == {'delay': 0.3, 'decay': 0.6} for a in echoes)


def test_full_table4_counts():
    result = counts(read_json(ROOT / 'configs/voices.json'))
    assert result['total_audio'] == 102720
    assert result['native']['all_audio'] == 49920
    assert result['natural']['all_audio'] == 14400
    assert result['synthetic']['all_audio'] == 38400


def test_matrix_and_no_target_leak(project):
    root, cfg, voices = project
    jobs, audio, _ = build(root, cfg, voices, 'main')
    assert len(audio) == (520 + 400 + 400) * 6
    assert len(jobs) == (520 + len(audio)) * 5
    assert len(unique_index(jobs)) == len(jobs)
    assert {a['text'] for a in audio} == {f'오늘의 질문 {i}' for i in range(520)}
    assert all(j['language'] == 'ko' for j in jobs)
    assert sum(j['condition'] == 'text' for j in jobs) == 520 * 5


def test_subset_is_not_silently_sampled(project):
    root, cfg, voices = project
    write_json(root / cfg['accent_subset_path'], {'provenance': None, 'prompt_ids': []})
    with pytest.raises(ValueError, match='400'):
        build(root, cfg, voices, 'main')
    jobs, _, _ = build(root, cfg, voices, 'main', ['native'])
    assert jobs


def test_defense_and_ablation(project):
    root, cfg, voices = project
    voices = [dict(voices[0], source_locale='de')]
    jobs, audio, _ = build(root, cfg, voices, 'defense')
    assert len(jobs) == 520 * 4 * 2
    assert len(audio) == 520
    assert all(j['model'] != 'diva' for j in jobs)
    assert sum(bool(j['system']) for j in jobs) == len(jobs) // 2
    jobs, audio, _ = build(root, cfg, voices, 'ablation')
    assert len(audio) == 520 * 5
    assert {(a['params']['delay'], a['params']['decay']) for a in audio} == {
        (0.1, 0.6), (0.3, 0.6), (0.6, 0.6), (0.3, 0.1), (0.3, 0.9)}


def test_effects_have_physical_expected_outputs():
    x = np.array([1., 0., 0.])
    y = transform(x, 10, 'echo', {'delay': 0.2, 'decay': 0.5})
    np.testing.assert_allclose(y, [1, 0, .5, 0, 0])
    np.testing.assert_allclose(transform(x, 10, 'reverb_room', {}, ir=np.array([1., .5])), [1, .5, 0, 0])
    np.testing.assert_array_equal(transform(x, 10, 'clean', {}), x)
    with pytest.raises(ValueError, match='Silent'):
        normalize(np.zeros(10))
    with pytest.raises(ValueError):
        normalize(np.array([np.nan]))


def test_whisper_reproducible():
    p = read_json(ROOT / 'configs/experiment.json')['effects']['whisper']
    x = np.sin(np.arange(1600) / 10)
    a = transform(x, 16000, 'whisper', p, seed=3)
    b = transform(x, 16000, 'whisper', p, seed=3)
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(a)) == 1
    assert not np.array_equal(a, transform(x, 16000, 'whisper', p, seed=4))


def test_audio_provenance_detects_tampering(tmp_path):
    p = tmp_path / 'a.wav'
    store_audio(p, np.array([.1, .2]), 16000, {'voice': 'a'})
    assert cached(p, {'voice': 'a'})
    with pytest.raises(ValueError, match='mismatch'):
        cached(p, {'voice': 'b'})
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        cached(p, {'voice': 'a'})


def test_labels_and_korean_wer():
    assert safety_label('unsafe\nS1') == 'unsafe'
    assert safety_label('safe') == 'safe'
    assert safety_label('This might be unsafe') is None
    assert safety_label('') is None
    assert word_errors('오늘 날씨가 좋다', '오늘 정말 날씨가 좋다')['errors'] == 1
    assert word_errors('가', '가 나 다')['errors'] == 2  # WER may exceed 1.


def test_audio_messages_never_contain_transcript():
    for model in ('qwen2', 'meralion', 'minicpm'):
        chat = messages('SECRET TRANSCRIPT', np.ones(2), '방어 지침', model)
        assert 'SECRET TRANSCRIPT' not in str(chat)
        assert chat[0] == {'role': 'system', 'content': '방어 지침'}


def test_resume_and_evaluation_end_to_end(project):
    root, cfg, voices = project
    jobs, _, provenance = build(root, cfg, [], 'main')
    jobs = [j for j in jobs if j['model'] == 'qwen2'][:3]
    manifest = root / 'manifest'
    write_jsonl(manifest / 'jobs.jsonl', jobs)
    write_jsonl(manifest / 'audio.jsonl', [])
    write_json(manifest / 'provenance.json', provenance)
    class FakeModel:
        def generate(self, text, audio, system):
            assert audio is None
            return '테스트 응답'
    output, labels = root / 'responses.jsonl', root / 'labels.jsonl'
    execute(root, manifest, output, 'qwen2', {}, device='cpu', limit=1, backend=FakeModel())
    execute(root, manifest, output, 'qwen2', {}, device='cpu', backend=FakeModel())
    assert len(read_jsonl(output)) == 3
    execute(root, manifest, output, 'qwen2', {}, device='cpu', backend=FakeModel())
    assert len(read_jsonl(output)) == 3
    with pytest.raises(ValueError, match='different'):
        execute(root, manifest, output, 'qwen2', {'changed': True}, device='cpu', backend=FakeModel())
    class FakeJudge:
        def generate(self, chat):
            assert chat[1]['role'] == 'assistant'
            return 'unsafe\nS1'
    judge_responses(manifest, output, labels, {}, backend=FakeJudge())
    result = summarize(manifest, output, labels)
    assert result[0]['percent'] == 100
    rows = read_jsonl(labels)
    rows.pop()
    write_jsonl(labels, rows)
    result = summarize(manifest, output, labels)
    assert result[0]['percent'] is None
    assert result[0]['partial_percent'] == 100
    assert result[0]['unjudged_or_invalid'] == 1


def test_imported_audio_to_model_without_transcript(project):
    from accent_ko.audio import import_clean, materialize
    import soundfile as sf
    root, cfg, voices = project
    voices = [dict(voices[0], voice_id=123, korean_supported=True, provenance='test fixture')]
    jobs, audio, provenance = build(root, cfg, voices, 'main')
    a = next(a for a in audio if a['condition'] == 'echo')
    j = next(j for j in jobs if j['audio_id'] == a['id'] and j['model'] == 'qwen2')
    source = root / 'fixture.wav'
    sf.write(source, np.sin(np.arange(1600) / 10) * .5, 16000)
    import_clean(root, a, cfg, source)
    path = materialize(root, a, cfg)
    assert path.exists()
    assert materialize(root, a, cfg) == path
    manifest = root / 'manifest'
    write_jsonl(manifest / 'jobs.jsonl', [j])
    write_jsonl(manifest / 'audio.jsonl', [a])
    write_json(manifest / 'provenance.json', provenance)
    class FakeAudioModel:
        def generate(self, text, audio, system):
            assert text is None
            assert len(audio) == 1600 + 4800
            return '테스트 응답'
    output = root / 'results.jsonl'
    execute(root, manifest, output, 'qwen2', {}, device='cpu', backend=FakeAudioModel())
    assert read_jsonl(output)[0]['status'] == 'ok'
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        execute(root, manifest, output, 'qwen2', {}, device='cpu', backend=FakeAudioModel())


def test_sqa_requires_actual_dataset(project):
    root, cfg, voices = project
    with pytest.raises(FileNotFoundError):
        build(root, cfg, voices, 'sqa')
    write_jsonl(root / cfg['sqa_path'], [{'id': f'sqa-{i:04d}', 'text': f'일반 질문 {i}',
                                       'answer': '정답', 'source': 'test fixture'} for i in range(100)])
    jobs, audio, _ = build(root, cfg, voices, 'sqa')
    assert len(audio) == 100
    assert len(jobs) == 500
    assert all(j['reference_answer'] == '정답' for j in jobs)


def test_failed_generation_is_not_safe(project):
    root, cfg, _ = project
    jobs, audio, provenance = build(root, cfg, [], 'main')
    j = next(j for j in jobs if j['model'] == 'qwen2')
    manifest = root / 'manifest'
    write_jsonl(manifest / 'jobs.jsonl', [j])
    write_jsonl(manifest / 'audio.jsonl', [])
    write_json(manifest / 'provenance.json', provenance)
    class BrokenModel:
        def generate(self, *args):
            raise RuntimeError('test failure')
    output = root / 'results.jsonl'
    with pytest.raises(RuntimeError):
        execute(root, manifest, output, 'qwen2', {}, device='cpu', backend=BrokenModel())
    labels = root / 'labels.jsonl'
    write_jsonl(labels, [])
    result = summarize(manifest, output, labels)[0]
    assert result['errors'] == 1 and result['valid'] == 0
    assert result['percent'] is None


def test_human_scores_and_delta(tmp_path):
    from accent_ko.evaluate import human_scores, add_deltas
    labels = tmp_path / 'labels.jsonl'
    write_jsonl(labels, [{'id': str(i), 'label': 'unsafe' if i < 20 else 'safe'} for i in range(50)])
    annotations = tmp_path / 'human.csv'
    with annotations.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'human_label'])
        w.writeheader()
        for i in range(50):
            w.writerow({'id': str(i), 'human_label': 'unsafe' if i < 25 else 'safe'})
    result = human_scores(annotations, labels)
    assert result['fn'] == 5 and result['false_negative_rate'] == .2
    assert result['fn_percent_of_all'] == 10
    base = dict(suite='main', model='qwen2', voice='x', defense=False, complete=True)
    rows = [dict(base, condition='clean', params='[]', percent=10),
            dict(base, condition='echo', params='echo', percent=30)]
    assert add_deltas(rows)[1]['delta_pp'] == 20


def test_clean_text_delta_uses_same_prompt_ids(project):
    from accent_ko.evaluate import add_deltas
    root, cfg, voices = project
    jobs, _, _ = build(root, cfg, [voices[1]], 'main')
    jobs = [j for j in jobs if j['model'] == 'qwen2' and j['condition'] in ('text', 'clean')]
    manifest = root / 'paired'
    write_jsonl(manifest / 'jobs.jsonl', jobs)
    responses = [{'id': j['id'], 'experiment': j['experiment'], 'status': 'ok'} for j in jobs]
    labels = [{'id': j['id'], 'experiment': j['experiment'], 'status': 'ok',
               'label': 'unsafe' if j['condition'] == 'text' and int(j['prompt_id'][-4:]) >= 400 else 'safe'}
              for j in jobs]
    output, judged = root / 'paired-responses.jsonl', root / 'paired-labels.jsonl'
    write_jsonl(output, responses)
    write_jsonl(judged, labels)
    rows = add_deltas(summarize(manifest, output, judged))
    clean = next(r for r in rows if r['condition'] == 'clean')
    assert clean['delta_pp'] == 0
    assert clean['text_baseline']['planned'] == 400
    assert clean['text_baseline']['percent'] == 0
    # Missing text responses outside the matched subset do not block this comparison.
    write_jsonl(output, [r for r in responses if r['id'] != jobs[500]['id']])
    clean = next(r for r in add_deltas(summarize(manifest, output, judged)) if r['condition'] == 'clean')
    assert clean['delta_pp'] == 0
    # Missing a matched text response must block the difference, not shrink its denominator.
    write_jsonl(output, [r for r in responses if r['id'] != jobs[0]['id']])
    clean = next(r for r in add_deltas(summarize(manifest, output, judged)) if r['condition'] == 'clean')
    assert clean['text_baseline']['missing'] == 1
    assert clean['delta_pp'] is None


def test_elevenlabs_manifest_audio_and_cache(project, monkeypatch):
    import io
    from accent_ko.audio import materialize, import_clean
    root, cfg, voices = project
    cfg['tts'] = read_json(ROOT / 'configs/tts_elevenlabs.json')
    voice = dict(voices[0], voice_id='fixture-voice', korean_supported=True, provenance='TEST ONLY')
    jobs, audio, _ = build(root, cfg, [voice], 'main')
    item = next(a for a in audio if a['condition'] == 'echo')
    pcm = np.array([1000, -1000] * 120, dtype='<i2').tobytes()
    requests = []
    def request(req, timeout):
        requests.append(req)
        return io.BytesIO(pcm)
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'fixture-key')
    monkeypatch.setattr('accent_ko.audio.urllib.request.urlopen', request)
    monkeypatch.setattr('accent_ko.audio.time.sleep', lambda _: None)
    path = materialize(root, item, cfg)
    assert materialize(root, item, cfg) == path
    assert len(requests) == 1
    assert requests[0].full_url.endswith('/fixture-voice?output_format=pcm_24000')
    body = json.loads(requests[0].data)
    assert body['text'] == item['text']
    assert body['model_id'] == cfg['tts']['model_id']
    assert isinstance(body['seed'], int)
    metadata = read_json(path.with_suffix('.json'))
    assert metadata['samples'] == 240 + 7200
    assert 'fixture-key' not in path.with_suffix('.json').read_text()
    # Same prompt ID with changed translation or TTS settings gets a distinct asset path.
    changed_cfg = dict(cfg, tts=dict(cfg['tts'], model_id='different-model'))
    _, changed_audio, _ = build(root, changed_cfg, [voice], 'main')
    assert changed_audio[0]['path'] != audio[0]['path']
    translated = root / TRANSLATION
    translated.write_text(translated.read_text(encoding='utf-8-sig').replace('오늘의 질문 0,', '수정한 질문 0,'), encoding='utf-8-sig')
    _, changed_audio, _ = build(root, cfg, [voice], 'main')
    assert changed_audio[0]['path'] != audio[0]['path']
    # An unverified voice cannot make a paid request.
    unverified = dict(item, voice_spec=dict(voice, korean_supported=None))
    with pytest.raises(ValueError, match='Korean'):
        materialize(root, unverified, cfg)
    assert len(requests) == 1
    # Importing an old/mismatched file cannot silently claim the new generation identity.
    with pytest.raises(ValueError, match='provenance'):
        import_clean(root, changed_audio[0], cfg, path)


def test_cli_freezes_elevenlabs_settings(project, monkeypatch):
    import io
    from accent_ko.cli import main
    root, cfg, voices = project
    write_json(root / 'configs/experiment.json', cfg)
    write_json(root / 'configs/models.json', {})
    tts = read_json(ROOT / 'configs/tts_elevenlabs.json')
    write_json(root / 'configs/tts_elevenlabs.json', tts)
    voice = dict(voices[0], voice_id='cli-voice', korean_supported=True, provenance='TEST ONLY')
    write_json(root / 'configs/voices_elevenlabs.json', [voice])
    manifest = root / 'cli-manifest'
    monkeypatch.setattr('sys.argv', ['accent-ko', '--root', str(root),
        '--voice-config', 'configs/voices_elevenlabs.json', '--tts-config', 'configs/tts_elevenlabs.json',
        'plan', '--groups', 'native', '--out', str(manifest)])
    main()
    frozen = read_json(manifest / 'provenance.json')
    assert frozen['config']['tts'] == tts
    assert frozen['voices'] == [voice]
    # Audio reads the frozen manifest, even if current settings are edited afterwards.
    write_json(root / 'configs/tts_elevenlabs.json', dict(tts, model_id='changed-after-plan'))
    requests = []
    def request(req, timeout):
        requests.append(json.loads(req.data))
        return io.BytesIO(np.array([1000, -1000], dtype='<i2').tobytes())
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'fixture-key')
    monkeypatch.setattr('accent_ko.audio.urllib.request.urlopen', request)
    monkeypatch.setattr('accent_ko.audio.time.sleep', lambda _: None)
    monkeypatch.setattr('sys.argv', ['accent-ko', '--root', str(root),
        'audio', '--manifest', str(manifest), '--limit', '1'])
    main()
    assert len(requests) == 1
    assert requests[0]['model_id'] == tts['model_id']
