"""Signal transforms and TTS asset provenance; no automatic voice replacement."""
import io
import os
import time
import urllib.request
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt, resample_poly
from math import gcd
from .io import digest, file_hash, read_json, write_json


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite, non-empty mono audio')
    peak = np.max(np.abs(x))
    if peak == 0:
        raise ValueError('Silent audio cannot be an experimental speech input')
    return x / peak


def load_audio(path, sr=None):
    x, rate = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    normalize(x)  # validate, without changing clean amplitude
    if sr and sr != rate:
        factor = gcd(rate, sr)
        x = resample_poly(x, sr // factor, rate // factor)
        rate = sr
    return x, rate


def transform(x, sr, condition, params, seed=2025, ir=None):
    normalize(x)
    if condition == 'clean':
        return x.copy()
    if condition == 'echo':
        delay, decay = params['delay'], params['decay']
        if delay < 0 or not np.isfinite(delay) or not 0 <= decay <= 1:
            raise ValueError('Invalid echo delay or decay')
        offset = int(sr * delay)
        y = np.zeros(len(x) + offset)
        y[:len(x)] += x
        y[offset:offset + len(x)] += decay * x
    elif condition.startswith('reverb_'):
        if ir is None:
            raise ValueError('Original impulse response file required')
        normalize(ir)
        y = fftconvolve(x, ir, mode='full')
    elif condition == 'whisper':
        if not 0 < params['cutoff'] < sr / 2:
            raise ValueError('Whisper cutoff must be below Nyquist frequency')
        # A.1 omits the two helper bodies. Causal Butterworth is an explicit reconstruction.
        filt = butter(params['order'], params['cutoff'], fs=sr, output='sos')
        y = sosfilt(filt, x * params['reduction'])
        y += np.random.default_rng(seed).normal(0, params['noise'], size=len(x))
    else:
        raise ValueError(f'Unknown audio condition: {condition}')
    return normalize(y)


def cached(path, provenance):
    path = Path(path)
    meta = path.with_suffix('.json')
    if not path.exists() and not meta.exists():
        return False
    if not path.exists() or not meta.exists():
        raise ValueError(f'Incomplete cached asset: {path}')
    saved = read_json(meta)
    if saved['provenance'] != provenance or saved['sha256'] != file_hash(path):
        raise ValueError(f'Asset provenance/hash mismatch: {path}')
    return True


def store_audio(path, x, sr, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.wav')
    sf.write(tmp, x, sr, subtype='PCM_16')
    tmp.replace(path)
    write_json(path.with_suffix('.json'), {'sha256': file_hash(path), 'sampling_rate': sr,
               'samples': len(x), 'provenance': provenance})


def elevenlabs_seed(config, item):
    parts = [config['seed'], item['text']]
    if config['tts'].get('seed_scope') != 'prompt':
        parts.insert(1, item['voice'])
    return int(digest(parts)[:8], 16)


def elevenlabs_payload(tts, voice, text, seed):
    return {'text': text, 'model_id': tts['model_id'],
            'voice_settings': {**tts['settings'], **voice['settings']}, 'seed': seed}


def elevenlabs_pcm(tts, voice, text, seed, key):
    if tts['output_format'] != 'pcm_24000':
        raise ValueError('ElevenLabs manifest synthesis requires pcm_24000')
    payload = elevenlabs_payload(tts, voice, text, seed)
    req = urllib.request.Request(
        f"{tts['endpoint']}/{voice['voice_id']}?output_format={tts['output_format']}",
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'xi-api-key': key})
    # Paid requests are not retried.
    with urllib.request.urlopen(req, timeout=120) as response:
        pcm = response.read()
    if not pcm or len(pcm) % 2:
        raise ValueError('ElevenLabs did not return non-empty 16-bit mono PCM')
    return pcm


def clean_location(root, item, config):
    provenance = {'text': item['text'], 'voice': item['voice_spec'], 'tts': config['tts']}
    if config['tts'].get('provider') == 'elevenlabs':
        provenance['seed'] = elevenlabs_seed(config, item)
    path = Path(root) / 'assets/audio' / ('clean-' + digest(provenance)[:24] + '.wav')
    return path, provenance


def make_clean(root, item, config):
    path, provenance = clean_location(root, item, config)
    if cached(path, provenance):
        return path
    voice = item['voice_spec']
    if voice['voice_id'] is None or not voice.get('provenance'):
        raise ValueError(f"Unresolved TTS voice: {voice['id']}")
    if voice.get('korean_supported') is not True:
        raise ValueError(f"Korean synthesis has not been verified for voice: {voice['id']}")
    tts = config['tts']
    provider = tts.get('provider', 'ttsmaker')
    if provider == 'elevenlabs':
        key = os.environ.get('ELEVENLABS_API_KEY')
        if not key:
            raise ValueError('Set ELEVENLABS_API_KEY')
        pcm = elevenlabs_pcm(tts, voice, item['text'], provenance['seed'], key)
        x = np.frombuffer(pcm, dtype='<i2').astype(np.float64) / 32768
        normalize(x)
        store_audio(path, x, 24000, provenance)
        time.sleep(1.05)
        return path
    if provider != 'ttsmaker':
        raise ValueError(f'Unsupported TTS provider: {provider}')
    token = os.environ.get('TTSMAKER_TOKEN')
    if not token:
        raise ValueError('Set TTSMAKER_TOKEN or import existing clean audio')
    settings = {**config['tts']['settings'], **voice['settings']}
    if set(settings) & {'text', 'token', 'voice_id'}:
        raise ValueError('TTS settings cannot override text, token or voice identity')
    payload = {**settings, 'token': token, 'text': item['text'], 'voice_id': voice['voice_id']}
    req = urllib.request.Request(config['tts']['endpoint'],
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    # Do not retry a paid creation request automatically.
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.load(response)
    if result.get('status') != 'success':
        raise ValueError('TTSMaker synthesis failed; inspect provider account (response omitted to protect token)')
    url = result.get('audio_file_url', '')
    if not url.startswith('https://'):
        raise ValueError('TTS provider did not return an HTTPS audio URL')
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    x, sr = sf.read(io.BytesIO(data), always_2d=True)
    x = x.mean(axis=1)
    normalize(x)
    store_audio(path, x, sr, provenance)
    time.sleep(1.05)
    return path


def import_clean(root, item, config, source):
    voice = item['voice_spec']
    if voice['voice_id'] is None or not voice.get('provenance') or voice.get('korean_supported') is not True:
        raise ValueError('Record the voice identity and verified Korean support before importing audio')
    path, provenance = clean_location(root, item, config)
    if config['tts'].get('provider') == 'elevenlabs':
        source_meta = Path(source).with_suffix('.json')
        if not source_meta.exists():
            raise ValueError('ElevenLabs import requires matching generation provenance')
        saved = read_json(source_meta)
        if saved.get('provenance') != provenance or saved.get('sha256') != file_hash(source):
            raise ValueError('ElevenLabs source provenance/hash mismatch')
    if cached(path, provenance):
        raise ValueError('Clean asset already exists; no overwriting')
    x, sr = load_audio(source)
    store_audio(path, x, sr, provenance)
    # Preserve origin separately; cached() still validates the generation identity.
    meta = read_json(path.with_suffix('.json'))
    meta['imported_from'] = {'path': str(Path(source).resolve()), 'sha256': file_hash(source)}
    write_json(path.with_suffix('.json'), meta)
    return path


def materialize(root, item, config):
    source = make_clean(root, item, config)
    x, sr = load_audio(source)
    params = item['params']
    ir = None
    ir_hash = None
    if item['condition'].startswith('reverb_'):
        ir_path = Path(root) / params['ir']
        ir, _ = load_audio(ir_path, sr)
        ir_hash = file_hash(ir_path)
    seed = int(digest([config['seed'], item['id']])[:8], 16)
    provenance = {'source_sha256': file_hash(source), 'condition': item['condition'],
                  'params': params, 'ir_sha256': ir_hash, 'seed': seed,
                  'implementation': 'accent-ko-v0.1-causal-butterworth'}
    dest = Path(root) / item['path']
    if not cached(dest, provenance):
        store_audio(dest, transform(x, sr, item['condition'], params, seed, ir), sr, provenance)
    return dest
