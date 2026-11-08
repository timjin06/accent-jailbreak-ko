"""Score the team's five clean Korean utterances per voice."""
import unicodedata

from .evaluate import edit_distance
from .io import file_hash, read_json, write_json

ASR_MODEL = 'openai/whisper-large-v3'
ASR_REVISION = '06f233fe06e710322aca913c1bc4249a0d71fce1'


def normalize(text):
    return ''.join(char for char in unicodedata.normalize('NFC', text).casefold()
                   if unicodedata.category(char)[0] in 'LN')


def score_samples(folder, voices, output):
    from .audio import load_audio

    samples = []
    for voice in voices:
        for index in range(5):
            prompt = f'advbench-{index:04d}'
            audio = folder / voice['id'] / f'{prompt}.wav'
            sidecar = audio.with_suffix('.json')
            if not audio.is_file() or not sidecar.is_file():
                raise ValueError(f'Missing sample {audio}; run sample with the same owner and options')
            meta = read_json(sidecar)
            if meta['sha256'] != file_hash(audio):
                raise ValueError(f'Audio hash mismatch: {audio}')
            provenance = meta['provenance']
            if provenance['voice'] != voice or provenance['prompt_id'] != prompt:
                raise ValueError(f'Sample identity mismatch: {audio}')
            samples.append((voice['id'], prompt, audio, meta))

    import torch
    from transformers import pipeline

    gpu = torch.cuda.is_available()
    asr = pipeline('automatic-speech-recognition', model=ASR_MODEL, revision=ASR_REVISION,
                   torch_dtype=torch.float16 if gpu else torch.float32, device=0 if gpu else -1)
    rows = []
    for voice, prompt, audio, meta in samples:
        signal, rate = load_audio(audio, 16000)
        transcript = asr({'array': signal, 'sampling_rate': rate},
                         generate_kwargs={'language': 'ko', 'task': 'transcribe',
                                          'do_sample': False})['text']
        reference = meta['provenance']['text']
        normalized = normalize(reference)
        rows.append({'voice': voice, 'prompt_id': prompt, 'audio_sha256': meta['sha256'],
                     'reference': reference, 'transcript': transcript,
                     'errors': edit_distance(normalized, normalize(transcript)),
                     'reference_chars': len(normalized)})

    summary = []
    for voice in voices:
        group = [row for row in rows if row['voice'] == voice['id']]
        errors = sum(row['errors'] for row in group)
        chars = sum(row['reference_chars'] for row in group)
        summary.append({'voice': voice['id'], 'errors': errors, 'reference_chars': chars,
                        'cer_percent': 100 * errors / chars})
    write_json(output, {'asr_model': ASR_MODEL, 'asr_revision': ASR_REVISION,
                        'language': 'ko', 'normalization': 'NFC-casefold-letters-numbers',
                        'summary': summary, 'samples': rows})
    return {'output': str(output), 'summary': summary}
