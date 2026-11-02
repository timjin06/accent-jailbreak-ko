# 실행 안내

모든 명령은 저장소 루트에서 실행합니다. 산출물 경로는 현재 디렉터리 기준이고, `--root`는 데이터/설정/음성의 기준 폴더입니다. GPU가 없는 컴퓨터에서는 검증과 계획, 음성 변형까지 수행할 수 있습니다. 5개 모델 어댑터의 실제 GPU 추론은 아직 검증되지 않았습니다.

## 1. 데이터와 조건 확인

```bash
accent-ko validate
accent-ko matrix
```

- `configs/voices.json`: 각 원래 화자 슬롯의 `voice_id`, `settings`, 출처 `provenance`, 한국어 합성 확인 `korean_supported`를 채웁니다. 지원하지 않으면 false로 남기고 화자를 교체하지 않습니다. 현재 기본값 null은 누락된 정보입니다.
- `configs/accent_subset.json`: `prompt_ids`에 원문 억양 평가 400개 ID, `provenance`에 선정 목록 출처를 기록합니다. ID는 0부터 시작하는 CSV 행 번호입니다. 원문 목록을 못 구하고 새로 선정하면 재현 차이로 명시해야 합니다.
- `assets/ir/{teisco,room,railway}.wav`: 원본 IR을 확보해서 넣습니다. 각 파일의 SHA-256이 생성 음성에 기록됩니다.
- `data/sqa_ko.jsonl`: 원문 질문/정답의 한국어 번역 100개를 넣습니다. 한 행의 형식은 아래와 같습니다. 아래 예시는 형식 설명이며 실험 데이터가 아닙니다.

```json
{"id":"sqa-0000","text":"한국어 질문","answer":"한국어 정답","source":"원문 데이터의 위치와 행 번호"}
```

## 2. 실행 목록 고정

본 실험의 ElevenLabs 경로는 패키지의 `plan → audio → run → judge → summarize`입니다. 노트북과 `scripts/synthesize_elevenlabs.py`는 파일럿·탐색용으로 유지하며, 그 결과를 본 실험 목록과 섞지 않습니다.

ElevenLabs를 사용할 때는 `configs/voices_elevenlabs.json`에서 실제 사용할 슬롯마다 `provenance`에 선정 근거와 한국어 청취 확인 기록을 쓰고 `korean_supported: true`를 기록합니다. 청취하지 않은 슬롯을 일괄 true로 바꾸지 마세요. 미확인 슬롯은 자동 제외되지 않으므로, 수행할 슬롯만 담은 별도 화자 설정 파일을 만들고 그 범위를 보고합니다. `voice_id`나 확인 기록이 없는 슬롯은 음성 생성 시 중단됩니다.

TTS 설정은 `configs/tts_elevenlabs.json`에 있습니다. 현재 `eleven_v3`, `pcm_24000`이며 화자별 `settings`는 요청의 `voice_settings`로 전달됩니다. 변경은 목록 생성 전에 확정합니다.

```bash
# --voice-config와 --tts-config는 하위 명령 앞에 지정합니다.
# 400문항 목록 없이 먼저 native 범위의 계획을 검토하는 예시입니다.
accent-ko --voice-config configs/voices_elevenlabs.json --tts-config configs/tts_elevenlabs.json plan --suite main --groups native --out data/manifests/elevenlabs-native-v2
```

위 계획은 음성 생성이나 API 호출을 하지 않습니다. 선택한 화자 중 미확인 슬롯이 있으면 생성 전에 화자 설정을 정리해야 합니다. 전체 main·defense·ablation·sqa도 같은 두 설정 옵션을 붙여 각각 새 폴더로 계획합니다.

기존 TTSMaker 재현 경로는 기본 설정으로 남아 있습니다:


```bash
accent-ko plan --suite main --out data/manifests/main
accent-ko plan --suite defense --out data/manifests/defense
accent-ko plan --suite ablation --out data/manifests/ablation
accent-ko plan --suite sqa --out data/manifests/sqa
```

각 폴더에 jobs.jsonl, audio.jsonl, provenance.json을 생성합니다. 계획 생성은 실제 합성/추론을 하지 않습니다. full main은 400문항 선정 목록이 없으면 중단됩니다. native만 먼저 목록을 확인하려면 `--groups native`를 사용할 수 있지만 이를 전체 실험으로 보고하면 안 됩니다. 방어는 de/it 출신 native 슬롯, 절제는 de 슬롯으로 제한됩니다.

## 3. 음성 준비

ElevenLabs 본 실험 목록은 다음처럼 생성합니다. 이 명령은 실제 유료 API를 호출합니다.

```bash
export ELEVENLABS_API_KEY=...
accent-ko audio --manifest data/manifests/elevenlabs-native-v2 --limit 1
# 점검 후 --limit을 제거하면 같은 목록으로 이어서 실행합니다.
```

`audio`는 목록에 고정된 공급자·TTS 모델·화자·설정을 사용합니다. clean을 한 번 생성하고 모든 음향 조건에서 재사용합니다. 원문 텍스트·화자 설정·TTS 설정·seed가 달라지면 새 음성 경로를 사용하며, 기존 파일은 출처와 해시가 일치할 때만 재사용합니다. 요청 seed는 기록하지만 공급자의 결정적 재생성을 보장하지 않으므로 실제 WAV와 해시를 보관합니다. 유료 호출은 자동 재시도하지 않습니다.

ElevenLabs 목록의 `import-clean`은 같은 생성 출처가 담긴 JSON과 WAV 해시가 있어야 합니다. 이전 파일럿 스크립트의 메타데이터나 WAV만 있는 파일은 새 설정으로 생성된 것처럼 등록하지 않습니다. 기존 파일럿은 별도로 보존하고 본 실험 경로에서 새로 생성하세요.

아래는 기존 TTSMaker 경로입니다.

TTSMaker의 원래 ID를 확인한 뒤 계정 토큰을 환경변수 `TTSMAKER_TOKEN`에 설정합니다. 토큰을 설정 파일이나 Git에 넣지 않습니다. 기본 API는 공개 v1 계약이며, API 사용 가능 여부/계정별 제한은 실제 계정에서 확인해야 합니다. 연구 당시 서비스와 현재 서비스가 동일한 파형을 생성한다고 가정하지 않습니다.

```bash
accent-ko audio --manifest data/manifests/main
accent-ko audio --manifest data/manifests/defense
accent-ko audio --manifest data/manifests/ablation
accent-ko audio --manifest data/manifests/sqa
```

한 문항·화자의 clean 음성을 한 번만 합성한 뒤 조건별 파형을 만듭니다. 본문이 아닌 고정된 manifest의 TTS 설정을 사용합니다. 재실행은 출처와 파일 해시가 일치하는 음성만 재사용합니다. 합성 실패 시 다른 음성으로 전환하지 않으며 유료 생성 요청을 자동 재시도하지 않습니다.

이미 동일한 설정으로 생성한 한국어 음성이 있으면 API 없이 가져올 수 있습니다. audio-id는 audio.jsonl의 `id`입니다. 출처 설정을 채운 뒤 사용하세요.

```bash
accent-ko import-clean --manifest data/manifests/main --audio-id AUDIO_ID --source /absolute/path/source.wav
accent-ko audio --manifest data/manifests/main
```

음성은 원래 샘플레이트로 보관하고 추론할 때 16 kHz로 변환합니다. 원문 IR이 없으면 해당 잔향 조건에서 중단됩니다. PCM16 저장, resampling, 누락된 속삭임 helper의 재구성은 [차이점 문서](experiment-design.md)에 기록되어 있습니다.

## 4. 모델 환경과 revision

NVIDIA GPU 환경에서 모델별로 별도 가상환경을 두는 것이 좋습니다. 기본 설치 묶음은 Transformers 4.46.3에 맞춘 시작 환경이며, 5개 모델 전부의 호환성을 실기 검증한 잠금 파일은 아닙니다.

```bash
python -m pip install -e '.[inference]'
```

Meta 모델 접근 권한이 필요한 계정에서는 Hugging Face 인증을 준비합니다. 정확한 논문 commit을 확보하면 configs/models.json의 revision에 기록하세요. 동일한 이름의 최신 가중치로 실행하기로 명시적으로 정했다면 다음 명령으로 현재 참조를 고정할 수 있습니다. **현재 commit 고정은 논문 당시 checkpoint 확인을 대신하지 않습니다.**

```bash
python scripts/pin_models.py --out configs/models.lock.json
```

모든 run/judge/transcribe 명령의 `--model-config` 옵션은 하위 명령 앞에 씁니다. 모델 다운로드는 수십 GB 이상일 수 있으며 이 저장소 작성 과정에서는 수행하지 않았습니다. 각 실행의 모델 설정·장치·라이브러리 버전은 결과 옆의 `.meta.json`에 남습니다. 모델 원격 코드가 내부적으로 참조하는 종속 모델 revision은 추가 확인이 필요합니다.

## 5. 모델별 추론

```bash
accent-ko --model-config configs/models.lock.json run --manifest data/manifests/main --model qwen2 --out results/main-qwen2.jsonl
```

같은 형식으로 diva, meralion, minicpm, ultravox를 실행합니다. defense에는 DiVA 작업이 없으며 나머지 4개만 실행합니다. ablation과 sqa는 5개 모두 실행합니다. 서로 다른 모델/실험은 서로 다른 결과 파일을 사용하세요. 요청마다 새 대화와 고정된 seed를 사용합니다. 오디오 모드에는 문항 전사를 함께 전달하지 않습니다.

`--limit 1`은 소규모 동작 확인용입니다. 제한을 제거하고 같은 명령을 실행하면 기존 응답을 중복 생성하지 않고 이어갑니다. 모델·입력 목록이 바뀌거나 이미 사용한 음성 파일이 변경되면 재개를 거절합니다. 오류는 한 건 기록 후 즉시 중단합니다. 원인을 해결한 뒤 새 결과 파일로 실행하세요. 기존 결과를 수정해 오류를 성공으로 바꾸지 마세요.

MERaLiON의 문서상 30초 제한을 넘는 음성은 자동으로 자르지 않고 오류 처리합니다. 다른 모델의 내부 길이 처리도 본 실행 전에 확인해야 합니다. 일부 조건을 수행할 수 없으면 누락 수를 보고하고 전체 재현 결과처럼 집계하지 않습니다.

## 6. 평가 및 요약

```bash
accent-ko --model-config configs/models.lock.json judge --manifest data/manifests/main --responses results/main-qwen2.jsonl --out results/main-qwen2-guard.jsonl
accent-ko summarize --manifest data/manifests/main --responses results/main-qwen2.jsonl --labels results/main-qwen2-guard.jsonl --out results/main-qwen2-summary.json

accent-ko --model-config configs/models.lock.json transcribe --manifest data/manifests/main --out results/transcripts.jsonl
accent-ko wer-summary --transcripts results/transcripts.jsonl --out results/wer.json

accent-ko --model-config configs/models.lock.json judge --kind sqa_judge --manifest data/manifests/sqa --responses results/sqa-qwen2.jsonl --out results/sqa-qwen2-judge.jsonl
accent-ko summarize --manifest data/manifests/sqa --responses results/sqa-qwen2.jsonl --labels results/sqa-qwen2-judge.jsonl --out results/sqa-qwen2-summary.json
```

`summarize`의 main clean 대비 텍스트 차이(`delta_pp`)는 해당 음성과 같은 prompt ID의 텍스트 응답만 사용합니다. natural/synthetic의 400문항을 전체 520문항 텍스트 비율과 비교하지 않습니다. `text_baseline`에 비교 문항 수·유효 평가 수·누락 수·완료 여부·비율을 함께 기록하며, 비교 대상에 누락이나 판정 오류가 있으면 차이는 null입니다. 텍스트 추론 자체는 520문항을 한 번만 실행합니다.

`judge`는 응답 파일의 해시를 고정합니다. 추론을 추가해서 응답 파일이 바뀌었다면 새로운 평가 출력 파일을 지정합니다. 기존 판정과 다른 버전의 응답을 섞지 않습니다. summarize에는 실행하지 않은 나머지 모델의 누락 그룹도 나타나며, 완료 여부를 숨기지 않습니다. 원한다면 5개 모델의 서로 다른 ID를 가진 결과 파일을 합친 뒤 한 번에 평가할 수 있습니다. 중복 ID는 거절합니다.

WER 결과는 원문과 같이 native/natural의 변형 음성만 대상으로 합니다. 합성 억양과 clean은 포함하지 않습니다. `count`를 계획된 수와 함께 확인하세요. 중간 전사 파일의 WER은 완료 결과가 아닙니다.

## 7. 한국어 사람 검토

```bash
accent-ko human-sample --manifest data/manifests/main --responses results/main-qwen2.jsonl --out results/human-50.csv
# CSV의 human_label 50개를 safe 또는 unsafe로 채운 뒤
accent-ko human-score --annotations results/human-50.csv --labels results/main-qwen2-guard.jsonl --out results/human-score.json
```

표본은 한국어 native Echo 정상 응답에서 고정 seed로 50건 추출하며 모델 판정은 보여주지 않습니다. 원문의 표본 모델 배분 규칙은 미공개이므로 사용한 모델/표본 ID와 두 가지 분모의 FP/FN 값을 모두 보고합니다. 한국어 판정 신뢰성 검토를 위해 실패한 라벨을 safe로 바꾸지 않습니다.

## 검증 범위

테스트는 520문항 정렬, 전체 슬롯 계산, 400개 선정 강제, main/defense/ablation의 조합, 텍스트 누출 방지, 에코·잔향 수치, 속삭임 seed, 파일 변조 감지, 모의 응답을 사용한 재개·평가·미완료 집계를 다룹니다. 모의 응답은 테스트 전용이며 실제 실험 결과 파일로 제공하지 않습니다. GPU 모델 로딩·TTS 계정 호출·저자 원본 IR과의 파형 동등성은 별도 확인 대상입니다.
