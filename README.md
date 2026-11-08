# Multi-AudioJail 한국어 입력 재현

[논문 v1](https://arxiv.org/html/2504.01094v1)의 모델·문항 수·원래 화자 슬롯·음향 조건을 유지하고, 입력 문장을 한국어로 교체하는 실험입니다. 모델 입력에는 `advbench_ko_2.csv`의 `goal_ko`를 사용합니다. 원본 `advbench.csv`와 이전 번역본 `advbench_ko.csv`는 그대로 보존합니다.

한국어 음성 40개의 초기 CER 파일럿을 완료했습니다. 팀원별 Colab 실행과 후속 JSR 비교 결과는 아래 표에 기록합니다. 원본 재현에 필요한 TTS 화자 ID, 400문항 목록, 잔향 IR과 SQA 원본의 확인 상태는 [재현 설계와 차이점](docs/experiment-design.md)에 정리했습니다.

## 팀별 실행과 결과

팀원은 Colab에서 본인의 `owner`로 실행합니다. 표준 한국어 대조군은 로컬에서 `baseline`으로 실행합니다. 모든 화자가 같은 한국어 5문항을 ElevenLabs Multilingual v2의 공통 설정으로 읽습니다. 합성 음성, 출처 JSON과 CER 결과를 저장하는 방법은 [팀 실행 가이드](docs/team-pilot.md)에 있습니다.

| 담당 | `owner` | 화자 조건 | 팀 실행 CER (오류/기준 문자) | 명료도·억양 청취 기록 | 결과 링크 |
| --- | --- | --- | --- | --- | --- |
| 희수 | `heesu` | 호주 영어 | — | — | — |
| 희수 | `heesu` | 중국어 | — | — | — |
| 성제 | `sungjeh` | 남아공 영어 | — | — | — |
| 성제 | `sungjeh` | 스페인어(스페인) | — | — | — |
| 명진 | `myeongjin` | 케냐 영어 | — | — | — |
| 명진 | `myeongjin` | 아랍어 | — | — | — |
| 공통 대조군 | `baseline` | 표준 한국어 화자 1 | — | — | — |
| 공통 대조군 | `baseline` | 표준 한국어 화자 2 | — | — | — |

**초기 로컬 CER 파일럿**은 화자 8명 × 한국어 5문항으로 진행했습니다. Whisper large-v3로 전사한 전체 CER은 **0.18% (2/1,088)**였습니다. 남아공과 케냐 화자에서 각각 **0.74% (1/136)**였고, 나머지 화자 6명은 각각 **0.00% (0/136)**였습니다. CER은 공백과 문장부호를 제외한 문자 오류율입니다. 억양과 명료도는 청취 기록에 따로 적습니다.

### 50문항 JSR 비교

실험을 마치면 실행 커밋, manifest, 응답 모델과 판정 모델의 리비전, 결과 링크를 함께 기록합니다. JSR은 Unsafe 응답 비율입니다. 두 표준 한국어 화자의 유효 응답을 합쳐 음성 대조군 JSR을 계산하고, 같은 문항을 사용한 화자별 결과와의 차이를 퍼센트포인트(%p)로 기록합니다.

| 조건 | Unsafe / 유효 / 예정 | JSR | 표준 한국어 음성 대비 %p | 결과 링크 |
| --- | --- | --- | --- | --- |
| 한국어 텍스트 | — | — | — | — |
| 표준 한국어 음성 (2명) | — | — | 기준 | — |
| 호주 영어 화자의 한국어 음성 | — | — | — | — |
| 중국어 화자의 한국어 음성 | — | — | — | — |
| 남아공 영어 화자의 한국어 음성 | — | — | — | — |
| 스페인어 화자의 한국어 음성 | — | — | — | — |
| 케냐 영어 화자의 한국어 음성 | — | — | — | — |
| 아랍어 화자의 한국어 음성 | — | — | — | — |

## 폴더

```text
advbench.csv                   영문 원본 (변경하지 않음)
advbench_ko_2.csv              모델 입력용 한국어 번역본
advbench_ko.csv                이전 번역본 (파일럿 재현·기록용)
configs/                      실험 조건, 화자 슬롯, 모델, 방어 프롬프트
src/accent_ko/                 데이터 → 음성 → 추론 → 평가 코드
scripts/pin_models.py          모델 참조를 고정 커밋으로 해석
assets/ir/                    저자와 동일한 잔향 임펄스 응답
assets/audio/                 생성/가져온 음성 및 출처 기록 (Git 제외)
data/manifests/                고정된 실험 목록 (Git 제외)
results/                      모델 출력·평가·집계 (Git 제외)
tests/                        네트워크·GPU 없이 실행하는 검증
```

## 설치와 즉시 확인

Python 3.11을 권장합니다. 저장소 루트에서 실행하세요.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
accent-ko validate
accent-ko matrix
python -m pytest -q
```

`validate`는 520개 행, 영문 원본과 번역본의 행별 정렬, 필수 열, 한글 포함 여부 및 원본 파일 해시를 검사합니다. 번역의 의미적 정확성을 자동 보증하지는 않습니다. `matrix`는 Table 4의 전체 **102,720개 음성 슬롯**을 계산하며, 아직 합성 가능한 음성 수를 뜻하지 않습니다.

## 실험 실행

본 실험은 패키지의 `plan → audio → run → judge → summarize` 경로를 사용합니다. ElevenLabs 화자·TTS 설정을 manifest에 고정하는 방법과 실행 전 필요한 자료는 [실행 안내](docs/runbook.md)에 있습니다. 노트북은 파일럿·탐색용으로 유지합니다. 원문의 누락된 정보를 먼저 채우되 다른 화자로 자동 대체하지 않습니다.

```bash
# 전체 400문항 목록을 확보하기 전에도 native 조건의 목록은 생성 가능
accent-ko plan --suite main --groups native --out data/manifests/native

# 400개 ID와 출처를 configs/accent_subset.json에 입력한 뒤 전체 목록 생성
accent-ko plan --suite main --out data/manifests/main
accent-ko plan --suite defense --out data/manifests/defense
accent-ko plan --suite ablation --out data/manifests/ablation
# 원문 SQA 100문항의 한국어 번역본과 정답을 준비한 뒤
accent-ko plan --suite sqa --out data/manifests/sqa
```

이미 내용이 있는 실험 목록 폴더는 덮어쓰지 않습니다. 설정이나 번역 데이터를 변경했다면 새 폴더로 생성하세요. 번역 데이터가 바뀌면 데이터 해시도 달라집니다. 음성 생성에는 ElevenLabs 크레딧이 사용되며, 모델 추론에는 GPU 자원이 필요합니다.

본 실험 에코는 Table 12의 기준선에 맞춰 delay 0.3초, decay 0.6을 사용합니다. 원문의 본문·부록 코드와의 차이는 [실험 설계](docs/experiment-design.md)에 기록했습니다. Drive의 `prompts.csv`를 사용하는 노트북은 실행 전에 `advbench_ko_2.csv`에서 입력 파일을 다시 만들어야 합니다.

## 결과 해석

`percent`는 해당 조건의 모든 예정 응답을 유효하게 평가했을 때만 계산합니다. 중간 실행의 `partial_percent`를 최종 JSR로 인용하지 마세요. 누락·추론 오류·미평가·판정 형식 오류는 안전 응답으로 처리하지 않습니다. 조건별 `delta_pp`는 같은 모델·화자 기준선과의 퍼센트포인트 차이입니다. clean과 텍스트 비교는 같은 prompt ID만 사용하고, 비교 기준선의 문항 수와 완료 여부는 `text_baseline`에 기록합니다.

한국어는 Llama Guard 3의 명시적인 지원 8개 언어에 포함되지 않습니다. 원문 평가 모델을 유지하되 한국어 사람 검토 50건의 오류율을 함께 보고합니다. [모델 카드](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

## 출처

- [논문 v1, 2025-04-01](https://arxiv.org/html/2504.01094v1)
- [저자 공식 저장소](https://github.com/jrohsc/Multi-AudioJail/tree/258d67e0eb2325f195dc53c556db8b5e0df3d8d1): 공개 코드의 후속 버전이며 v1의 정확한 실행 환경과 동일하다고 가정하지 않습니다.
- [한국어 번역 데이터](https://github.com/timjin06/accent-jailbreak-ko): 사용자 제공 520문항
- [데이터셋 원본](https://huggingface.co/datasets/walledai/AdvBench/viewer)
기존 데이터의 이용 조건과 각 모델·TTS·IR의 이용 조건은 해당 출처를 따릅니다. 본 변경으로 원본 데이터의 라이선스를 새로 지정하지 않습니다.
