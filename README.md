# Multi-AudioJail 한국어 입력 재현

[논문 v1](https://arxiv.org/html/2504.01094v1)의 모델·문항 수·원래 화자 슬롯·음향 조건을 유지하고, 입력 문장을 한국어로 교체하는 실험입니다. 원본 `advbench.csv`와 사용자 번역본 `advbench_ko.csv`는 그대로 보존합니다.

**현재 상태:** 데이터 검사, 실험 목록 생성, TTS/기존 음성 가져오기, 음향 변형, 5개 모델 어댑터, Llama Guard 평가, Whisper WER, SQA 평가, 사람 검토 표본과 집계 코드가 있습니다. GPU 실험 결과는 아직 없습니다. 저자의 정확한 TTS 화자 ID·400문항 선정 목록·잔향 IR·SQA 원본이 확인되지 않아, 완전 동일 재현이 완료되었다고 주장하지 않습니다. [재현 설계와 차이점](docs/experiment-design.md)을 먼저 읽어 주세요.

## 폴더

```text
advbench.csv / advbench_ko.csv  원문·사용자 번역본 (변경하지 않음)
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

전체 순서와 실행 전 필요한 자료는 [실행 안내](docs/runbook.md)에 있습니다. 원문의 누락된 정보를 먼저 채우되 다른 화자로 자동 대체하지 않습니다.

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

이미 내용이 있는 실험 목록 폴더는 덮어쓰지 않습니다. 설정을 변경했다면 새 폴더로 생성하세요. 음성 생성과 추론 명령은 실제 자원/API를 사용하므로 이 저장소 작성 과정에서는 실행하지 않았습니다.

## 결과 해석

`percent`는 해당 조건의 모든 예정 응답을 유효하게 평가했을 때만 계산합니다. 중간 실행의 `partial_percent`를 최종 JSR로 인용하지 마세요. 누락·추론 오류·미평가·판정 형식 오류는 안전 응답으로 처리하지 않습니다. 조건별 `delta_pp`는 같은 모델·화자 기준선과의 퍼센트포인트 차이입니다.

한국어는 Llama Guard 3의 명시적인 지원 8개 언어에 포함되지 않습니다. 원문 평가 모델을 유지하되 한국어 사람 검토 50건의 오류율을 함께 보고합니다. [모델 카드](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

## 출처

- [논문 v1, 2025-04-01](https://arxiv.org/html/2504.01094v1)
- [저자 공식 저장소](https://github.com/jrohsc/Multi-AudioJail/tree/258d67e0eb2325f195dc53c556db8b5e0df3d8d1): 공개 코드의 후속 버전이며 v1의 정확한 실행 환경과 동일하다고 가정하지 않습니다.
- [한국어 번역 데이터](https://github.com/timjin06/accent-jailbreak-ko): 사용자 제공 520문항

기존 데이터의 이용 조건과 각 모델·TTS·IR의 이용 조건은 해당 출처를 따릅니다. 본 변경으로 원본 데이터의 라이선스를 새로 지정하지 않습니다.
