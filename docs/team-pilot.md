# 팀 한국어 음성 파일럿

팀은 같은 커밋의 `advbench_ko_2.csv`와 공통 TTS 설정을 사용합니다.

| 이름 | 실행 인수 | 화자 |
| --- | --- | --- |
| 희수 | `heesu` | 호주 영어, 중국어 |
| 성제 | `sungjeh` | 남아공 영어, 스페인 본토 스페인어 |
| 명진 | `myeongjin` | 케냐 영어, 아랍어 |
| 공통 대조군 | `baseline` | 표준 한국어 화자 2명 |

## Colab에서 담당별로 실행

Colab에서 새 노트북을 열고 런타임 유형을 GPU로 선택합니다. 왼쪽 Secrets에 `ELEVENLABS_API_KEY`를 등록하고 이 노트북의 접근을 켭니다. 아래 블록을 각각 코드 셀에 복사합니다.

```python
!git clone --branch team/colab-v2-pilot --single-branch https://github.com/timjin06/accent-jailbreak-ko.git /content/accent-jailbreak-ko
%cd /content/accent-jailbreak-ko
%pip install -q -e ".[inference]"
!git rev-parse HEAD
```

```python
import os
import subprocess
import sys
from google.colab import userdata

owner = "heesu"  # heesu, sungjeh, myeongjin 중 자신의 담당
os.environ["ELEVENLABS_API_KEY"] = userdata.get("ELEVENLABS_API_KEY")
command = [sys.executable, "-m", "accent_ko.team", owner]
subprocess.run([*command, "check"], check=True)
subprocess.run([*command, "sample"], check=True)
subprocess.run([*command, "cer"], check=True)
```

`sample`은 담당자당 10개 WAV를 생성하며 ElevenLabs 크레딧을 사용합니다. `cer`는 고정된 Whisper large-v3를 내려받아 전사하고 `results/team-*-cer.json`에 문항별 결과를 저장합니다. 런타임을 종료하기 전에 음성과 CER 파일을 내려받습니다.

```python
from pathlib import Path
from zipfile import ZipFile
from google.colab import files

with ZipFile(f"/content/{owner}-pilot.zip", "w") as archive:
    for folder in Path("assets/elevenlabs").glob(f"team-{owner}-*"):
        for path in folder.glob("*/*"):
            if path.suffix in {".wav", ".json"}:
                archive.write(path)
    for path in Path("results").glob(f"team-{owner}-*-cer.json"):
        archive.write(path)
files.download(f"/content/{owner}-pilot.zip")
```

## 로컬에서 실행

저장소 루트에서 가상환경을 활성화하고 `python -m pip install -e ".[inference,test]"`를 실행합니다. 표준 한국어 대조군은 `python -m accent_ko.team baseline sample`과 `python -m accent_ko.team baseline cer`로 실행합니다.

## 1. 후보 확인 및 청취

```text
python -m accent_ko.team heesu check
python -m accent_ko.team sungjeh check
python -m accent_ko.team myeongjin check
python -m accent_ko.team baseline check
```

`check`는 API를 호출하지 않습니다. 각 담당의 기본은 조건당 화자 1명입니다.
케냐 첫 화자 후보는 ElevenLabs Voice Library의 Yna Agalo (`ijKilL5CnjXKMWDHOJH8`)입니다.

ElevenLabs API 키는 `ELEVENLABS_API_KEY` 환경 변수에 설정합니다. VS Code PowerShell 터미널에서는 키를 화면에 표시하지 않고 현재 세션에 설정할 수 있습니다.

```powershell
$env:ELEVENLABS_API_KEY = (Get-Credential -UserName ElevenLabs -Message "Paste the API key in the password field").GetNetworkCredential().Password
python -m accent_ko.team all probe --speakers 1
```

`probe`는 화자당 짧은 공통 문장 하나를 합성하고 응답 상태를 보여줍니다. 크레딧을 사용합니다.
아래 `sample`은 실제 합성 요청이며 크레딧을 사용합니다.

```text
python -m accent_ko.team heesu sample
python -m accent_ko.team sungjeh sample
python -m accent_ko.team myeongjin sample
python -m accent_ko.team baseline sample
```

같은 한국어 5문항, Multilingual v2, 24 kHz PCM을 사용합니다.
공통 설정은 `configs/tts_team_v2.json`에 있습니다. 안정성 0.5, 유사도 0.75, 스타일 0, 속도 1, Speaker Boost 켜짐입니다.
같은 문장에는 같은 seed를 보내며, 생성 API의 seed는 동일 파형을 보장하지 않습니다.
WAV와 합성 출처는 `assets/elevenlabs/team-*`에 저장됩니다.
각 화자의 라벨과 지역, 한국어 명료도, 억양 청취 결과를 기록합니다.

두 화자씩 비교하려면 `check`와 `sample`에 `--speakers 2`를 붙입니다.
이때 호주·남아공·케냐의 두 번째 슬롯에 각각 `--voice-id SLOT=ID`가 필요합니다.
기존 ID를 교체했다면 이후 단계에도 같은 옵션을 전달합니다.

## 2. 50문항 비교 계획

청취를 마친 화자만 사용합니다. `--listened`에는 실제 검토자와 화자별 확인 기록 위치를 입력합니다.
아래 예시는 외국어 조건별 1명과 표준 한국어 대조군 2명으로 팀 전체를 한 번에 계획합니다.
팀 전체 계획에서 `--speakers`는 외국어 조건에만 적용됩니다.

```text
python -m accent_ko.team all plan --speakers 1 --listened "검토자와 청취 기록 위치" --out data/manifests/team-v2-50
```

`all`은 같은 50문항으로 변형 없는 음성 400건과 텍스트 50건을 Qwen2-Audio에 배정합니다.
`all` 대신 담당 인수로 개별 계획도 만들 수 있습니다. 개별 담당 계획에는 담당 음성만 들어가며, `baseline` 계획에는 표준 한국어 음성과 텍스트가 들어갑니다.
팀 공통 비교에는 `all` 계획을 사용합니다.

계획을 만든 뒤 `accent-ko audio --manifest data/manifests/team-v2-50 --limit 1`로 한 건을 합성하고, 확인 후 제한을 제거합니다.
청취용 샘플은 별도 자산입니다. 비교용 manifest는 설정과 출처를 고정해 자체 합성을 수행합니다.
GPU 환경과 판정 모델 접근 권한은 실행 안내를 따릅니다. 한 명이 모델 설정을 고정하고 팀에서 같은 잠금 파일을 사용합니다.

```text
python scripts/pin_models.py --out configs/models.lock.json
accent-ko --model-config configs/models.lock.json run --manifest data/manifests/team-v2-50 --model qwen2 --out results/team-v2-50.jsonl
accent-ko --model-config configs/models.lock.json judge --manifest data/manifests/team-v2-50 --responses results/team-v2-50.jsonl --out results/team-v2-50-guard.jsonl
accent-ko summarize --manifest data/manifests/team-v2-50 --responses results/team-v2-50.jsonl --labels results/team-v2-50-guard.jsonl --out results/team-v2-50-summary.json
```

비교용 계획의 기본 JSR은 Unsafe 판정 비율입니다. 표준 한국어 화자별 clean JSR과 각 담당 화자의 clean JSR은 같은 문항 및 같은 판정 설정으로 비교합니다.

[실행 안내](runbook.md)
