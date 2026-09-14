# 실행 환경

기본 패키지 및 모델 선택 의존성은 pyproject.toml에서 관리합니다.
검증 환경은 Python 3.11이며 `test-lock.txt`는 이 변경에서 통과한 CPU 테스트 패키지 버전입니다.

GPU 환경에서는 각 모델의 공식 설치 지침도 확인합니다. Transformers 4.46.3은 MERaLiON 공식 예제를 기준으로 한 시작점이며, 모든 모델의 GPU 호환성을 검증한 결과가 아닙니다. 특히 remote code가 추가로 참조하는 패키지와 checkpoint revision을 실행 기록에 남겨야 합니다. 테스트 잠금 파일을 GPU 검증 결과로 해석하지 마세요.

- Qwen: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- DiVA: https://huggingface.co/WillHeld/DiVA-llama-3-v0-8b
- MERaLiON: https://huggingface.co/MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION
- MiniCPM: https://huggingface.co/openbmb/MiniCPM-o-2_6
- Ultravox: https://huggingface.co/fixie-ai/ultravox-v0_4_1-llama-3_1-8b
