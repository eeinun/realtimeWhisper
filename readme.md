- `stt.py` : whisper 모델에 CUDA 사용. 테스트한 버전 CUDA 12.6, torch 2.5.1+cu124.
- `stt_parallel.py` : 미완. CPU 사용
- `Ctrl+C`를 입력하고 `1`을 제외한 다른 키를 입력하면 임시저장된 오디오를 초기화할 수 있습니다. 받아쓰기가 이상해질 때 사용하면 도움이 됩니다.
- 오디오 입력장치로 `스테레오 믹스(Realtek(R) Audio)`가 선택되어있습니다. 다른 입력 장치로 바꾸려면 다음 ffmpeg 커맨드로 입력장치 이름을 확인한 후 설정해주세요.
  - `ffmpeg -list_devices true -f dshow -i _`