import re
import subprocess
import time

import numpy as np
import whisper
from collections import deque


# ffmpeg audio capture command
command = [
    'ffmpeg',
    '-f', 'dshow',                                                # DirectShow 입력
    '-i', 'audio=스테레오 믹스(Realtek(R) Audio)',                   # 출력용 캡처 디바이스 이름
    '-f', 's16le',                                                # raw PCM 데이터 형식
    '-acodec', 'pcm_s16le',                                       # PCM 16비트 리틀엔디언
    '-ar', '16000',                                               # 샘플링 레이트: 16kHz
    '-ac', '1',                                                   # 1채널(모노)
    'pipe:1'                                                      # 파이프 출력
]

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7)
model = whisper.load_model("turbo")
model = model.to("cuda")
cnt = 0
worker = 4
capture_length = 1
context_length = 10
context_offset = 9
context_q = deque([np.array([], dtype=np.float32) for _ in range(context_length)], context_length)

prev_stable = ''

print(f"Capture chunk size {capture_length} secs")
print(f"Overlapping length {context_offset} chunks")
print(f"Expected delay is at least {(context_length - context_offset) * capture_length} secs")
print("Transcribe starts in 2 seconds...")
time.sleep(2)
init_time = time.time()

try:
    while True:
        t_next_step = int(time.time()) + capture_length
        start = time.time() - init_time
        raw_audio = process.stdout.read(int(16000 * capture_length) * 2)
        audio_array = np.frombuffer(raw_audio, dtype=np.int16)

        normalized_audio = audio_array.astype(np.float32) / 32768.0
        context_q.append(normalized_audio)
        # context_audio = np.pad(np.concatenate(context_q), (0, 16000 * 30 - len(normalized_audio)), mode='constant')
        context_audio = np.concatenate(context_q)

        if cnt % (context_length - context_offset) == 1 or context_length - context_offset == 1:
            pred = model.transcribe(audio=context_audio, language='en')
            if prev_stable != pred['text'].capitalize():
                print(f"[{start:.3f} - {time.time() - init_time:.3f}] {pred['text']} {pred}")
                prev_stable = pred['text'].capitalize()

        cnt += 1
        while time.time() < t_next_step - 0.1:
            time.sleep(0.1)

except KeyboardInterrupt:
    process.terminate()
    process.wait()