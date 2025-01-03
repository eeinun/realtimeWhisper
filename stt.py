import subprocess
import time

import numpy as np
import whisper

# for translation
from googletrans import Translator
import asyncio


async def translate(text, dest):
    print("T> ", end='')
    if len(text) < 4:
        return
    async with Translator() as translator:
        result = await translator.translate(text, dest=dest)
    print(result.text)


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

capture_length = 0.5
SAMPLE_RATE = 16000
MAXIMUM_BUFFERING_LENGTH = 10
DESTINATION_LANG = 'ko'

normalized_buffer = np.zeros(SAMPLE_RATE * 2 * 4).astype(np.float32)

print(f"Capture chunk size {capture_length} secs")
model.transcribe(np.zeros(1).astype(np.float32))
prev = [{'end': 3.0, 'text': ''}]

print("Transcribe starts in 1 second...")
time.sleep(1)
init_time = time.time()

while True:
    try:
        while True:
            t_next_step = int(time.time()) + capture_length
            start = time.time() - init_time
            raw_audio = process.stdout.read(int(SAMPLE_RATE * capture_length) * 2)
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)

            normalized_audio = audio_array.astype(np.float32) / 32768.0
            normalized_buffer = np.hstack((normalized_buffer, normalized_audio))

            pred = model.transcribe(
                audio=np.concatenate((normalized_buffer, np.zeros(30 * SAMPLE_RATE - len(normalized_buffer)).astype(np.float32))),
                language='en',
                initial_prompt="Segment should be sentence-wise. You must NOT write anything if there is silence. Silence. No you are NOT ALLOWED TO WRITE ANYTHING.''"
            )

            # print(f"[{start:.3f} - {time.time() - init_time:.3f}][{len(normalized_buffer)}] < {pred['segments']} >")
            if (
                len(pred['segments']) > 2 or
                len(normalized_buffer) > MAXIMUM_BUFFERING_LENGTH * SAMPLE_RATE or
                pred['segments'][-1]['end'] > MAXIMUM_BUFFERING_LENGTH or
                (pred['segments'][0]['text'] == prev[0]['text'] and pred['segments'][0]['end'] == prev[0]['end']) or
                pred['segments'][0]['text'].upper() == prev[0]['text'].upper()
            ):
                if not (pred['segments'][0]['no_speech_prob'] > 1.5e-11 and 'THANK' in pred['segments'][0]['text'].upper()) or not pred['segments'][0]['text'] == ' .':
                    print(f"[{start:.3f} - {time.time() - init_time:.3f}]")
                    print(f"{pred['segments'][0]['text']}")
                    asyncio.run(translate(pred['segments'][0]['text'], dest=DESTINATION_LANG))
                normalized_buffer = normalized_buffer[int(pred['segments'][0]['end'] * SAMPLE_RATE):]
            prev = pred['segments']
            while time.time() < t_next_step - 0.1:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Transcribe Stopped. Press any key to continue or press 1 to quit.")
        try:
            if input() == '1':
                process.terminate()
                process.wait()
                exit()
            normalized_buffer = np.zeros(SAMPLE_RATE * 2 * 4).astype(np.float32)
        except Exception:
            process.terminate()
            process.wait()
            exit()