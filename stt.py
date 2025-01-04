import subprocess
import time
import shutil

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
BG_NOISE_THRESHOLD = 50
DUMMY_SEGMENT = {'segments': [{'end': 0.0, 'text': '', 'no_speech_prob': -1}]}

normalized_buffer = np.zeros(SAMPLE_RATE * 2 * 4).astype(np.float32)

print(f"Capture chunk size {capture_length} secs")
model.transcribe(np.zeros(1).astype(np.float32))
prev = DUMMY_SEGMENT.copy()

print("Transcribe starts in 1 second...")
time.sleep(1)
init_time = time.time()

while True:
    try:
        while True:
            t_next_step = int(time.time()) + capture_length
            start = time.time() - init_time
            raw_audio = process.stdout.read(int(SAMPLE_RATE * capture_length) * 2)
            audio_array = np.frombuffer(raw_audio, dtype=np.int16).copy()
            audio_array[np.abs(audio_array) < BG_NOISE_THRESHOLD] = 0

            normalized_audio = audio_array.astype(np.float32) / 32768.0 * 20
            normalized_buffer = np.hstack((normalized_buffer, normalized_audio))

            if audio_array.mean() != 0.0:
                pred = model.transcribe(
                    audio=np.concatenate((normalized_buffer, np.zeros(30 * SAMPLE_RATE - len(normalized_buffer)).astype(np.float32))),
                    language='en',
                    temperature=0.0,
                    # initial_prompt="Segment should be sentence-wise. You must NOT write anything if there is silence. Silence. No you are NOT ALLOWED TO WRITE ANYTHING.''"
                )
            else:
                pred = prev.copy()

            terminal_size = shutil.get_terminal_size(fallback=(120, 50))
            w = terminal_size.columns
            first_segment = pred['segments'][0]['text'].strip() if len(pred['segments']) > 0 else ""
            # print(f"\n[{start:.3f} - {time.time() - init_time:.3f}][{len(normalized_buffer)}] < {pred['segments']} >")
            # print(f"\r< {audio_array.min()}, {audio_array.max()} | {audio_array.mean()} >{first_segment[:min(len(first_segment), w)]}:" + ' ' * max(0, w - len(first_segment)), end='')
            print(f"\r{first_segment[:min(len(first_segment), w)]}|" + ' ' * max(0, w - len(first_segment)), end='')
            if ( 1 # AND conditions. This ensures no error from statement body.
                    and len(prev['segments']) > 0
                    and len(pred['segments']) > 0
                    and pred['segments'][-1]['end'] * SAMPLE_RATE <= len(normalized_buffer)
            ) and ( 0 # OR conditions. This detects when should segment is popped.
                or len(pred['segments']) > 1                                        # There are more than 2 segments.
                or len(normalized_buffer) > MAXIMUM_BUFFERING_LENGTH * SAMPLE_RATE  # Need to modify buffer.
                or pred['segments'][-1]['end'] > MAXIMUM_BUFFERING_LENGTH           # Need to reduce buffer size.
                or audio_array.mean() == 0.0                                        # Silence. Maybe segment is completed.
                or (first_segment.upper() == prev['segments'][0]['text'].upper() and pred['segments'][0]['end'] == prev['segments'][0]['end'])  # Same text transcribed at same segment.
            ):
                # remove part of audio which corresponds to popped segment
                normalized_buffer = normalized_buffer[int(pred['segments'][0]['end'] * SAMPLE_RATE):]
                # remove front silence to reduce misinterpretation or hallucination
                normalized_buffer = normalized_buffer[np.concatenate((np.nonzero(normalized_buffer)[0], np.array([0])))[0]:]
                # Check if it is the most frequent hallucination "Thank you." or just nothing.
                if not (pred['segments'][0]['no_speech_prob'] > 2e-11 and 'THANK' in first_segment.upper()) or not first_segment == ' .':
                    print(f"\r[{start:.3f} - {time.time() - init_time:.3f}]" + ' ' * max(0, w - 15))
                    print(f"{first_segment}")
                    asyncio.run(translate(first_segment, dest=DESTINATION_LANG))
                    # edit pred just for prev update
                    if len(pred['segments']) > 0:
                        pred['segments'] = pred['segments'][1:]

            prev = pred.copy()
            # wait for next timeslot
            while time.time() < t_next_step - 0.2:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print()
        print("Transcribe Stopped. Press any key to continue or press 1 to quit.")
        try:
            if input() == '1':
                process.terminate()
                process.wait()
                exit()
            normalized_buffer = np.zeros_like(normalized_buffer)
        except Exception:
            process.terminate()
            process.wait()
            exit()