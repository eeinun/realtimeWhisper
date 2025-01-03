import subprocess
import multiprocessing
import time

import numpy as np
import whisper
from collections import deque


def infer(audio_data):
    return model.transcribe(audio=audio_data, language='en')


def process_task(pid, task_q, result_q):
    while True:
        task_data = task_q.get()
        if task_data is None:
            break
        timestamp, audio_data = task_data
        print(infer(audio_data))
        result_q.put(pid)


if __name__ == "__main__":
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
    model = whisper.load_model("base")
    cnt = 0

    NUM_PROCESSES = 4
    task_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()

    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=process_task, args=(i + 1, task_q, result_q))
        p.start()
        processes.append(p)

    capture_length = 0.5
    context_length = 5
    context_offset = 3
    context_q = deque([np.array([], dtype=np.float32) for _ in range(context_length)], context_length)
    init_time = time.time()

    try:
        active_processes = [i + 1 for i in range(NUM_PROCESSES)]
        process_status = {i + 1: True for i in range(NUM_PROCESSES)}

        while True:
            t_next_step = time.time() + capture_length
            start = time.time() - init_time
            raw_audio = process.stdout.read(int(16000 * capture_length) * 2)
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)
            print(f"[{len(audio_array)}]")
            normalized_audio = audio_array.astype(np.float32) / 32768.0
            context_q.append(normalized_audio)
            context_audio = np.concatenate(context_q)

            # Busy wait
            while cnt % (context_length - context_offset) == 1:
                # Check finished process
                while not result_q.empty():
                    finished_process_id = result_q.get()
                    process_status[finished_process_id] = True  # 해당 프로세스를 다시 가용 상태로 변경

                # 가용한 프로세스를 찾음
                available_process = None
                for proc_id, is_available in process_status.items():
                    if is_available:
                        available_process = proc_id
                        break

                if available_process is not None:
                    # 가용한 프로세스에게 작업 전달
                    process_status[available_process] = False  # 프로세스 상태를 사용 중으로 설정
                    task_q.put((f"[{start:.3f} - {time.time() - init_time:.3f}]", context_audio))  # 작업 큐에 오디오 데이터 전달
                    break

                print("All processes are busy now")
                time.sleep(0.1)

            cnt += 1
            while time.time() < t_next_step:
                time.sleep(0.1)

    except KeyboardInterrupt:
        process.terminate()
        process.wait()