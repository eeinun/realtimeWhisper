import subprocess
import time
import shutil

import numpy as np
import whisper
from faster_whisper import WhisperModel

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
    
    
def transcribe_faster(whisper_faster_model: WhisperModel, audio_data, model_params: dict):
    iter, _ = whisper_faster_model.transcribe(
        audio=audio_data,
        **model_params
    )
    return [{'start': x.start, 'end': x.end, 'text': x.text, 'no_speech_prob': x.no_speech_prob} for x in iter]


def transcribe_whisper(whisper_model: whisper.Whisper, audio_data, model_params):
    pred = whisper_model.transcribe(
        audio=audio_data,
        **model_params
    )
    return pred['segments']

def get_transcribe_function(model, name, model_params):
    if name == 'whisper':
        print("Using whisper model")
        def wrapper(audio_data):
            return transcribe_whisper(model, audio_data, model_params)
        return wrapper
    if name == 'faster-whisper':
        print("Using faster-whisper model")
        def wrapper(audio_data):
            return transcribe_faster(model, audio_data, model_params)
        return wrapper

# Issues
# Whisper got stuck when sudden volume increase

# --------------------------------------------------|
# Internal constants
# Audio capture parameters
CAPTURE_LENGTH = 0.5                                # Capture audio every this second.
CAPTURE_DEVICE = '스테레오 믹스(Realtek(R) Audio)'     # Capture device. To change, read readme.md
SAMPLE_RATE = 16000                                 # Sample rate of audio capture. Whisper uses 16kHz.
MAXIMUM_BUFFERING_LENGTH = 20                       # If buffer grows longer than this second, oldest segment would be popped out.
BG_NOISE_THRESHOLD = 70                             # To erase small unwanted noise. Lower if no background noise. Increase if noise interferes.
# Whisper parameters
WHISPER_MODEL_NAME = 'turbo'                        # Whisper model name. Refer to offical whispher docs for more info. https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
WHISPER_SOURCE_LANG = 'en'                          # What language would whisper detect. Set this to None if source language is unknown.
VAD_FILTER = True                                   # Filter that detects voice activity.
INITIAL_PROMPT = ''                                 # Initial prompt. Setting this some prompt may improve detection.
                                                    # or you can add glossaries.
                                                    # However, it may say some part of prompt itself
WHISPER_PARAMS = {
    'language': WHISPER_SOURCE_LANG,
    'temperature': 0.0,
    'initial_prompt': INITIAL_PROMPT
}
WHISPER_FASTER_PARAMS = {
    'language': WHISPER_SOURCE_LANG,
    'temperature': 0.0,
    'vad_filter': VAD_FILTER,
    'initial_prompt': INITIAL_PROMPT
}
DUMMY_SEGMENTS = [{'end': 0.0, 'text': '', 'no_speech_prob': -1}] # Dummy

# Translator parameter
TRANS_DESTINATION_LANG = 'ko'

# ffmpeg audio capture command
FFMPEG_COMMAND = [
    'ffmpeg',
    '-f', 'dshow',                                  # DirectShow input
    '-i', f'audio={CAPTURE_DEVICE}',                # Capture device name
    '-f', 's16le',                                  # Raw PCM data type
    '-acodec', 'pcm_s16le',                         # PCM 16-bit little endian
    '-ar', str(SAMPLE_RATE),                        # Sample rate = 16kHz
    '-ac', '1',                                     # Mono channel
    'pipe:1'                                        # Pipe output
]
# --------------------------------------------------|

# --------------------------------------------------|
# External (User defined) constants
TRANSCRIPTION_MODEL = 'faster-whisper'
INFERENCE_DEVICE = 'cuda'
# --------------------------------------------------|

# Ready to capture audio
process = subprocess.Popen(FFMPEG_COMMAND, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7)
normalized_buffer = np.zeros(SAMPLE_RATE * 2 * 4).astype(np.float32)
print(f"Capture chunk size {CAPTURE_LENGTH} secs")

# Prepare transcription function
if TRANSCRIPTION_MODEL == 'whisper':
    model = whisper.load_model(WHISPER_MODEL_NAME)
    model = model.to(INFERENCE_DEVICE)
    transcribe = get_transcribe_function(model, TRANSCRIPTION_MODEL, WHISPER_PARAMS)
elif TRANSCRIPTION_MODEL == 'faster-whisper':
    model = WhisperModel(WHISPER_MODEL_NAME, device=INFERENCE_DEVICE, compute_type='float16')
    transcribe = get_transcribe_function(model, TRANSCRIPTION_MODEL, WHISPER_FASTER_PARAMS)
else:
    raise Exception(f'Given model "{WHISPER_MODEL_NAME}" is invalid.')

# Warm up, initialize, and start
transcribe(np.zeros(1).astype(np.float32))
prev = DUMMY_SEGMENTS.copy()
print("Transcribe starts in 1 second...")
time.sleep(1)
init_time = time.time()

while True:
    try:
        while True:
            t_next_step = int(time.time()) + CAPTURE_LENGTH
            start = time.time() - init_time

            terminal_size = shutil.get_terminal_size(fallback=(120, 50))
            w = terminal_size.columns

            # Process audio
            raw_audio = process.stdout.read(int(SAMPLE_RATE * CAPTURE_LENGTH) * 2)
            audio_array = np.frombuffer(raw_audio, dtype=np.int16).copy()
            audio_array[np.abs(audio_array) < BG_NOISE_THRESHOLD] = 0
            normalized_audio = audio_array.astype(np.float32) / 32768.0 * 20
            normalized_buffer = np.hstack((normalized_buffer, normalized_audio))

            if audio_array.mean() != 0.0: # If audio_array's mean is exactly 0.0, it means there was no speech.
                pred = transcribe(
                    np.concatenate( # Pad to 30sec
                        (normalized_buffer[:min(len(normalized_buffer), 30 * SAMPLE_RATE)],
                        (np.zeros(max(0, 30 * SAMPLE_RATE - len(normalized_buffer)))).astype(np.float32))
                    )
                )
            else:
                pred = prev.copy()

            first_segment = pred[0]['text'].strip() if len(pred) > 0 else ""
            # print(f"\n[{start:.3f} - {time.time() - init_time:.3f}][{len(normalized_buffer)}] < {pred} >")
            # print(f"\r< {audio_array.min()}, {audio_array.max()} | {audio_array.mean()} >{first_segment[:min(len(first_segment), w)]}:" + ' ' * max(0, w - len(first_segment)), end='')

            rt_text = ' '.join([x['text'] for x in pred])
            print(f"\r{rt_text[:min(len(rt_text), w - 2)]} {'|' if int(time.time() / CAPTURE_LENGTH) % 2 else ':'}" + ' ' * max(0, w - len(rt_text) - 2), end='')

            if ( # AND conditions. This ensures no error from statement body.
                    len(prev) > 0
                and len(pred) > 0
            ) and ( # OR conditions. This detects when should segment be popped.
                   len(pred) > 1 # There are more than 2 segments.
                or len(normalized_buffer) > MAXIMUM_BUFFERING_LENGTH * SAMPLE_RATE  # Need to modify buffer.
                or pred[-1]['end'] > MAXIMUM_BUFFERING_LENGTH                       # Need to reduce buffer size or whisper error.
                or len(normalized_buffer) > SAMPLE_RATE * MAXIMUM_BUFFERING_LENGTH  # Need to reduce buffer size.
                or audio_array.mean() == 0.0                                        # Silence. Maybe segment is completed.
                or (first_segment.upper() == prev[0]['text'].upper() and pred[0]['end'] == prev[0]['end'])  # Same text transcribed at same segment.
            ):
                # remove part of audio which corresponds to popped segment
                normalized_buffer = normalized_buffer[int(pred[0]['end'] * SAMPLE_RATE):]
                # remove front silence to reduce misinterpretation or hallucination
                normalized_buffer = normalized_buffer[np.concatenate((np.nonzero(normalized_buffer)[0], np.array([0])))[0]:]

                # Check if it is the most frequent hallucination "Thank you." or just nothing.
                if not (pred[0]['no_speech_prob'] > 2e-11 and 'THANK' in first_segment.upper()):
                    buffer_length_disp = (
                            f"[{start:.3f} - {time.time() - init_time:.3f}] "
                            + "-" * int(len(normalized_buffer) / SAMPLE_RATE)
                            + f"({len(normalized_buffer) // SAMPLE_RATE})"
                    )
                    print(f"\r{buffer_length_disp}" + ' ' * max(0, w - len(buffer_length_disp)))
                    print(f"{first_segment}")
                    asyncio.run(translate(first_segment, dest=TRANS_DESTINATION_LANG))
                    # edit pred just for prev update
                    if len(pred) > 0:
                        pred = pred[1:]

            prev = pred.copy()
            if len(normalized_buffer) > SAMPLE_RATE * 30:
                normalized_buffer = np.zeros(int(SAMPLE_RATE * CAPTURE_LENGTH)).astype('float32')
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