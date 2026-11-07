import json
import re
import time
import queue
import threading
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from collections import deque
import requests
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from kokoro_onnx import Kokoro
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from io import BytesIO

from vosk import Model as VoskModel, KaldiRecognizer

import reminders     # task reminders (reminders.py, same folder)
import permissions   # read-only, permission-controlled laptop access (permissions.py)

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:1.5b"

VOSK_DIR = BASE_DIR / "models" / "vosk-model-small-en-us-0.15"

# ---------------- Web UI ----------------
UI_HOST = "127.0.0.1"
UI_PORT = 8081
UI_URL = f"http://{UI_HOST}:{UI_PORT}"
UI_FILE = BASE_DIR / "index.html"

# ---------------- Kokoro TTS (lightweight int8) ----------------
KOKORO_DIR = BASE_DIR / "models" / "kokoro"
KOKORO_MODEL = KOKORO_DIR / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = KOKORO_DIR / "voices-v1.0.bin"

# CPU threads for TTS. Kam = kam CPU load, par thoda slow. 2 ya 3 try karo.
TTS_THREADS = 2

VOICE_GENDER = "female"

# female: af_heart (best), af_bella, af_nicole, af_sarah, af_nova
# male:   am_michael, am_adam, am_eric, am_onyx
VOICE_NAMES = {
    "male": "am_michael",
    "female": "af_heart",
}

TTS_SPEED = 1.05  # 1.0 normal, 1.1 thoda fast
TTS_LANG = "en-us"

REQUEST_TIMEOUT = 180
MAX_HISTORY = 8

# Context budget for conversation history sent to LLM (chars).
# Leaves room for system prompt, conversation context, and response.
CONTEXT_CHAR_BUDGET = 1500

# Microphone configuration
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

# "alex" -> male voice, "alexa" -> female voice
WAKE_PHRASE = re.compile(r"\b(alexa|alex)\b", re.IGNORECASE)
LEADING_CALL_NAME = re.compile(
    r"^\s*(?:hey\s+)?(?P<name>alexa|alex)\b"
    r"(?:[\s,.:;!?-]+(?:(?:or)\s+)?(?:alexa|alex)\b)*"
    r"[\s,.:;!?-]*",
    re.IGNORECASE,
)
WAKE_RESPONSE = "Hey Shree, how can I help you?"

WHISPER_MODEL_NAME = "base.en"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

MAX_UTTERANCE_SECONDS = 12
SILENCE_SECONDS = 1.0
SILENCE_THRESHOLD = 250

# Number of seconds of audio retained before wake detection
PRE_ROLL_SECONDS = 1.5
PRE_ROLL_BLOCKS = int(PRE_ROLL_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

# Queues and events
audio_queue = queue.Queue(maxsize=100)
control_audio_queue = queue.Queue(maxsize=40)
speech_queue = queue.Queue()
keyboard_queue = queue.Queue()

MIC_MUTED = threading.Event()
STOP_EVENT = threading.Event()
INTERRUPT_SPEECH = threading.Event()

_tts = None
_conversation_history = []
_web_whisper_model = None
_web_whisper_lock = threading.Lock()

# ---------------- State shown by the UI ----------------
# waiting   : waiting for the wake word (Alex / Alexa)
# listening : active, waiting for you to speak
# hearing   : you are speaking
# thinking  : LLM is working
# speaking  : Alex is talking
UI_STATE = {"state": "starting", "voice": VOICE_GENDER, "heard": "", "reply": ""}
ACTIVE = False


def set_state(state):
    UI_STATE["state"] = state
    UI_STATE["voice"] = VOICE_GENDER


def idle_state():
    return "listening" if ACTIVE else "waiting"


def get_tts():
    global _tts

    if _tts is None:
        if not KOKORO_MODEL.exists() or not KOKORO_VOICES.exists():
            raise FileNotFoundError(
                f"Kokoro files not found in {KOKORO_DIR}\n"
                "Download kokoro-v1.0.int8.onnx and voices-v1.0.bin there."
            )

        print("[Loading Kokoro voice model...]")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = TTS_THREADS
        opts.inter_op_num_threads = 1

        session = ort.InferenceSession(
            str(KOKORO_MODEL),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _tts = Kokoro.from_session(session, str(KOKORO_VOICES))

    return _tts


def clean_for_speech(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#~>|]+", "", text)
    text = re.sub(r"[\U00010000-\U0010FFFF\u2600-\u27BF]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def synthesize(text):
    tts = get_tts()

    samples, sample_rate = tts.create(
        text,
        voice=VOICE_NAMES[VOICE_GENDER],
        speed=TTS_SPEED,
        lang=TTS_LANG,
    )

    if samples is None or len(samples) == 0:
        return None, None

    return np.asarray(samples, dtype=np.float32), sample_rate


def speak_text(text):
    text = clean_for_speech(text)

    if not text or not re.search(r"\w", text):
        return

    try:
        audio, sample_rate = synthesize(text)

        if audio is None or audio.size == 0:
            return

        pause = 0.20

        audio = np.concatenate((
            audio,
            np.zeros(
                int(sample_rate * pause),
                dtype=np.float32,
            ),
        ))

        # Poll playback so a local stop command can interrupt output.
        set_state("speaking")
        sd.play(audio, samplerate=sample_rate, blocking=False)
        while sd.get_stream() is not None and sd.get_stream().active:
            if INTERRUPT_SPEECH.is_set() or STOP_EVENT.is_set():
                sd.stop()
                break
            time.sleep(0.03)

    except Exception as exc:
        print(f"\n[TTS error: {exc}]")


def speech_worker():
    while True:
        job = speech_queue.get()

        try:
            if job is None:
                return

            speak_text(job)

        except Exception as exc:
            print(f"\n[Speech worker error: {exc}]")

        finally:
            speech_queue.task_done()


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True,
)
speech_thread.start()


def speak_reply(text):
    """Speak while temporarily preventing self-triggering."""

    MIC_MUTED.set()

    try:
        clear_audio_queue()
        speech_queue.put(text)
        speech_queue.join()

    finally:
        clear_audio_queue()
        MIC_MUTED.clear()
        set_state(idle_state())


def on_reminder_due(message):
    """Called by the reminder thread. Waits until Alex is free, then speaks."""
    while not STOP_EVENT.is_set() and (
        MIC_MUTED.is_set() or UI_STATE["state"] in ("hearing", "thinking", "speaking")
    ):
        time.sleep(0.5)

    if STOP_EVENT.is_set():
        return

    print(f"\nAlex: {message}")
    UI_STATE["reply"] = message
    speak_reply(message)


def _init_managers():
    """Start a fresh, in-memory conversation for this assistant run."""
    global _conversation_history
    _conversation_history = []
    print("[Started a new private conversation; chat is not saved to disk]")


SYSTEM_BASE = """You are Alex, a friendly personal AI assistant
running locally on the user's laptop.

Your name is always Alex. The user may say "Alex" or "Alexa" to get
your attention; those are wake words, not a change to your identity.
Never claim to be Amazon Alexa or another assistant.

Reply in 1 to 3 short sentences. Be natural and helpful.
Reply only in simple natural English. Do not use emojis, markdown,
bullets, or asterisks.

IMPORTANT CONVERSATION RULES:
- Answer the user's latest request directly.
- Do not repeat your previous answer or repeat the same sentence.
- Do not repeat or paraphrase the user's words unless needed for clarity.
- If the user says 'do not repeat', briefly acknowledge once and then stop repeating.
- Do not mention today's date unless the user asks for the date or time.
- Avoid filler and do not give the same answer again unless asked.
- You cannot open files, folders, apps, websites, email or chats yourself.
  Never claim that you opened, read, sent, deleted or changed anything."""


# ============================================================
# PC DATE / TIME / DAY
# ============================================================

def build_system_prompt():
    """System prompt that also carries the PC's current clock."""
    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    return (
        SYSTEM_BASE
        + f"\n\nThe user's PC clock says: {now}. "
        "Use this only when the user asks about the date, day or time."
    )


TIME_QUERY = re.compile(r"\b(?:time|date|day)\b", re.IGNORECASE)
TIME_INTENT = re.compile(
    r"\b(?:what(?:'s| is)?|tell me|current|which|today'?s|time now|date today)\b",
    re.IGNORECASE,
)


def local_time_answer(text):
    """Answer time/date/day questions from the PC clock, without the LLM."""
    t = text.lower()

    if not (TIME_QUERY.search(t) and TIME_INTENT.search(t)):
        return None

    # Only short questions, so normal chat that contains "day" is not hijacked.
    if len(t.split()) > 8:
        return None

    now = datetime.now()
    want_time = bool(re.search(r"\btime\b", t))
    want_date = bool(re.search(r"\bdate\b", t))
    want_day = bool(re.search(r"\bday\b", t))

    if not (want_time or want_date or want_day):
        return None

    clock = now.strftime("%I:%M %p").lstrip("0")
    day = now.strftime("%A")
    date = now.strftime("%d %B %Y").lstrip("0")

    if want_time and not (want_date or want_day):
        return f"It's {clock}."
    if want_day and not (want_date or want_time):
        return f"Today is {day}."
    if want_date and not (want_day or want_time):
        return f"Today's date is {date}."

    return f"It's {clock} on {day}, {date}."


def get_answer(user_text):
    global _conversation_history
    _conversation_history.append({"role": "user", "content": user_text})
    context_messages = []
    char_count = 0
    for item in reversed(_conversation_history[-MAX_HISTORY:]):
        size = len(item["content"]) + 8
        if context_messages and char_count + size > CONTEXT_CHAR_BUDGET:
            break
        context_messages.append(item)
        char_count += size
    context_messages.reverse()

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            *context_messages,
        ],
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.6,
            "num_ctx": 2048,
            "num_predict": 160,
        },
    }

    answer_parts = []
    buffer = ""
    start = time.perf_counter()

    def flush_sentence(sentence):
        sentence = clean_for_speech(sentence)
        if sentence:
            speech_queue.put(sentence)

    try:
        with requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:

            response.raise_for_status()
            print("\nAlex: ", end="", flush=True)

            for line in response.iter_lines():
                if STOP_EVENT.is_set() or INTERRUPT_SPEECH.is_set():
                    break
                if not line:
                    continue

                data = json.loads(line)

                if data.get("error"):
                    raise RuntimeError(data["error"])

                content = data.get("message", {}).get("content", "")

                if content:
                    print(content, end="", flush=True)

                    answer_parts.append(content)
                    buffer += content

                    # Speak complete sentences as they arrive.
                    while True:
                        buffer = buffer.lstrip()

                        match = re.match(
                            r"^(.+?[.!?])\s+",
                            buffer,
                            re.DOTALL,
                        )

                        if not match:
                            break

                        flush_sentence(match.group(1))
                        buffer = buffer[match.end():]

                if data.get("done"):
                    break

        if buffer.strip():
            flush_sentence(buffer)

        print()
        print(f"[LLM: {time.perf_counter() - start:.2f}s]")

        answer = "".join(answer_parts).strip()
        UI_STATE["reply"] = answer

        if not answer:
            return False

        _conversation_history.append({"role": "assistant", "content": answer})
        del _conversation_history[:-MAX_HISTORY]

        return True

    except requests.exceptions.ConnectionError:
        print("\nOllama is not running. Start it using: ollama serve")

    except requests.exceptions.Timeout:
        print("\nOllama timed out. Try again.")

    except (
        requests.exceptions.RequestException,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"\nOllama error: {exc}")

    return False

HELP_TEXT = """
Alex commands:
  :voice male       - Switch to male voice
  :voice female     - Switch to female voice
  :help             - Show commands
  exit              - Quit Alex

Ask: "what time is it", "what's the date", "which day is it"

Tasks and reminders (saved on disk, they survive restarts):
  "remind me to call mom at 5 pm"
  "remind me to drink water in 10 minutes"
  "remind me tomorrow at 9 am to submit the form"
  "remind me after 1 hour to do homework and after 2 hours to call sir"  (many at once)
  "add task buy milk"
  "what are my tasks"
  "mark task 1 done" / "delete task 2" / "clear all tasks"
  "snooze" (after a reminder) or "snooze 5 minutes"

Laptop access (READ-ONLY, only folders listed in security_config.json,
Alex asks "May I ...?" and you must say "yes" every single time):
  "read file notes.txt in projects"
  "list folder projects"
  "open website example.com"      (only sites listed in security_config.json)
  "open chrome"                   (only apps listed in security_config.json)
"""


def handle_command(user_text):
    global VOICE_GENDER

    command = user_text.strip().lower()

    if command in ("exit", "quit", ":exit", "exit alex", "stop listening"):
        return "exit"

    if command in ("stop", "cancel speech"):
        INTERRUPT_SPEECH.set()
        sd.stop()
        clear_speech_queue()
        return True

    # Handle this locally so the LLM cannot accidentally repeat its last answer.
    if re.search(r"\b(?:do not|don't|stop)\s+repeat(?:ing)?\b", command):
        print("Alex: Okay, I won't repeat myself.")
        speech_queue.put("Okay, I won't repeat myself.")
        return True

    if command in (":help", "help", "?"):
        print(HELP_TEXT)
        return True

    if command.startswith(":voice "):
        selected = command.split(maxsplit=1)[1]

        if selected in ("male", "female"):
            VOICE_GENDER = selected
            print(f"Alex voice changed to {VOICE_GENDER}.")
            speech_queue.put("Voice changed.")
        else:
            print("Use :voice male or :voice female.")

        return True

    return False


def process_text(user_text):
    """Process either typed text or a recognized voice request."""
    global VOICE_GENDER

    if not user_text.strip():
        return True

    set_state("thinking")
    UI_STATE["heard"] = user_text.strip()
    UI_STATE["reply"] = ""

    # Treat a leading Alex/Alexa as the assistant's call name, not as a
    # question for the language model. A name-only call gets a fixed reply.
    call_match = LEADING_CALL_NAME.match(user_text)
    if call_match:
        VOICE_GENDER = "female" if call_match.group("name").lower() == "alexa" else "male"
    request_text = LEADING_CALL_NAME.sub("", user_text, count=1).strip()
    if not request_text:
        INTERRUPT_SPEECH.clear()
        speak_reply(WAKE_RESPONSE)
        return True

    INTERRUPT_SPEECH.clear()

    result = handle_command(request_text)

    if result == "exit":
        return False

    # Mute the microphone during the complete response.
    MIC_MUTED.set()

    try:
        clear_audio_queue()

        if result is not True:
            # Order matters:
            #  1. a waiting approval question gets the next message ("yes" / anything else)
            #  2. reminders and tasks
            #  3. read-only laptop access requests (always ask first)
            #  4. time / date / day from the PC clock
            local = (
                permissions.handle_pending("voice", request_text)
                or reminders.handle(request_text)
                or permissions.handle(request_text, "voice")
                or local_time_answer(request_text)
            )
            if local:
                print(f"Alex: {local}")
                UI_STATE["reply"] = local
                speech_queue.put(local)
            else:
                get_answer(request_text)

        speech_queue.join()

    finally:
        clear_audio_queue()
        MIC_MUTED.clear()
        set_state(idle_state())

    return True


# ============================================================
# MICROPHONE AUDIO CALLBACK
# ============================================================

def clear_audio_queue():
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def clear_speech_queue():
    while True:
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            break


def microphone_callback(indata, frames, time_info, status):
    if STOP_EVENT.is_set():
        return

    audio = indata[:, 0].copy()

    if MIC_MUTED.is_set():
        try:
            control_audio_queue.put_nowait(audio)
        except queue.Full:
            try:
                control_audio_queue.get_nowait()
                control_audio_queue.put_nowait(audio)
            except queue.Empty:
                pass
        return

    try:
        audio_queue.put_nowait(audio)

    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            pass


def stop_command_monitor(vosk_model):
    """Recognize fixed local controls during speech; never forward them to Ollama."""
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    stop_patterns = re.compile(r"\b(?:stop|cancel speech|stop listening|exit alex)\b", re.I)
    while not STOP_EVENT.is_set():
        try:
            frame = control_audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if recognizer.AcceptWaveform(frame.tobytes()):
                heard = json.loads(recognizer.Result()).get("text", "").lower()
                if stop_patterns.search(heard):
                    if "stop listening" in heard:
                        handle_command("stop listening")
                        STOP_EVENT.set()
                    elif "exit alex" in heard:
                        handle_command("exit alex")
                        STOP_EVENT.set()
                    else:
                        handle_command("stop")
                    if "stop listening" in heard or "exit alex" in heard:
                        STOP_EVENT.set()
                recognizer.Reset()
        except (ValueError, TypeError):
            recognizer.Reset()


# ============================================================
# SPEECH-TO-TEXT USING FASTER-WHISPER
# ============================================================

def transcribe_audio(whisper_model, audio_frames):
    if not audio_frames:
        return ""

    audio_int16 = np.concatenate(audio_frames)

    if audio_int16.size == 0:
        return ""

    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    try:
        segments, _ = whisper_model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        # Consume the generator to complete transcription.
        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return transcript.strip()

    except Exception as exc:
        print(f"\n[Whisper error: {exc}]")
        return ""


# ============================================================
# RECORD THE USER'S REQUEST
# ============================================================

def record_request(whisper_model, initial_frames):
    print("\n[Listening to your question...]")

    frames = list(initial_frames)

    block_seconds = BLOCK_SIZE / SAMPLE_RATE

    speech_started = any(
        np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        >= SILENCE_THRESHOLD
        for frame in frames
    )

    silence_duration = 0.0
    elapsed = 0.0

    while not STOP_EVENT.is_set():
        if elapsed >= MAX_UTTERANCE_SECONDS:
            break

        try:
            frame = audio_queue.get(timeout=0.25)

        except queue.Empty:
            elapsed += 0.25

            if speech_started:
                silence_duration += 0.25

                if silence_duration >= SILENCE_SECONDS:
                    break

            continue

        frames.append(frame)
        elapsed += block_seconds

        rms = float(np.sqrt(
            np.mean(frame.astype(np.float32) ** 2)
        ))

        if rms >= SILENCE_THRESHOLD:
            speech_started = True
            silence_duration = 0.0

        elif speech_started:
            silence_duration += block_seconds

            if silence_duration >= SILENCE_SECONDS:
                break

    if not speech_started:
        print("[No speech detected. Try again.]")
        return ""

    transcript = transcribe_audio(whisper_model, frames)

    # Remove the wake phrase if it appears in the transcription.
    transcript = WAKE_PHRASE.sub("", transcript, count=1)

    return transcript.strip(" \t\r\n,.:;!?-")


# ============================================================
# CONTINUOUS CONVERSATION AFTER ONE-TIME WAKE WORD
# ============================================================

def record_active_utterance(whisper_model):
    """
    Once Alex is activated, wait for normal speech and record one
    utterance until a pause is detected. No wake word is required.
    """
    frames = []
    block_seconds = BLOCK_SIZE / SAMPLE_RATE
    speech_started = False
    silence_duration = 0.0
    elapsed = 0.0
    set_state("listening")

    while not STOP_EVENT.is_set():
        if MIC_MUTED.is_set():
            time.sleep(0.03)
            continue

        try:
            frame = audio_queue.get(timeout=0.25)
        except queue.Empty:
            if speech_started:
                silence_duration += 0.25
                elapsed += 0.25
                if silence_duration >= SILENCE_SECONDS:
                    break
            continue

        rms = float(np.sqrt(
            np.mean(frame.astype(np.float32) ** 2)
        ))

        if not speech_started:
            # Ignore room noise until actual speech begins.
            if rms < SILENCE_THRESHOLD:
                continue
            speech_started = True
            set_state("hearing")
            print("\n[Listening to your question...]")

        frames.append(frame)
        elapsed += block_seconds

        if rms >= SILENCE_THRESHOLD:
            silence_duration = 0.0
        else:
            silence_duration += block_seconds

        if silence_duration >= SILENCE_SECONDS:
            break

        if elapsed >= MAX_UTTERANCE_SECONDS:
            break

    if not frames:
        return ""

    transcript = transcribe_audio(whisper_model, frames)
    return transcript.strip(" \t\r\n,.:;!?-")


def listen_for_wake_word(vosk_model, whisper_model):
    """
    Wait for 'Alex' (male voice) or 'Alexa' (female voice) once.
    After activation, stay in continuous conversation mode until Alex is stopped.
    """
    global VOICE_GENDER, ACTIVE

    set_state("waiting")
    print("\nWaiting for wake word: Alex (male) or Alexa (female)")
    print("After activation, you can keep talking without repeating it.")
    print("Microphone is active only for this run after your approval.")
    print("You can also type messages in the terminal.")
    print("Press Ctrl+C to stop.\n")

    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    pre_roll = deque(maxlen=PRE_ROLL_BLOCKS)
    activated = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        callback=microphone_callback,
    ):
        while not STOP_EVENT.is_set():

            # Keyboard input remains available at all times.
            try:
                typed = keyboard_queue.get_nowait()
            except queue.Empty:
                typed = None

            if typed is not None:
                if not process_text(typed):
                    break
                clear_audio_queue()
                pre_roll.clear()
                continue

            if MIC_MUTED.is_set():
                time.sleep(0.03)
                clear_audio_queue()
                continue

            # Once activated, listen for ordinary speech continuously.
            if activated:
                transcript = record_active_utterance(whisper_model)

                if STOP_EVENT.is_set():
                    break

                if transcript:
                    print(f"\nYou: {transcript}")
                    if not process_text(transcript):
                        break
                continue

            # Before activation, only the wake phrase can activate Alex.
            try:
                frame = audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            pre_roll.append(frame)
            accepted = recognizer.AcceptWaveform(frame.tobytes())

            if accepted:
                detected_text = json.loads(
                    recognizer.Result()
                ).get("text", "").lower()
            else:
                detected_text = json.loads(
                    recognizer.PartialResult()
                ).get("partial", "").lower()

            if detected_text:
                print(
                    f"\rWaiting for 'Alex' / 'Alexa' | Heard: {detected_text}       ",
                    end="",
                    flush=True,
                )

            # Sirf final result pe decide karo, warna "alexa" bolne pe partial "alex" pe male trigger ho jayega
            if not accepted:
                continue

            wake_match = WAKE_PHRASE.search(detected_text)
            if not wake_match:
                continue

            VOICE_GENDER = "female" if wake_match.group(1).lower() == "alexa" else "male"
            activated = True
            ACTIVE = True
            print(f"\n[Wake word detected! Alex is now active ({VOICE_GENDER} voice).]")
            speak_reply(WAKE_RESPONSE)

            # Discard wake-word audio so it is not transcribed as a question.
            pre_roll.clear()
            clear_audio_queue()
            recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
            print("\n[Continuous conversation enabled. Speak naturally.]")


# ============================================================
# OPTIONAL KEYBOARD INPUT
# ============================================================

def keyboard_listener():
    while not STOP_EVENT.is_set():
        try:
            typed = input(
                "\nType a message or command: "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            keyboard_queue.put("exit")
            return

        if typed:
            if MIC_MUTED.is_set() and typed.lower().strip() in (
                "stop", "cancel speech", "stop listening", "exit alex", "exit", "quit"
            ):
                handle_command(typed)
                if typed.lower().strip() in ("stop listening", "exit alex", "exit", "quit"):
                    STOP_EVENT.set()
                continue
            keyboard_queue.put(typed)


# ============================================================
# MICROPHONE CONSENT
# ============================================================

def request_microphone_consent():
    """Ask before Alex opens the microphone input stream."""
    print("\nAlex needs microphone access for voice commands.")
    print("No microphone stream will be opened unless you approve.")

    try:
        answer = input("Allow Alex to access your microphone? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nMicrophone permission was not granted. Alex will close.")
        return False

    if answer not in ("y", "yes"):
        print("Microphone access denied. Alex will close without listening.")
        return False

    print("Microphone consent granted for this run.")
    print("Windows may also require microphone access in Settings.")
    return True


# ============================================================
# WEB API (used by the Queen UI in D:\Alex\static\index.html)
# ============================================================

class WebChatIn(BaseModel):
    conversation_id: int = 1
    message: str


app = FastAPI(title="Alex")
_web_histories = {}
_web_history_lock = threading.Lock()


@app.middleware("http")
async def protect_api_origin(request: Request, call_next):
    """Block cross-site browser access to every API route, including GETs."""
    if request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site", "").lower()
        allowed = (origin is not None and origin.rstrip("/") == UI_URL)
        allowed = allowed or (origin is None and fetch_site == "same-origin")
        if not allowed:
            return JSONResponse(status_code=403, content={"detail": "Forbidden or missing origin"})
    return await call_next(request)


def _require_local_ui_origin(request: Request):
    """Require an explicit same-origin browser request for sensitive API posts."""
    origin = request.headers.get("origin")
    if not origin or origin.rstrip("/") != UI_URL:
        raise HTTPException(status_code=403, detail="Forbidden or missing origin")


@app.get("/api/health")
async def web_health():
    return {"ok": True}


@app.get("/api/state")
async def web_state():
    return UI_STATE


@app.get("/api/time")
async def web_time():
    """Current PC date, day and time (handy for the UI)."""
    now = datetime.now()
    return {
        "time": now.strftime("%I:%M:%S %p"),
        "date": now.strftime("%d %B %Y"),
        "day": now.strftime("%A"),
        "iso": now.isoformat(timespec="seconds"),
    }


@app.get("/api/tasks")
async def web_tasks():
    """Pending tasks / reminders (for the UI)."""
    return {"tasks": reminders.all_tasks(), "done": reminders.recent_done()}


@app.post("/api/chat")
async def web_chat(data: WebChatIn, request: Request):
    # Only our own UI may talk to this endpoint. This stops a random website
    # in your browser from sending "read file ..." and then "yes" for you.
    _require_local_ui_origin(request)

    text = data.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message must not be empty")

    channel = f"web:{data.conversation_id}"

    # Approval answer first, then reminders, then laptop access, then PC clock.
    local = (
        permissions.handle_pending(channel, text)
        or reminders.handle(text)
        or permissions.handle(text, channel)
        or local_time_answer(text)
    )
    if local:
        with _web_history_lock:
            history = _web_histories.setdefault(data.conversation_id, [])
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": local})
            del history[:-MAX_HISTORY]

        def quick_reply():
            yield json.dumps({"content": local, "done": False}) + "\n"
            yield json.dumps({"content": "", "done": True}) + "\n"

        return StreamingResponse(quick_reply(), media_type="application/x-ndjson")

    with _web_history_lock:
        history = _web_histories.setdefault(data.conversation_id, [])
        history.append({"role": "user", "content": text})
        context = list(history[-MAX_HISTORY:])

    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": build_system_prompt()}, *context],
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.6, "num_ctx": 2048, "num_predict": 160},
    }

    def stream_reply():
        answer_parts = []
        buffer = ""
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        answer_parts.append(content)
                        buffer += content
                        while True:
                            buffer = buffer.lstrip()
                            match = re.match(r"^(.+?[.!?])\s+", buffer, re.DOTALL)
                            if not match:
                                break
                            sentence = match.group(1).strip()
                            buffer = buffer[match.end():]
                            yield json.dumps({"content": sentence, "done": False}) + "\n"
                    if chunk.get("done"):
                        break
            if buffer.strip():
                yield json.dumps({"content": buffer.strip(), "done": False}) + "\n"
            answer = "".join(answer_parts).strip()
            if answer:
                with _web_history_lock:
                    history = _web_histories.setdefault(data.conversation_id, [])
                    history.append({"role": "assistant", "content": answer})
                    del history[:-MAX_HISTORY]
            yield json.dumps({"content": "", "done": True}) + "\n"
        except requests.exceptions.ConnectionError:
            yield json.dumps({"content": "Ollama is not running. Start it using: ollama serve", "done": True}) + "\n"
        except requests.exceptions.Timeout:
            yield json.dumps({"content": "Ollama timed out. Try again.", "done": True}) + "\n"
        except Exception as exc:
            yield json.dumps({"content": f"Alex error: {exc}", "done": True}) + "\n"

    return StreamingResponse(stream_reply(), media_type="application/x-ndjson")


def transcribe_browser_audio(audio):
    global _web_whisper_model
    if not audio:
        return ""
    if _web_whisper_model is None:
        with _web_whisper_lock:
            if _web_whisper_model is None:
                _web_whisper_model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                    cpu_threads=4,
                )
    segments, _ = _web_whisper_model.transcribe(
        BytesIO(audio),
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


@app.post("/api/transcribe")
async def web_transcribe(request: Request, file: UploadFile = File(...)):
    _require_local_ui_origin(request)
    audio = await file.read(20 * 1024 * 1024 + 1)
    if len(audio) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio recording is too large")
    try:
        text = await run_in_threadpool(transcribe_browser_audio, audio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local voice recognition failed: {exc}") from exc
    if not text:
        raise HTTPException(status_code=422, detail="I couldn't hear any speech. Please try again.")
    return {"text": text}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not UI_FILE.is_file():
        raise HTTPException(status_code=404, detail=f"UI file not found: {UI_FILE}")
    return UI_FILE.read_text(encoding="utf-8")


def start_ui(open_browser=True):
    """Start the web server in a background thread and open the Queen UI."""
    import uvicorn

    if not UI_FILE.is_file():
        print(f"[UI not found: expected {UI_FILE}]")
        return

    server = uvicorn.Server(
        uvicorn.Config(app, host=UI_HOST, port=UI_PORT, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()

    # Wait until the server answers, then open the browser.
    for _ in range(50):
        try:
            if requests.get(f"{UI_URL}/api/health", headers={"Origin": UI_URL}, timeout=0.5).ok:
                break
        except requests.exceptions.RequestException:
            time.sleep(0.2)

    print(f"[Queen UI running at {UI_URL}]")
    if open_browser:
        webbrowser.open(UI_URL)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nStarting Alex...")

    # Ask before loading the voice models or opening the microphone.
    if not request_microphone_consent():
        STOP_EVENT.set()
        return

    if not VOSK_DIR.exists():
        raise FileNotFoundError(
            f"Vosk model not found: {VOSK_DIR}\n"
            "Download and extract the English Vosk model first."
        )

    # Start a temporary in-memory conversation.
    _init_managers()

    # Load Kokoro now and warm it up so the first reply is not slow.
    tts = get_tts()
    print("[Warming up voice...]")
    for voice_name in VOICE_NAMES.values():
        tts.create("Hello.", voice=voice_name, speed=TTS_SPEED, lang=TTS_LANG)

    print("[Loading Vosk wake-word model...]")
    vosk_model = VoskModel(str(VOSK_DIR))

    print("[Loading faster-whisper base.en on CPU...]")
    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=4,
    )

    print("\nAlex is ready!")
    print(HELP_TEXT)

    # Start the Queen UI together with the voice assistant.
    start_ui()

    keyboard_thread = threading.Thread(
        target=keyboard_listener,
        daemon=True,
    )
    keyboard_thread.start()
    control_thread = threading.Thread(
        target=stop_command_monitor,
        args=(vosk_model,),
        daemon=True,
    )
    control_thread.start()

    # Saved tasks are loaded from disk; due ones are announced by voice.
    reminders.start_scheduler(on_reminder_due)
    print(f"[Reminders: {reminders.pending_count()} pending task(s) in {reminders.TASKS_FILE}]")

    try:
        # No startup greeting: Alex stays silent until wake word.
        listen_for_wake_word(
            vosk_model,
            whisper_model,
        )

    except KeyboardInterrupt:
        print("\nClosing Alex...")

    finally:
        STOP_EVENT.set()
        reminders.stop_scheduler()
        MIC_MUTED.set()
        INTERRUPT_SPEECH.set()
        sd.stop()

        clear_audio_queue()

        clear_speech_queue()
        speech_queue.put(None)
        speech_queue.join()
        speech_thread.join(timeout=3)
        control_thread.join(timeout=1)

        print("Alex closed.")


if __name__ == "__main__":
    if "--web" in sys.argv:
        # UI only (no terminal voice mode)
        reminders.start_scheduler(on_reminder_due)
        start_ui()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nClosing Alex...")
    else:
        # Voice assistant + UI together
        main()
