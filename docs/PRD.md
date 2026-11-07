# Alex — Product Requirements Document (PRD)

This PRD describes the requirements of the **Alex Personal AI Assistant** based on what is actually implemented in the codebase. Requirements that are not implemented are explicitly listed as such; planned ideas are never presented as delivered functionality.

## 1. Product overview and goals

**Alex** is a personal AI assistant that runs locally on the user's Windows laptop and communicates through voice.

### Goals

- Provide a hands-free, voice-first assistant for personal use.
- Keep all processing on the local machine (local LLM, local STT, local TTS).
- Respond with short, natural spoken answers.
- Persist user tasks and reminders across restarts.
- Restrict laptop access so the assistant can only perform narrowly scoped, user-approved actions.

## 2. Target users and use cases

- Primary user: a single person (the project owner) on their personal laptop.
- Use cases:
  - Wake the assistant with "Alex" / "Alexa" and ask questions verbally.
  - Get spoken answers for time/date/day questions.
  - Set reminders ("remind me to call mom at 5 pm") and have them spoken when due.
  - Read small text files or list folders, after explicit approval.
  - Open approved websites / launch approved apps, after explicit approval.
  - Use a local web UI to view state, conversation and task queue.

## 3. Implemented features

Verified against source code — all of the following currently exist in the implementation:

| # | Feature | Implementation |
| --- | --- | --- |
| 1 | Wake word activation ("Alex" / "Alexa") | `Chatapp.py` — Vosk final-result wake detection |
| 2 | Continuous conversation after activation | `Chatapp.py` — `record_active_utterance`, no wake word needed afterwards |
| 3 | Speech-to-text | `Chatapp.py` — faster-whisper `base.en` (English, CPU, int8, VAD) |
| 4 | Text-to-speech | `Chatapp.py` — Kokoro `kokoro-v1.0.int8.onnx`, female default |
| 5 | LLM response generation | `Chatapp.py` — Ollama `qwen2.5:1.5b` streaming chat |
| 6 | Local PC time/date/day answers | `Chatapp.py` — regex-based local answers |
| 7 | Task / reminder management | `reminders.py` — `data/tasks.json` persistence + scheduler |
| 8 | Read-only laptop access with approvals | `permissions.py` — allowlists + single-use approvals |
| 9 | Voice commands (`:voice`, `:help`, exit / stop) | `Chatapp.py` — `handle_command` |
| 10 | Local web API + Queen UI | `Chatapp.py` FastAPI app + `index.html` |
| 11 | Web audio transcription endpoint | `Chatapp.py` — `POST /api/transcribe` (backend only; see limitations) |

## 4. Functional requirements

### FR-001 — Wake word activation (implemented)

- The assistant shall detect the wake phrase "Alex" (male voice) or "Alexa" (female voice) using the Vosk wake model and final recognition results.
- The wake reply shall be spoken and continuous conversation shall start afterwards.

### FR-002 — Continuous conversation (implemented)

- After activation the assistant shall accept spontaneous speech without repeating the wake word.

### FR-003 — Speech-to-text (implemented)

- The assistant shall transcribe English speech with faster-whisper `base.en` running on CPU with int8 quantization and VAD filtering.

### FR-004 — LLM answer (implemented)

- The assistant shall send the system prompt plus conversation context to Ollama (`/api/chat`) and stream a response.
- Answers shall be 1–3 short sentences, plain simple English, no markdown/emojis.

### FR-005 — Local time queries (implemented)

- Time, date and day questions shall be answered locally from the PC clock without invoking the LLM.

### FR-006 — Reminders (implemented)

- The user shall be able to add tasks / reminders with times, list them, mark them done, delete them, clear all, and snooze.
- Reminders shall be persisted in `data/tasks.json` and announced when due.

### FR-007 — Read-only laptop access (implemented)

- The assistant shall only read approved folders, list approved folders, open approved websites and launch approved apps.
- Every action shall require a fresh single-use approval ("yes").

### FR-008 — Voice control commands (implemented)

- `:voice male | :voice female`, `:help`, `exit`, `stop`, `stop listening`, `exit alex` shall be handled locally.

### FR-009 — Web API (implemented, partially consumed by UI)

- The backend shall expose health, state, time, tasks, chat and transcribe endpoints on `127.0.0.1:8081`.

## 5. Non-functional requirements

### Privacy (implemented intent, verified partially)

- Processing is intended to be fully local; no cloud LLM is used.
- Voice conversation history is not written to disk.
- The audit log stores paths and decisions, never file contents.

### Security (implemented, with known limits)

- Deny-by-default access control; drive C: blocked; blocked credential-like names; link traversal protection; text-file-only reads with size cap.
- Single-use, expiring approvals; explicit `yes` required.

### Reliability

- Ollama connection/timeout errors are handled and reported.
- The reminder scheduler never exits on error.

### Performance

- Small local LLM (`qwen2.5:1.5b`), bounded context (`num_ctx 2048`, `num_predict 160`, 1500-char context budget).

## 6. User workflows and expected behavior

1. User starts Alex (`python Chatapp.py`) or the web UI (`python Chatapp.py --web`).
2. Alex asks for microphone consent (voice mode only).
3. Alex waits for wake word; on "Alex"/"Alexa" it speaks a greeting and enables continuous listening.
4. User asks a question; Alex transcribes, answers (local handlers first, then LLM), speaks the answer.
5. User can stop Alex with stop commands or Ctrl+C.
6. Reminders fire and Alex speaks them when idle.
7. Laptop access requests ask for approval; only "yes" proceeds.

## 7. Hardware and software requirements

- Windows 10/11 (uses `os.startfile`, sounddevice WASAPI).
- Python 3.11.
- Microphone and speaker hardware.
- Local Ollama installation with `qwen2.5:1.5b`.
- Downloaded Vosk and Kokoro model files.

## 8. Installation and configuration requirements

- `pip install -r requirements-web.txt`
- `ollama serve` and `ollama pull qwen2.5:1.5b`
- Vosk model under `models/vosk-model-small-en-us-0.15`
- Kokoro files under `models/kokoro`
- `security_config.json` allowlists

## 9. Known limitations and unimplemented features

- No persistent conversation memory (in-memory only; voice resets per run, web per conversation).
- STT is English-only (`base.en`).
- No web chat input is currently rendered in `index.html` (backend endpoint exists but is not consumed by the UI).
- No email/chat reading capability (explicitly refused).
- No write / delete / install operations possible by design.
- No authentication on the local web API beyond origin checks.
- Single-user, single-device design.

## 10. Future enhancements (separate from implemented capabilities)

- Persistent long-term memory.
- Additional wake words / languages.
- GPU acceleration.
- Multi-conversation persistence.
- Broader app / website approval support.

## 11. Acceptance criteria

- AC-1: Saying the wake word makes Alex speak the greeting and listen continuously.
- AC-2: A spoken question is transcribed and a spoken reply is produced.
- AC-3: Time/date/day questions are answered correctly.
- AC-4: A due reminder is announced.
- AC-5: A laptop access request requires an approving "yes" before execution.

See `docs/SystemDesign.md` for how these criteria map to implementation details.
