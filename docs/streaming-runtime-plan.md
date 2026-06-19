# Streaming Runtime Plan

This document records the next implementation plan after the offline single-turn pipeline and persistent non-streaming runtime have been validated.

## Current Baseline

The current stable baseline is:

```text
input wav/text -> ASR -> LLM -> TTS -> output wav
```

`auralis_lab/runtime.py` keeps ASR, LLM, and TTS available in one process and reduces repeated interaction latency. Keep it as the non-streaming baseline.

The next runtime should be implemented in a new file, for example:

```text
auralis_lab/streaming_runtime.py
```

Do not replace the current `runtime.py` until the streaming path is stable.

## Target Interaction Shape

The final interactive system should start by helping the user select audio devices:

```text
Auralis needs to inspect available microphone and speaker devices.

Available microphone devices:
[0] ...
[1] ...

Available speaker devices:
[3] ...
[4] ...

Select microphone device id:
Select speaker device id:
```

After that, the selected microphone is used for live audio capture, and the selected speaker is used for TTS playback.

## Audio Input Policy

Auralis uses a 16 kHz mono ASR input flow.

For early streaming validation:

```text
microphone input, mono or multichannel
-> take channel 0
-> resample to 16 kHz
-> VAD
-> ASR
```

This is intentional. It lets the project validate true live interaction using normal headsets or built-in microphones before adding microphone-array frontend algorithms.

For the later microphone-array target:

```text
raw multichannel mic-array audio
-> custom frontend algorithm
-> enhanced 16 kHz mono audio
-> VAD
-> ASR
```

The code should therefore keep an explicit audio frontend layer. The first version can be a `FirstChannelFrontend`; the later version can be replaced by a mic-array enhancement frontend.

## Resampling Rules

### Before ASR

Always normalize audio to:

```text
16 kHz, mono
```

Reasons:

- The Auralis project has standardized the ASR/audio frontend flow on 16 kHz.
- Current ASR candidates are designed for 16 kHz speech input.
- Real microphone devices may expose 16 kHz, 44.1 kHz, 48 kHz, or other sample rates.
- A consistent internal sample rate makes VAD and ASR integration simpler.

### Before Speaker Playback

TTS does not need to be forced to 16 kHz.

Recommended playback policy:

```text
TTS output at model-native sample rate
-> playback adapter
-> if speaker stream supports it, play directly
-> otherwise resample to the speaker stream sample rate, commonly 48 kHz
```

ASR input and TTS playback have different requirements:

- ASR input should be fixed at 16 kHz.
- TTS output can remain model-native unless playback requires conversion.

## VAD Requirement

A VAD module is needed for a practical streaming voice assistant.

VAD decides:

- when the user starts speaking
- when the user has finished an utterance
- when to send accumulated audio to ASR
- how to avoid sending continuous background noise to ASR

Candidate VAD engines:

- `fsmn-vad` from FunASR
- `silero-vad`
- `webrtcvad`

Initial recommendation:

Use either FunASR `fsmn-vad` or `silero-vad` first. Keep the VAD module independent from ASR so that the ASR default can remain `sherpa_onnx + SenseVoice` while VAD can be chosen separately.

## Runtime Architecture

The first streaming runtime should use threads and queues, not multiple processes.

Recommended architecture:

```text
Main Thread
  - enumerate devices
  - ask the user to select mic and speaker
  - start and stop workers
  - display runtime status

Audio Capture Thread
  - read microphone frames continuously
  - take channel 0 if input is multichannel
  - resample to 16 kHz mono
  - push audio frames to audio_frame_queue

VAD / Utterance Thread
  - read short frames from audio_frame_queue
  - detect speech start and speech end
  - accumulate one complete utterance
  - push utterance audio to utterance_queue

ASR + LLM + TTS Worker
  - read utterances from utterance_queue
  - run ASR
  - run LLM
  - run TTS
  - push synthesized audio to playback_queue

Playback Thread
  - read audio from playback_queue
  - resample if needed for the selected speaker
  - play the response audio
```

This design keeps each responsibility separate while avoiding the complexity and GPU memory overhead of multiple processes.

## Why Threads First

Threads are enough for the first streaming MVP because:

- audio capture and playback are I/O-oriented
- queues are simple and reliable for passing audio blocks between stages
- ASR, LLM, and TTS can remain blocking calls inside a worker
- model loading stays in one process

Consider multiple processes later only if:

- a model library is unstable and should be isolated
- ASR and TTS need independent GPU scheduling
- the project becomes a service architecture with separate ASR, LLM, and TTS services

## Streaming Levels

Do not jump directly to token-level streaming.

Recommended first streaming MVP:

```text
streaming mic capture
-> VAD utterance segmentation
-> non-streaming ASR
-> non-streaming LLM
-> non-streaming TTS
-> speaker playback
```

This already gives a live voice assistant experience: the user speaks, VAD detects the end of speech, then the system responds.

Later advanced streaming:

```text
streaming ASR partial results
-> streaming LLM tokens
-> streaming or chunked TTS
-> playback while generating
```

## Implementation Plan

### Step 1: Device Enumeration and Manual Selection

Goal:

```text
list mic devices
list speaker devices
select mic id
select speaker id
```

Validation:

- selected microphone can record 3 seconds of audio
- recording is saved as 16 kHz mono wav
- selected speaker can play an existing wav file

Suggested library:

```text
sounddevice
```

### Step 2: Live Capture Base

Goal:

```text
selected mic -> continuous audio frames -> channel 0 -> 16 kHz mono
```

Validation:

- capture can run until Ctrl+C
- saved audio is clean enough for ASR
- no ASR, LLM, or TTS involved yet

### Step 3: WebSocket Connectivity Test

Goal:

```text
Windows client -> WebSocket -> Linux server
ping -> pong
```

Minimal server command:

```bash
cd /home/xiezc/Auralis
python -m pip install -r requirements/realtime.txt
python auralis_lab/ws_ping_server.py --host 0.0.0.0 --port 8765
```

The server must bind to `0.0.0.0`, not `127.0.0.1`, so that the Windows client can connect over the LAN.

### Step 4: VAD Utterance Segmentation

Before VAD, validate that the Windows client can upload a complete 16 kHz mono wav to the server:

```powershell
python -m auralis_client.upload_wav_client --wav outputs/capture-stream-test.wav --server-url ws://192.168.16.206:8765
```

The temporary server saves uploaded audio under:

```text
outputs/ws_uploads/
```

This upload prototype may allow large WebSocket messages for validation convenience. The final streaming protocol should send chunked audio frames instead of a long wav as one message.

The temporary server can also simulate TTS audio return:

```bash
python auralis_lab/ws_ping_server.py --host 0.0.0.0 --port 8765 --reply-wav outputs/cosyvoice-sft.wav
```

For the first real offline ASR/LLM/TTS turn, use:

```bash
cd /home/xiezc/Auralis
export PYTHONPATH=/home/xiezc/Auralis/third_party/CosyVoice:$PYTHONPATH
python auralis_lab/ws_pipeline_server.py --host 0.0.0.0 --port 8765
```

This WebSocket pipeline server keeps ASR and TTS loaded persistently after startup. This avoids reloading sherpa-onnx and CosyVoice on every client turn.

Goal:

```text
live 16 kHz mono audio -> VAD -> complete utterance wav
```

Validation:

- each detected utterance is saved under `outputs/utterances/`
- speech start/end are stable enough in a normal room
- silence and background noise do not trigger too many false utterances

### Step 5: Live ASR

Goal:

```text
mic -> VAD -> ASR_TEXT
```

Validation:

- every detected utterance is transcribed
- ASR text is printed in the terminal
- default ASR can start with `sherpa_onnx + SenseVoice`

### Step 6: Live ASR + LLM

Goal:

```text
mic -> VAD -> ASR -> LLM_TEXT
```

Validation:

- LLM reply is concise
- no `<think>` content appears
- response is suitable for speech synthesis

### Step 7: Full Live Voice Loop

Goal:

```text
mic -> VAD -> ASR -> LLM -> TTS -> speaker
```

Validation:

- user can speak one sentence and hear a response
- selected speaker plays the response
- latency is measured per stage
- system can handle multiple turns

### Step 8: Playback and Barge-In Policy

Questions to resolve:

- Should microphone capture pause during TTS playback?
- Should VAD ignore audio while the system is speaking?
- Should the user be allowed to interrupt TTS playback?
- Is echo cancellation required?

Initial recommendation:

For the first version, pause or ignore user input during TTS playback. Add barge-in later.

### Step 9: Performance Optimization

Optimize after the full live loop works.

Potential directions:

- keep all models warm
- shorten LLM replies
- use faster TTS defaults when needed
- record stage latency per turn
- test VAD thresholds
- later evaluate streaming ASR and streaming TTS

## Recommended Next Action

Start with audio I/O only:

```text
device enumeration
-> mic selection
-> speaker selection
-> record 3 seconds as 16 kHz mono wav
-> play a wav through selected speaker
```

This isolates the most important new dependency: real hardware audio I/O. Once this is stable, add VAD.
