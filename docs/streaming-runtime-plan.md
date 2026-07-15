# Streaming Runtime Plan

This document records the next implementation plan after the offline single-turn pipeline and persistent non-streaming runtime have been validated.

## Validation Status: Milestone 02 Complete

As of 2026-07-14, the first live streaming, half-duplex interaction loop has been validated across the Windows client and Linux server:

```text
Windows mic
-> channel 0 -> continuous 16 kHz mono PCM16 WebSocket frames
-> Silero VAD endpointing
-> sherpa-onnx SenseVoice ASR
-> Ollama qwen3:8b with four-turn history
-> persistent CosyVoice SFT
-> reply WAV WebSocket transfer
-> Windows speaker playback
```

Validated details:

- 48 kHz Windows microphone capture is resampled continuously to 16 kHz before VAD and ASR.
- Silero VAD was selected as the current default after comparison with FunASR FSMN-VAD and WebRTC VAD.
- Noise-like ASR results are filtered before reaching the LLM.
- The client and server exchange `turn_started`, `utterance_saved`, `asr_result`, `llm_result`, `tts_result`, and `reply_audio` events for each valid turn.
- The tested CosyVoice SFT synthesis time was approximately 4.8 to 4.9 seconds per short reply; ASR and LLM were both below one second in the observed turns.
- The previously validated loop used a logical half-duplex policy. The HP21 single-WASAPI-Stream hardware test has now passed; the client integration keeps that Stream open and pauses only upstream frame delivery during a reply. Its server-backed regression test is the next validation step.

This is live frame streaming with utterance-level ASR, LLM, and TTS. It is not yet token-level or audio-chunk-level streaming.

## Current Baselines

The retained offline regression baseline is:

```text
input wav/text -> ASR -> LLM -> TTS -> output wav
```

`auralis_lab/runtime.py` keeps ASR, LLM, and TTS available in one process and reduces repeated interaction latency. Keep it for non-streaming regression tests.

The validated live runtime is:

```text
auralis_lab/ws_stream_tts_server.py
```

It runs the server-side VAD, ASR, LLM, and TTS stages while `auralis_client/stream_upload_client.py` handles Windows capture and playback. Do not replace `runtime.py`; the two scripts validate different interaction modes.

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

Current selection:

Use `silero-vad` as the current default. It had the best overall endpointing behavior in the local comparison; WebRTC VAD produced more false positives, while FunASR FSMN-VAD produced occasional false triggers. Keep the VAD module independent from ASR so that the ASR default remains `sherpa_onnx + SenseVoice` and the VAD can still be switched for later regression tests.

## Runtime Architecture

### Current Cross-Machine Implementation

The validated deployment is split across the two projects:

```text
Windows AuralisClient
  sounddevice capture callback -> 16 kHz PCM16 frames -> WebSocket

Linux Auralis server
  asyncio WebSocket handler -> VAD -> ASR -> LLM -> persistent TTS -> reply WAV

Windows AuralisClient
  receive reply WAV -> main-thread WASAPI playback -> reopen capture
```

The current server processes one endpointed utterance at a time per connection. This is intentional for the half-duplex MVP.

### Future In-Process Architecture

For a future single-machine runtime, use threads and queues before considering multiple processes.

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

### Step 4: Streaming Frame Upload Test

Goal:

```text
Windows client mic
-> channel 0
-> 16 kHz mono PCM16 frames
-> WebSocket binary messages
-> Linux server reconstructs wav
```

Server command:

```bash
cd /home/xiezc/Auralis
python auralis_lab/ws_stream_record_server.py --host 0.0.0.0 --port 8766
```

Client command:

```powershell
python -m auralis_client.stream_upload_client --input-device 26 --server-url ws://192.168.16.206:8766 --seconds 10 --frame-ms 100 --blocksize-frames 0
```

### Step 5: Streaming VAD Endpoint Test

Goal:

```text
continuous PCM16 stream
-> VAD speech start/end detection
-> save each detected utterance as wav
```

Server command with the lightweight energy fallback:

```bash
python auralis_lab/ws_stream_vad_server.py --host 0.0.0.0 --port 8767 --vad-engine energy
```

Server command with FunASR FSMN-VAD selected:

```bash
python auralis_lab/ws_stream_vad_server.py --host 0.0.0.0 --port 8767 --vad-engine funasr_fsmn
```

Server command with Silero VAD selected:

```bash
python auralis_lab/ws_stream_vad_server.py --host 0.0.0.0 --port 8767 --vad-engine silero
```

Server command with WebRTC VAD selected:

```bash
python auralis_lab/ws_stream_vad_server.py --host 0.0.0.0 --port 8767 --vad-engine webrtc --webrtc-aggressiveness 2
```

VAD model files should be stored under the project `models/vad` directory:

```text
models/vad/funasr-fsmn-vad
models/vad/silero-vad
```

Download FunASR FSMN-VAD:

```bash
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir models/vad/funasr-fsmn-vad
```

Download Silero VAD:

```bash
git clone https://github.com/snakers4/silero-vad models/vad/silero-vad
```

WebRTC VAD has no model file. Install the Python dependency:

```bash
python -m pip install webrtcvad
```

Client command:

```powershell
python -m auralis_client.stream_upload_client --input-device 26 --server-url ws://192.168.16.206:8767 --seconds 30 --frame-ms 100 --blocksize-frames 0
```

Detected utterance wav files are saved under:

```text
outputs/ws_stream_utterances/
```

Tune these values first when endpointing is too eager or too slow:

```text
--energy-threshold
--speech-start-ms
--speech-end-ms
--pre-speech-ms
--min-utterance-ms
```

Minimal server command:

```bash
cd /home/xiezc/Auralis
python -m pip install -r requirements/realtime.txt
python auralis_lab/ws_ping_server.py --host 0.0.0.0 --port 8765
```

The server must bind to `0.0.0.0`, not `127.0.0.1`, so that the Windows client can connect over the LAN.

### Legacy Offline WebSocket Validation

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

Historical next full streaming goal, now validated by Milestone 02:

```text
live 16 kHz mono audio -> VAD -> complete utterance wav
```

Validation:

- each detected utterance is saved under `outputs/utterances/`
- speech start/end are stable enough in a normal room
- silence and background noise do not trigger too many false utterances

### Step 6: Streaming VAD + ASR

Goal:

```text
continuous PCM16 stream
-> VAD speech start/end detection
-> save utterance wav
-> ASR transcription
-> send ASR text event back to client
```

Server command:

```bash
python auralis_lab/ws_stream_asr_server.py --host 0.0.0.0 --port 8768 --vad-engine silero --asr-engine sherpa_onnx
```

Alternative VAD engines:

```bash
python auralis_lab/ws_stream_asr_server.py --host 0.0.0.0 --port 8768 --vad-engine webrtc --asr-engine sherpa_onnx
python auralis_lab/ws_stream_asr_server.py --host 0.0.0.0 --port 8768 --vad-engine funasr_fsmn --asr-engine sherpa_onnx
```

Client command:

```powershell
python -m auralis_client.stream_upload_client --input-device 26 --server-url ws://192.168.16.206:8768 --seconds 60 --frame-ms 100 --blocksize-frames 0
```

The client prints `utterance_saved`, `asr_result`, and `asr_filtered` server events. `asr_filtered` means the utterance wav was saved, but the ASR text was considered noise-like and should not enter later LLM/TTS stages.

Useful ASR post-filter parameters:

```text
--min-asr-text-chars
--min-asr-duration-seconds
--asr-noise-phrases
--keep-empty-asr
```

### Step 7: Streaming VAD + ASR + LLM

Goal:

```text
continuous PCM16 stream
-> VAD utterance segmentation
-> ASR text and post-filtering
-> Ollama / Qwen3 reply
-> send LLM text event back to client
```

Server command:

```bash
python auralis_lab/ws_stream_llm_server.py --host 0.0.0.0 --port 8769 --vad-engine silero --asr-engine sherpa_onnx --llm-model qwen3:8b
```

Client command:

```powershell
python -m auralis_client.stream_upload_client --input-device 26 --server-url ws://192.168.16.206:8769 --seconds 90 --frame-ms 100 --blocksize-frames 0 --timeout 180
```

Each WebSocket connection keeps the most recent four user/assistant turns by default. Set `--max-history-turns 0` for independent single-turn replies.

Validation:

- LLM reply is concise
- no `<think>` content appears
- `asr_filtered` never produces `llm_result`
- `llm_result` includes the response latency and history size
- without an external data source, LLM replies do not claim real-time weather, traffic, price, news, or similar facts

### Step 8: Streaming VAD + ASR + LLM + TTS

Goal:

```text
continuous PCM16 stream
-> VAD utterance segmentation
-> ASR and LLM
-> persistent CosyVoice synthesis
-> reply WAV over WebSocket
-> client speaker playback
```

Server command:

```bash
cd /home/xiezc/Auralis
export PYTHONPATH=/home/xiezc/Auralis/third_party/CosyVoice:$PYTHONPATH
CUDA_VISIBLE_DEVICES=7 python auralis_lab/ws_stream_tts_server.py --host 0.0.0.0 --port 8770 --vad-engine silero --asr-engine sherpa_onnx --llm-model qwen3:8b --tts-engine cosyvoice --cosy-mode sft
```

Client command:

```powershell
python -m auralis_client.stream_upload_client --input-device 26 --output-device 23 --server-url ws://192.168.16.206:8770 --seconds 90 --frame-ms 100 --blocksize-frames 0 --timeout 300
```

The server emits `turn_started`, `utterance_saved`, `asr_result`, `llm_result`, `tts_result`, and `reply_audio` events. The client saves received reply WAV files under `outputs/stream_replies/`. When `--output-device` is set, it pauses upstream microphone frames as soon as `turn_started` arrives and resumes them after playback; this prevents feedback and avoids queuing microphone frames while the server runs LLM/TTS. On a WASAPI input/output pair, `--audio-mode auto` keeps one duplex Stream open throughout the session. Omit `CUDA_VISIBLE_DEVICES=7` when another GPU should run CosyVoice.

Validation:

- user can speak one sentence and hear a response
- selected speaker plays the response
- latency is measured per stage
- system can handle multiple turns

### Step 9: Playback and Barge-In Policy

Questions to resolve:

- Should microphone capture pause during TTS playback?
- Should VAD ignore audio while the system is speaking?
- Should the user be allowed to interrupt TTS playback?
- Is echo cancellation required?

Implemented baseline:

The client closes capture after `turn_started`, ignores input during reply generation and playback, and reopens capture after playback. Add barge-in later.

### Step 10: Performance Optimization

Optimize after the full live loop works.

Potential directions:

- keep all models warm
- shorten LLM replies
- use faster TTS defaults when needed
- record stage latency per turn
- test VAD thresholds
- later evaluate streaming ASR and streaming TTS

## Recommended Next Action

Milestone 02 now has a working utterance-level live loop. Optimize and extend from this stable baseline in this order:

1. Record per-turn latency from VAD endpoint through playback start, then reduce TTS response time and constrain reply length.
2. Add explicit playback state and a user-visible stop command, while retaining the current half-duplex policy.
3. Introduce streaming ASR partial results, then chunked/streaming TTS playback only after the endpointed baseline remains stable.
4. Add echo cancellation and barge-in only when the product requires open-speaker interaction.
