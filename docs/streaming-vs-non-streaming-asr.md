# Streaming vs Non-Streaming ASR

This document records the ASR architecture decision for Auralis.

## Conclusion

Auralis should eventually become a streaming voice interaction system.

However, the recommended engineering path is:

```text
non-streaming MVP -> semi-streaming interaction -> streaming ASR -> full streaming voice assistant
```

The current non-streaming ASR validation work is still necessary. It gives Auralis a stable quality and latency baseline before introducing real-time complexity.

## Non-Streaming ASR

Non-streaming ASR waits until a full utterance or full audio file is available, then performs recognition once.

Typical flow:

```text
user finishes speaking
recording ends
full audio is sent to ASR
ASR returns complete text
```

Current examples in Auralis:

- `faster-whisper`
- `FunASR Paraformer`
- `SenseVoiceSmall`
- `sherpa-onnx` offline recognition

Characteristics:

- Simpler to implement.
- Usually has better accuracy because the model can use full audio context.
- Easier to generate more stable sentence boundaries and punctuation.
- Higher interaction latency because recognition starts after recording ends.
- Good for validating model quality, deployment, and audio format assumptions.

Good fit for:

- Audio file transcription.
- Meeting notes.
- Push-to-talk voice commands.
- Short single-turn voice QA.
- First MVP versions.
- Offline batch processing.

Example interaction:

```text
press to record -> release -> ASR -> LLM -> TTS -> playback
```

## Streaming ASR

Streaming ASR processes audio incrementally while the user is still speaking.

Typical flow:

```text
audio chunk arrives
ASR updates partial text
more chunks arrive
ASR updates partial text again
speech endpoint is detected
ASR emits final text
```

Example output:

```text
partial: 今
partial: 今天
partial: 今天天气
partial: 今天天气怎么样
final: 今天天气怎么样？
```

The key point is that true streaming ASR does not require the full audio to be available. The model and decoder are designed for incremental inference with limited future context.

Characteristics:

- Lower first-token or first-character latency.
- Enables real-time subtitles and real-time intent detection.
- Supports more natural voice assistant experiences.
- Required for barge-in and low-latency interaction.
- Harder to implement and evaluate.
- Partial results can be unstable and may need correction.
- Accuracy, punctuation, and sentence boundaries may be weaker than offline models.

Good fit for:

- Real-time voice assistants.
- Smart speakers.
- Car voice interaction.
- Real-time meeting captions.
- Live customer service.
- Systems that need interruption or barge-in.

## What Counts As Streaming

Streaming is not just splitting audio into small files and repeatedly running an offline model.

True streaming ASR:

```text
accepts audio chunks
keeps decoding state
uses current and limited future context
emits partial and final results
```

Non-streaming ASR:

```text
needs complete audio
uses full context
emits one complete result
```

Running an offline model every second on a sliding window is only pseudo-streaming. It can be useful for experiments, but it often causes:

- repeated text
- unstable boundaries
- duplicated decoding work
- worse latency than true streaming
- harder result stitching

## Model-Level Difference

Common non-streaming model families:

- Whisper
- offline Paraformer
- offline Conformer
- SenseVoice offline mode

Common streaming model families:

- streaming Paraformer
- streaming Conformer
- U2++ Conformer streaming
- Zipformer streaming
- Transducer / RNN-T

Streaming models usually rely on mechanisms such as:

- chunk-based attention
- causal convolution
- limited right context
- incremental decoder state

FunASR streaming candidates worth testing later:

- `paraformer-zh-streaming`
- `conformer-streaming`
- `u2pp_conformer_streaming`

Sherpa-ONNX may also be useful later for streaming or deployment-oriented ASR experiments.

## Auralis Recommended Roadmap

### v0.1: Non-Streaming ASR File Validation

Goal:

```text
wav file -> ASR text
```

Purpose:

- Compare ASR model quality.
- Validate 16 kHz audio flow.
- Confirm local model loading.
- Measure latency and memory usage.

Current work belongs here.

### v0.2: Non-Streaming Single-Turn Voice Interaction

Goal:

```text
record full utterance -> ASR -> LLM -> TTS -> playback
```

This gives Auralis a complete single-turn MVP.

### v0.3: VAD-Based Semi-Streaming Interaction

Goal:

```text
microphone stream -> VAD detects speech segment -> offline ASR -> LLM -> TTS
```

ASR can still be non-streaming, but the user no longer has to manually provide an audio file.

This is a practical middle stage and should come before full streaming ASR.

### v0.4: Streaming LLM And Sentence-Level TTS

Goal:

```text
ASR final text -> LLM streaming output -> sentence-level TTS -> playback
```

This improves perceived latency even before streaming ASR is introduced.

### v0.5: Streaming ASR Experiments

Goal:

```text
audio chunks -> streaming ASR partial/final text
```

Key tasks:

- Evaluate FunASR streaming models.
- Evaluate sherpa-onnx streaming/deployment options.
- Handle partial result updates.
- Handle final result stabilization.
- Evaluate latency versus accuracy.

### v1.0: Full Streaming Voice Assistant

Goal:

```text
real-time audio input
-> streaming ASR
-> streaming LLM
-> streaming or sentence-level TTS
-> speaker playback
-> barge-in interruption
```

Additional required modules:

- VAD
- endpoint detection
- echo cancellation
- playback interruption
- dialogue state management
- microphone-array frontend

## Decision For Current Stage

Current stage should continue using non-streaming ASR.

Reason:

- It gives stable quality baselines.
- It avoids too many variables at once.
- It helps compare faster-whisper, FunASR, SenseVoice, and sherpa-onnx fairly.
- It makes later streaming ASR evaluation meaningful.

Final target should still be streaming, because Auralis is intended to be a local microphone-array voice interaction system rather than just an offline transcription tool.

