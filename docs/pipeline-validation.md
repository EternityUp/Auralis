# Pipeline Validation

This document tracks Auralis integration tests after ASR, LLM, and TTS have been validated independently.

## Step 1: Text Pipeline

Goal:

```text
manual text -> LLM -> TTS -> reply wav
```

This validates whether the LLM output is suitable for direct speech synthesis before ASR and microphone input are added.

Default stack:

- LLM: `ollama` + `qwen3:8b`
- TTS: `cosyvoice` SFT + `CosyVoice-300M-SFT`

Run:

```bash
cd /home/xiezc/Auralis
python auralis_lab/pipeline.py --input-type text --text "请用一句话介绍你自己。" --output outputs/text-pipeline-reply.wav
```

Expected terminal output:

```text
USER_TEXT:
...
LLM_TEXT:
...
TTS_OUTPUT: outputs/text-pipeline-reply.wav
LATENCY:
  llm_seconds: ...
  tts_seconds: ...
  total_seconds: ...
```

The `LLM_TEXT` label is for debugging only. The text passed to TTS is the raw LLM reply, not the label.

## Useful Variants

Use Piper:

```bash
python auralis_lab/pipeline.py --input-type text --tts-engine piper --text "请用一句话介绍你自己。" --output outputs/text-pipeline-piper.wav
```

Use VoxCPM-0.5B:

```bash
python auralis_lab/pipeline.py --input-type text --tts-engine voxcpm --voxcpm-variant 0.5b --text "请用一句话介绍你自己。" --output outputs/text-pipeline-voxcpm.wav
```

Use CosyVoice2 instruct2:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/pipeline.py --input-type text --tts-engine cosyvoice --cosy-mode instruct2 --tts-model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/boy.wav --instruct-text "用自然流畅的普通话朗读。<|endofprompt|>" --text "请用一句话介绍你自己。" --output outputs/text-pipeline-cosyvoice2-instruct.wav
```

## Pass Criteria

- LLM returns concise Chinese text without `<think>` content.
- TTS produces a playable wav file.
- The reply is natural enough for voice interaction.
- End-to-end latency is acceptable for a first MVP.

## Step 2: Audio Pipeline

Goal:

```text
input wav -> ASR -> LLM -> TTS -> reply wav
```

Default stack:

- ASR: `sherpa_onnx` + SenseVoice ONNX
- LLM: `ollama` + `qwen3:8b`
- TTS: `cosyvoice` SFT + `CosyVoice-300M-SFT`

Run:

```bash
cd /home/xiezc/Auralis
python auralis_lab/pipeline.py --input-type audio --audio samples/asr/boy.wav --output outputs/audio-pipeline-reply.wav

python auralis_lab/pipeline.py --input-type audio --audio samples/asr/girl.wav --output outputs/audio-pipeline-reply.wav
```

If the default sherpa-onnx path is not suitable for the input, use FunASR + punctuation:

```bash
python auralis_lab/pipeline.py --input-type audio --asr-engine funasr --audio samples/asr/boy.wav --output outputs/audio-pipeline-funasr-reply.wav

python auralis_lab/pipeline.py --input-type audio --asr-engine funasr --audio samples/asr/girl.wav --output outputs/audio-pipeline-funasr-reply.wav
```

Expected terminal output:

```text
PIPELINE_MODE: audio
AUDIO_INPUT:
...
ASR_TEXT:
...
LLM_TEXT:
...
TTS_OUTPUT: outputs/audio-pipeline-reply.wav
LATENCY:
  asr_seconds: ...
  llm_seconds: ...
  tts_seconds: ...
  total_seconds: ...
```

Pass criteria:

- ASR text is understandable.
- LLM answer is concise and suitable for speech.
- TTS output is playable.
- The total latency is acceptable for an offline first MVP.

The older `text_pipeline.py` and `audio_pipeline.py` scripts are kept as focused validation helpers. Use `pipeline.py` as the main single-turn entry point going forward.

## Default Stack Candidates

Use `qwen3:8b` as the fixed LLM during this round. Compare ASR and TTS combinations with the same input audio.

### Candidate A: Stability First

```text
sherpa_onnx SenseVoice -> qwen3:8b -> CosyVoice SFT
```

Run:

```bash
python auralis_lab/pipeline.py --input-type audio --asr-engine sherpa_onnx --tts-engine cosyvoice --cosy-mode sft --audio samples/asr/boy.wav --output outputs/pipeline-a-sherpa-cosy-sft.wav
```

Why:

- Best first default if stability and deployment simplicity matter.
- No prompt audio is required for TTS.
- Good baseline for latency measurement.

### Candidate B: Mandarin ASR Readability First

```text
FunASR Paraformer + punctuation -> qwen3:8b -> CosyVoice SFT
```

Run:

```bash
python auralis_lab/pipeline.py --input-type audio --asr-engine funasr --tts-engine cosyvoice --cosy-mode sft --audio samples/asr/boy.wav --output outputs/pipeline-b-funasr-cosy-sft.wav
```

Why:

- FunASR + punctuation is strong for Mandarin text readability.
- Useful when ASR punctuation affects the LLM response quality.

### Candidate C: CosyVoice Prompt-Audio Direction

```text
sherpa_onnx SenseVoice -> qwen3:8b -> CosyVoice2 zero-shot or instruct2
```

Run zero-shot:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/pipeline.py --input-type audio --asr-engine sherpa_onnx --tts-engine cosyvoice --cosy-mode zero_shot --tts-model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/girl.wav --prompt-text "总理对任何事情都要刨根问底。" --audio samples/asr/boy.wav --output outputs/pipeline-c-sherpa-cosy2-zero-shot.wav
```

Run instruct2:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/pipeline.py --input-type audio --asr-engine sherpa_onnx --tts-engine cosyvoice --cosy-mode instruct2 --tts-model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/boy.wav --instruct-text "用自然流畅的普通话朗读。<|endofprompt|>" --audio samples/asr/boy.wav --output outputs/pipeline-c-sherpa-cosy2-instruct2.wav
```

Why:

- Closer to the final desired prompt-audio-controlled voice.
- Heavier than SFT, so compare both quality and latency before choosing it as default.

### Candidate D: VoxCPM Alternative

```text
sherpa_onnx SenseVoice -> qwen3:8b -> VoxCPM
```

Run VoxCPM-0.5B:

```bash
python auralis_lab/pipeline.py --input-type audio --asr-engine sherpa_onnx --tts-engine voxcpm --voxcpm-variant 0.5b --audio samples/asr/boy.wav --output outputs/pipeline-d-sherpa-voxcpm-0.5b.wav
```

Run VoxCPM2:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/pipeline.py --input-type audio --asr-engine sherpa_onnx --tts-engine voxcpm --voxcpm-variant voxcpm2 --audio samples/asr/boy.wav --output outputs/pipeline-d-sherpa-voxcpm2.wav
```

Why:

- Useful TTS alternative to CosyVoice.
- Strong candidate if voice quality is better or setup is simpler in your environment.

## Latency Notes

For each candidate, record:

- `asr_seconds`
- `llm_seconds`
- `tts_seconds`
- `total_seconds`
- output audio quality
- whether ASR text was good enough for LLM

For the current offline MVP, compare total latency first. Later, interactive UX will also need first-token latency and streaming playback latency.

## Step 3: Persistent Runtime

`pipeline.py` is a single-turn validation script. It starts a new Python process for each test, so ASR and TTS model loading can distort latency observations.

Use `runtime.py` when you want to keep the pipeline alive and run multiple turns in one process:

```bash
cd /home/xiezc/Auralis
export PYTHONPATH=/home/xiezc/Auralis/third_party/CosyVoice:$PYTHONPATH
python auralis_lab/runtime.py
```

Default runtime stack:

```text
sherpa_onnx SenseVoice -> qwen3:8b -> CosyVoice SFT
```

Interactive commands:

```text
text 请用一句话介绍你自己。
audio samples/asr/boy.wav
quit
```

Runtime output is written to `outputs/runtime/runtime-reply-001.wav`, `outputs/runtime/runtime-reply-002.wav`, and so on.

Use a less busy GPU when needed:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/runtime.py
```

Use CosyVoice2 instruct2 in the runtime:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/runtime.py \
  --tts-model models/tts/cosyvoice/CosyVoice2-0.5B \
  --cosy-mode instruct2 \
  --prompt-audio samples/tts/boy.wav \
  --instruct-text "用自然流畅的普通话朗读。<|endofprompt|>"
```

For now, this runtime is still offline/non-streaming per turn: it waits for ASR, then LLM, then TTS. Its value is that model initialization happens once, which makes repeated interaction closer to the final application shape.
