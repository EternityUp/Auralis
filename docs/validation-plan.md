# Validation Plan

## Goal

Validate ASR, TTS, and LLM independently before building the full voice interaction pipeline.

The whole Auralis audio path is standardized on 16 kHz. All ASR test samples and microphone frontend outputs should use 16 kHz mono audio.

## Step 1: ASR

Inputs:

- Clean Mandarin test wav, 16 kHz mono if possible.
- One noisy or far-field wav after the clean sample works.

Metrics:

- Can run locally.
- Recognized text is usable.
- Latency is acceptable.
- GPU memory and CPU usage are understood.

Order:

1. `faster-whisper small`
2. `faster-whisper medium` or `large-v3`
3. `funasr paraformer-zh` with punctuation model
4. `sensevoice small`
5. `sherpa-onnx` SenseVoice or Paraformer ONNX model

Use local model directories for server experiments:

```powershell
python -m auralis_lab.asr --engine faster_whisper --model models/asr/faster-whisper-small --audio samples/asr/test.wav
python -m auralis_lab.asr --engine faster_whisper --model models/asr/faster-whisper-medium --audio samples/asr/test.wav
python -m auralis_lab.asr --engine faster_whisper --model models/asr/faster-whisper-large-v3 --audio samples/asr/test.wav
python -m auralis_lab.asr --engine funasr --model models/asr/funasr-paraformer-zh --punc-model models/asr/funasr-ct-punc --audio samples/asr/test.wav
python -m auralis_lab.asr --engine sensevoice --model models/asr/sensevoice-small --audio samples/asr/test.wav
python -m auralis_lab.asr --engine sherpa_onnx --sherpa-model-type sensevoice --model models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 --audio samples/asr/test.wav
```

## Step 2: TTS

Inputs:

- Short Chinese sentence.
- One longer answer with punctuation.

Metrics:

- Can generate an audio file.
- Audio is intelligible and pleasant enough.
- Latency is acceptable.
- Local/offline feasibility is clear.

Order:

1. `edge-tts` for fastest path validation
2. `piper` for local lightweight validation
3. `CosyVoice` for higher-quality Mandarin synthesis
4. `VoxCPM` for another high-quality local TTS and voice-cloning candidate
5. `ChatTTS` in a separate environment if still needed

Use the same text for every engine:

```bash
python auralis_lab/tts.py --engine edge_tts --text "你好，我是 Auralis。" --output outputs/edge-tts.wav
python auralis_lab/tts.py --engine piper --model models/tts/piper/zh_CN-huayan-medium.onnx --text "你好，我是 Auralis。" --output outputs/piper.wav
python auralis_lab/tts.py --engine cosyvoice --cosy-mode sft --model models/tts/cosyvoice/CosyVoice-300M-SFT --speaker "中文女" --text "你好，我是 Auralis。" --output outputs/cosyvoice-sft.wav
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode zero_shot --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/prompt.wav --prompt-text "这段参考音频对应的文字。" --text "你好，我是 Auralis。" --output outputs/cosyvoice-zero-shot.wav
python auralis_lab/tts.py --engine voxcpm --voxcpm-variant 0.5b --model models/tts/voxcpm/VoxCPM-0.5B --text "你好，我是 Auralis。" --output outputs/voxcpm-0.5b.wav
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine voxcpm --voxcpm-variant voxcpm2 --model models/tts/voxcpm/VoxCPM2 --text "你好，我是 Auralis。" --output outputs/voxcpm2.wav
```

ChatTTS was deprioritized because its dependency changes disturbed the validated CosyVoice2 environment during our tests.

## Step 3: LLM

Inputs:

- One factual Chinese prompt.
- One instruction-following prompt.
- One dialogue-style prompt.

Metrics:

- Can run locally.
- Chinese response quality is acceptable.
- Latency is acceptable.
- Context window and memory usage are understood.

Order:

1. Ollama + Qwen3-8B
2. Smaller model if latency is too high
3. Larger model if quality is insufficient and GPU memory allows

First command:

```bash
ollama pull qwen3:8b
python auralis_lab/llm.py --engine ollama --model qwen3:8b --thinking off --prompt "请用一句话介绍你自己。"
```

## Step 4: Text Pipeline MVP

Before adding ASR back into the loop, validate the text-only integration path:

```text
manual text -> LLM -> TTS -> reply wav
```

Run:

```bash
python auralis_lab/pipeline.py --input-type text --text "请用一句话介绍你自己。" --output outputs/text-pipeline-reply.wav
```

## Step 5: Audio Pipeline MVP

Validate the first offline audio turn:

```text
input wav -> ASR -> LLM -> TTS -> reply wav
```

Run:

```bash
python auralis_lab/pipeline.py --input-type audio --audio samples/asr/boy.wav --output outputs/audio-pipeline-reply.wav
```
