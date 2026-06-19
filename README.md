# Auralis

Auralis is a local intelligent voice interaction system for microphone-array scenarios.

The first milestone is to validate the three core model capabilities independently:

1. ASR: audio to text
2. TTS: text to speech
3. LLM: text understanding and response generation

After these parts are stable, they can be connected into a single-turn pipeline:

```text
audio -> ASR -> LLM -> TTS -> speaker
```

## Audio Standard

Auralis uses a 16 kHz audio flow end to end.

- Input audio should be converted to 16 kHz mono before ASR.
- Microphone-array frontend modules should output 16 kHz mono audio.
- TTS output may be generated at a model-native rate, but playback or downstream processing should convert it when needed.

## Environment

Recommended Python version:

```bash
conda create -n auralis python=3.10 -y
conda activate auralis
python -m pip install -U pip setuptools wheel
```

Install ASR dependencies:

```bash
python -m pip install -r requirements/asr.txt
```

Install TTS dependencies:

```bash
python -m pip install -r requirements/tts.txt
```

For GPU PyTorch on a CUDA 12.8 driver:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## ASR Validation

All ASR models should be stored under `models/asr/`.

Current supported engines:

- `faster_whisper`
- `funasr`
- `sensevoice`
- `sherpa_onnx`

Example commands:

```bash
python auralis_lab/asr.py --engine faster_whisper --model models/asr/faster-whisper-large-v3 --audio samples/asr/Tencent_test.wav
python auralis_lab/asr.py --engine funasr --model models/asr/funasr-paraformer-zh --punc-model models/asr/funasr-ct-punc --audio samples/asr/Tencent_test.wav
python auralis_lab/asr.py --engine sensevoice --model models/asr/sensevoice-small --audio samples/asr/Tencent_test.wav
python auralis_lab/asr.py --engine sherpa_onnx --sherpa-model-type sensevoice --model models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 --audio samples/asr/Tencent_test.wav
```

## TTS Validation

All local TTS models should be stored under `models/tts/`.

Current supported engines:

- `edge_tts`
- `piper`
- `cosyvoice`
- `voxcpm`
- `chattts`

Example commands:

```bash
python auralis_lab/tts.py --engine edge_tts --text "你好，我是 Auralis。" --output outputs/edge-tts.wav
python auralis_lab/tts.py --engine piper --model models/tts/piper/zh_CN-huayan-medium.onnx --text "你好，我是 Auralis。" --output outputs/piper.wav
python auralis_lab/tts.py --engine cosyvoice --cosy-mode sft --model models/tts/cosyvoice/CosyVoice-300M-SFT --speaker "中文女" --text "你好，我是 Auralis。" --output outputs/cosyvoice-sft.wav
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode zero_shot --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/prompt.wav --prompt-text "这段参考音频对应的文字。" --text "你好，我是 Auralis。" --output outputs/cosyvoice-zero-shot.wav
python auralis_lab/tts.py --engine voxcpm --voxcpm-variant 0.5b --model models/tts/voxcpm/VoxCPM-0.5B --text "你好，我是 Auralis。" --output outputs/voxcpm-0.5b.wav
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine voxcpm --voxcpm-variant voxcpm2 --model models/tts/voxcpm/VoxCPM2 --text "你好，我是 Auralis。" --output outputs/voxcpm2.wav
```

`chattts` is currently kept as an exploratory engine. In our current shared environment, the dependency changes needed to run ChatTTS disturbed the validated CosyVoice2 setup, so test it in a separate conda environment if needed.

## LLM Validation

```bash
python -m pip install -r requirements/llm.txt
ollama pull qwen3:8b
ollama serve
python auralis_lab/llm.py --engine ollama --model qwen3:8b --thinking off --prompt "请用一句话介绍你自己。"
```

## Text Pipeline MVP

After LLM and TTS are available, validate the first single-turn text pipeline:

```bash
python auralis_lab/pipeline.py --input-type text --text "请用一句话介绍你自己。" --output outputs/text-pipeline-reply.wav
```

## Audio Pipeline MVP

After the text pipeline works, validate one offline audio turn:

```bash
python auralis_lab/pipeline.py --input-type audio --audio samples/asr/boy.wav --output outputs/audio-pipeline-reply.wav
```

## Persistent Runtime

After the single-turn pipeline is stable, use the persistent runtime for repeated local turns without reloading ASR/TTS models each time:

```bash
export PYTHONPATH=/home/xiezc/Auralis/third_party/CosyVoice:$PYTHONPATH
python auralis_lab/runtime.py
```

Default stack:

```text
sherpa_onnx SenseVoice -> qwen3:8b -> CosyVoice SFT
```

Runtime commands:

```text
text 请用一句话介绍你自己。
audio samples/asr/boy.wav
quit
```

Generated replies are saved under `outputs/runtime/`.

## Notes

- See `docs/model-candidates.md` for model comparison notes.
- See `docs/validation-plan.md` for the staged validation plan.
- See `docs/streaming-vs-non-streaming-asr.md` for the ASR streaming decision.
- See `docs/tts-validation.md` for TTS candidate setup and test commands.
- See `docs/llm-validation.md` for Qwen3-8B local deployment and validation.
- See `docs/pipeline-validation.md` for text and audio pipeline validation.
- See `docs/streaming-runtime-plan.md` for the live microphone/speaker runtime implementation plan.
- See `docs/troubleshooting.md` for environment, proxy, model download, and runtime issues we have already solved.
