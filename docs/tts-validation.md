# TTS Validation

This document tracks the first TTS candidates for Auralis.

## Goal

Validate multiple TTS engines independently before connecting ASR, LLM, and TTS into a single-turn voice interaction pipeline.

Evaluation dimensions:

- Mandarin naturalness
- pronunciation accuracy
- punctuation handling
- latency
- local/offline feasibility
- installation complexity
- output format and sample rate
- PC deployment potential

Use the same test text for every engine.

Suggested short text:

```text
你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。
```

Suggested longer text:

```text
今天我们先验证语音合成模块。一个合格的本地语音助手，需要说话自然、响应及时，并且能够稳定部署在普通电脑上。
```

## Common Setup

```bash
cd /home/xiezc/Auralis
python -m pip install -r requirements/tts.txt
mkdir -p models/tts outputs
```

## edge-tts

Role: fastest TTS path validation.

Pros:

- Very easy to run.
- Mandarin voice quality is good.
- Good for quickly validating the text-to-audio path.

Cons:

- Requires network access.
- Not a pure local deployment choice.

Run:

```bash
python auralis_lab/tts.py --engine edge_tts --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/edge-tts.wav
```

Try another Mandarin voice:

```bash
python auralis_lab/tts.py --engine edge_tts --voice zh-CN-YunxiNeural --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统" --output outputs/edge-tts-yunxi.wav
```

## Piper

Role: lightweight local TTS baseline.

Pros:

- Local and deployment-friendly.
- Simple command-line runtime.
- Good fit for lightweight PC-side experiments.

Cons:

- Mandarin voice quality and available voices need validation.
- Usually less expressive than heavier neural TTS models.

Expected model path:

```text
models/tts/piper/zh_CN-huayan-medium.onnx
models/tts/piper/zh_CN-huayan-medium.onnx.json
```

Run:

```bash
python auralis_lab/tts.py --engine piper --model models/tts/piper/zh_CN-huayan-medium.onnx --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/piper.wav
```

If the `piper` executable is not on `PATH`, pass it explicitly:

```bash
python auralis_lab/tts.py --engine piper --executable /path/to/piper --model models/tts/piper/zh_CN-huayan-medium.onnx --text "你好，我是 Auralis。" --output outputs/piper.wav
```

## CosyVoice

Setup:

cd /home/xiezc/Auralis
export PYTHONPATH=$PWD/third_party/CosyVoice:$PYTHONPATH

Role: higher-quality Mandarin TTS candidate.

Pros:

- Stronger Mandarin naturalness potential.
- Better fit for assistant-like speech if the environment can support it.

Cons:

- Heavier setup.
- Requires installing the CosyVoice project and its dependencies.
- API may vary by repository version.

CosyVoice supports multiple inference modes. Auralis exposes three of them:

- `sft`: fixed speaker from an SFT model.
- `zero_shot`: voice cloning from a prompt audio and its transcript.
- `instruct2`: CosyVoice2 instruction mode with prompt audio.

### CosyVoice SFT

Expected model path:

```text
models/tts/cosyvoice/CosyVoice-300M-SFT
```

Download:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice-300M-SFT', local_dir='models/tts/cosyvoice/CosyVoice-300M-SFT')"
```

List available speakers:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "/home/xiezc/Auralis/third_party/CosyVoice")
from cosyvoice.cli.cosyvoice import AutoModel

model = AutoModel(model_dir="/home/xiezc/Auralis/models/tts/cosyvoice/CosyVoice-300M-SFT")
print(model.list_available_spks())
PY
```

Run:

```bash
python auralis_lab/tts.py --engine cosyvoice --cosy-mode sft --model models/tts/cosyvoice/CosyVoice-300M-SFT --speaker "中文女" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/cosyvoice-sft.wav
```

Replace `中文女` with one of the actual speakers printed by `list_available_spks()`.

### CosyVoice2 Zero-Shot

Expected model path:

```text
models/tts/cosyvoice/CosyVoice2-0.5B
```

Download:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='models/tts/cosyvoice/CosyVoice2-0.5B')"
```

Prompt audio requirements:

- Duration: **5–10 seconds** is a good validation range. Very short clips may produce unstable speaker embeddings, while very long clips are unnecessary for these first tests.
- Sample rate: 16 kHz mono is suitable. CosyVoice internally loads the same prompt at 16 kHz for speech tokens and speaker embedding, and at 24 kHz for acoustic feature extraction.
- `--prompt-text` should accurately match the spoken content in `--prompt-audio`. Word or sentence mismatches can hurt zero-shot conditioning and may cause unstable or unnatural output. Punctuation does not need to be treated as a strict byte-for-byte match, but it should reflect the intended pauses and sentence boundaries where possible.

Run:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode zero_shot --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/girl.wav --prompt-text "总理对任何事情都要刨根问底。" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/cosyvoice-zero-shot.wav
```

If Python cannot import `cosyvoice.cli.cosyvoice`, install CosyVoice from its official repository and make sure that repository is on `PYTHONPATH`.

### CosyVoice2 Instruct2

`instruct_text` should describe the desired speaking style, emotion, dialect, pace, tone, or role. For Mandarin TTS validation, Chinese instructions are usually the most stable choice. Treat it as a speech-style instruction, not as a general LLM system prompt.

Important: append `<|endofprompt|>` to `--instruct-text`.

CosyVoice2 instruct2 feeds the instruction into the LLM as prompt text. `<|endofprompt|>` is the boundary token that tells the model where the instruction ends and where the target synthesis text begins. Without this token, the instruction itself may be treated as part of the text to synthesize, so the generated audio can read both the instruction and the actual `--text`.

Run:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode instruct2 --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/boy.wav --instruct-text "用自然流畅的普通话朗读。<|endofprompt|>" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/cosyvoice-instruct2.wav
```

Other instruct examples:

```bash
# Adjust speaking style
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode instruct2 --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/boy.wav --instruct-text "语速慢一些，声音低沉。<|endofprompt|>" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/cosyvoice-instruct2-slow.wav

# Dialect
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode instruct2 --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/boy.wav --instruct-text "用四川话说。<|endofprompt|>" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/cosyvoice-instruct2-sichuan.wav
```

If CUDA runs out of memory, first make sure no other Python process is occupying the GPU:

```bash
nvidia-smi
```

Then retry with the CPU fallback:

```bash
python auralis_lab/tts.py --engine cosyvoice --cosy-mode instruct2 --cosy-device cpu --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/boy.wav --instruct-text "用自然流畅的普通话朗读。<|endofprompt|>" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/cosyvoice-instruct2-cpu.wav
```

Notes:

- `instruct2` is heavier than SFT and zero-shot.
- `CUDA_VISIBLE_DEVICES=7` is used in the examples because GPU 7 was the idle GPU during validation. Use `nvidia-smi` and select an available GPU on the target server.
- If CosyVoice2 starts producing abnormal zero-shot or instruct2 audio after ChatTTS dependency changes, reinstalling CosyVoice dependencies from `third_party/CosyVoice/requirements.txt` restored the validated setup in our tests. Keep ChatTTS in a separate environment while CosyVoice2 remains the priority.
- CPU fallback is slow, but it is useful for confirming quality when GPU memory is insufficient.
- Auralis sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before loading CosyVoice unless the environment already defines it.

## ChatTTS

Role: expressive Chinese conversational TTS candidate.

Pros:

- Expressive conversational style.
- Useful for exploring a more natural assistant voice.

Cons:

- Engineering stability needs validation.
- Voice consistency and deployment behavior need careful testing.
- API may vary by package/repository version.
- In our current environment, the dependency changes used to make ChatTTS run disturbed the validated CosyVoice2 setup. For now, ChatTTS is not recommended in the main Auralis environment.

Expected local model path:

```text
models/tts/chattts
```

Run:

```bash
python auralis_lab/tts.py --engine chattts --model models/tts/chattts --text "你好，我是 Auralis。" --output outputs/chattts.wav
```

If no local model path exists, the wrapper will attempt the default ChatTTS load behavior. For reproducible deployment, prefer a local model directory. Current validation result: ChatTTS is not selected for the main Auralis TTS path because its dependency changes disturbed the validated CosyVoice2 environment in our tests.

## VoxCPM

Role: tokenizer-free Mandarin TTS candidate with zero-shot voice cloning.

Pros:

- Tokenizer-free, end-to-end diffusion autoregressive architecture.
- Strong Mandarin naturalness and expressive zero-shot voice cloning.
- Local and deployment-friendly once the model is downloaded.

Cons:

- Heavier setup than Piper.
- The validated setup uses a recent PyTorch/CUDA environment. Follow the VoxCPM model README if a different server environment is used.
- API differs across model families (see note below).

VoxCPM has two model families. Auralis selects the default model with `--voxcpm-variant`:

- `0.5b`: `VoxCPM-0.5B`, uses the `prompt_wav_path` / `prompt_text` cloning API.
- `voxcpm2`: `VoxCPM2`, uses the `reference_wav_path` cloning API.

The wrapper tries the VoxCPM2 signature first and falls back to the VoxCPM-0.5B signature, so both families work without extra flags.

### VoxCPM-0.5B

Expected model path:

```text
models/tts/voxcpm/VoxCPM-0.5B
```

Download:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/VoxCPM-0.5B', local_dir='models/tts/voxcpm/VoxCPM-0.5B')"
```

Run:

```bash
python auralis_lab/tts.py --engine voxcpm --voxcpm-variant 0.5b --model models/tts/voxcpm/VoxCPM-0.5B --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/voxcpm-0.5b.wav
```

### VoxCPM2

Expected model path:

```text
models/tts/voxcpm/VoxCPM2
```

Download:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/VoxCPM2', local_dir='models/tts/voxcpm/VoxCPM2')"
```

Run:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine voxcpm --voxcpm-variant voxcpm2 --model models/tts/voxcpm/VoxCPM2 --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/voxcpm2.wav
```

### Zero-Shot Voice Cloning

Prepare a mono prompt wav and its exact transcript:

```text
samples/tts/prompt.wav
```

Run:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine voxcpm --voxcpm-variant voxcpm2 --model models/tts/voxcpm/VoxCPM2 --prompt-audio samples/tts/boy.wav --prompt-text "你就是那个爱打篮球的人。" --text "你好，我是 Auralis。很高兴和你一起完成本地智能语音交互系统。" --output outputs/voxcpm2-clone.wav
```

Notes:

- `--voxcpm-cfg` (default `2.0`) controls LM guidance; higher follows the prompt more closely.
- `--voxcpm-timesteps` (default `10`) trades quality for speed; higher is better, lower is faster.
- Denoising is disabled by default. Pass `--voxcpm-denoise` to enable it (VoxCPM-0.5B may require an extra denoiser model).
- If CUDA runs out of memory, retry with `--voxcpm-device cpu` as a slow fallback.
- Auralis sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before loading VoxCPM unless the environment already defines it.

## Recommended Test Order

1. `edge-tts`
2. `piper`
3. `CosyVoice`
4. `VoxCPM`
5. `ChatTTS` in an isolated environment, if still needed

Rationale:

- `edge-tts` validates the basic TTS path fastest.
- `piper` validates lightweight local deployment.
- `CosyVoice` evaluates high-quality Mandarin synthesis.
- `VoxCPM` evaluates a tokenizer-free alternative with strong zero-shot cloning.
- `ChatTTS` is currently deprioritized because its dependency changes disturbed CosyVoice in the shared environment during validation.

## Experiment Log Template

| Date | Engine | Model/Voice | Output | Latency | Naturalness | Pronunciation | Local/Offline | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | edge-tts | zh-CN-XiaoxiaoNeural | outputs/edge-tts.wav |  |  |  | No |  |
|  | piper | zh_CN-huayan-medium | outputs/piper.wav |  |  |  | Yes |  |
|  | cosyvoice | CosyVoice-300M-SFT / speaker | outputs/cosyvoice-sft.wav |  |  |  | Yes |  |
|  | cosyvoice | CosyVoice2-0.5B / zero-shot | outputs/cosyvoice-zero-shot.wav |  |  |  | Yes |  |
|  | cosyvoice | CosyVoice2-0.5B / instruct2 | outputs/cosyvoice-instruct2.wav |  |  |  | Yes |  |
|  | voxcpm | VoxCPM-0.5B | outputs/voxcpm-0.5b.wav |  |  |  | Yes |  |
|  | voxcpm | VoxCPM2 | outputs/voxcpm2.wav |  |  |  | Yes |  |
|  | voxcpm | VoxCPM2 / zero-shot | outputs/voxcpm2-clone.wav |  |  |  | Yes |  |
|  | chattts | local/default | outputs/chattts.wav |  |  |  | Yes |  |
