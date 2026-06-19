# Troubleshooting Notes

This document records the environment and model issues encountered during Auralis validation.

## Project Rules We Settled

- The full Auralis audio flow uses 16 kHz audio.
- ASR test audio should be 16 kHz mono wav when possible.
- ASR models are stored under `models/asr/`.
- Runtime scripts should prefer local model directories instead of downloading models during inference.
- Chinese ASR output is converted to simplified Chinese by default with OpenCC.

## Python Environment

Recommended environment:

```bash
conda create -n auralis python=3.10 -y
conda activate auralis
python -m pip install -U pip setuptools wheel
```

If `conda activate` is unavailable in a raw SSH shell:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate auralis
```

If conda itself is inconvenient, use the environment Python directly:

```bash
/home/xiezc/anaconda3/envs/auralis/bin/python auralis_lab/asr.py --engine faster_whisper --audio samples/asr/Tencent_test.wav
```

## SSH Proxy Through Local VPN

Problem:

The server can SSH to the user session but cannot directly access Hugging Face or other model hosts. The local PC has VPN.

Solution:

Start SSH from the local PC with reverse forwarding. Example when the local proxy is `127.0.0.1:7890`:

```bash
ssh -R 8888:127.0.0.1:7890 xiezc@192.168.16.206
```

On the server SSH session:

```bash
export HTTP_PROXY=http://127.0.0.1:8888
export HTTPS_PROXY=http://127.0.0.1:8888
export http_proxy=http://127.0.0.1:8888
export https_proxy=http://127.0.0.1:8888
```

Test:

```bash
curl -I https://huggingface.co
curl -I https://repo.anaconda.com
```

## Conda libmamba Error

Problem:

```text
Error while loading conda entry point: conda-libmamba-solver
module 'libmambapy' has no attribute 'QueryFormat'
```

Fast workaround:

Use the environment Python directly:

```bash
/home/xiezc/anaconda3/envs/auralis/bin/python auralis_lab/asr.py --engine faster_whisper --audio samples/asr/Tencent_test.wav
```

Possible repair after proxy works:

```bash
conda config --set solver classic
conda remove -n base conda-libmamba-solver libmambapy
```

or reinstall:

```bash
conda install -n base conda-libmamba-solver libmambapy -c defaults --force-reinstall
```

## PyTorch And CUDA

Observed server driver:

```text
Driver Version: 570.133.07
CUDA Version: 12.8
```

Recommended PyTorch install:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Validate:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## Hugging Face CLI

Problem:

`huggingface-cli` may print:

```text
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
```

Solution:

Use `hf`:

```bash
python -m pip install -U huggingface_hub
hf download Systran/faster-whisper-small --local-dir models/asr/faster-whisper-small
```

## faster-whisper Model Download

Problem:

When `faster-whisper` receives model name `small`, it tries to download from Hugging Face. If the network is unavailable or slow, inference appears stuck.

Solution:

Download models outside runtime and pass local model paths:

```bash
hf download Systran/faster-whisper-small --local-dir models/asr/faster-whisper-small
hf download Systran/faster-whisper-medium --local-dir models/asr/faster-whisper-medium
hf download Systran/faster-whisper-large-v3 --local-dir models/asr/faster-whisper-large-v3
```

Run:

```bash
python auralis_lab/asr.py --engine faster_whisper --model models/asr/faster-whisper-small --audio samples/asr/Tencent_test.wav
```

## Simplified Chinese And Punctuation

Problem:

`faster-whisper-small` may output traditional Chinese and no punctuation.

Solution:

- Auralis converts Chinese text to simplified Chinese with OpenCC by default.
- Use `--text-script raw` to inspect raw model output.
- Punctuation is not a hard switch in Whisper-family models. It depends on model size, audio pauses, segmentation, and decoding.
- For stable punctuation, use FunASR with a punctuation model or SenseVoice.

## FunASR Dependency Confusion

Problem:

The script may show `Missing dependency: funasr` while the real issue is missing PyTorch.

Diagnosis:

```bash
which python
python -m pip show funasr
python -m pip show torch
python -c "import funasr; print(funasr.__file__)"
```

Install ASR dependencies:

```bash
python -m pip install -r requirements/asr.txt
```

Install PyTorch separately according to GPU/CUDA needs.

## FunASR Model Cache

Problem:

FunASR prints:

```text
Downloading Model from https://www.modelscope.cn to directory: ~/.cache/modelscope/...
```

This does not always mean a full re-download. It may be ModelScope resolving or checking the cached model.

Solution:

Keep project-local copies:

```bash
cp -r ~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch models/asr/funasr-paraformer-zh
```

Disable update checks in code:

```python
funasr.AutoModel(..., disable_update=True)
```

## FunASR Punctuation Model

Download:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', local_dir='models/asr/funasr-ct-punc')"
```

Run:

```bash
python auralis_lab/asr.py --engine funasr --model models/asr/funasr-paraformer-zh --punc-model models/asr/funasr-ct-punc --audio samples/asr/Tencent_test.wav
```

## SenseVoiceSmall

Download:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/SenseVoiceSmall', local_dir='models/asr/sensevoice-small')"
```

Run:

```bash
python auralis_lab/asr.py --engine sensevoice --model models/asr/sensevoice-small --audio samples/asr/Tencent_test.wav
```

Problem:

With `trust_remote_code=True`, FunASR may print:

```text
Loading remote code failed: model, No module named 'model'
```

Resolution:

Using `trust_remote_code=False` removes this warning in the current local model setup, and inference still works.

## sherpa-onnx

Download Sherpa's SenseVoice ONNX model:

```bash
cd /home/xiezc/Auralis/models/asr
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

If `wget` is unavailable:

```bash
curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

Run:

```bash
cd /home/xiezc/Auralis
python auralis_lab/asr.py --engine sherpa_onnx --sherpa-model-type sensevoice --model models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 --audio samples/asr/Tencent_test.wav
```

Problem:

Some `sherpa-onnx` Python package versions do not provide `sherpa_onnx.read_wave`.

Solution:

Auralis now reads 16-bit PCM wav files internally with Python's `wave` module and `numpy`, then feeds samples into sherpa-onnx.

## CosyVoice Dependency And CUDA Notes

CosyVoice is the preferred high-quality local TTS candidate.

Current project decision:

- Prioritize CosyVoice quality and stability.
- Keep the `transformers` version required by the installed CosyVoice repository. In the validated environment, CosyVoice2 worked after reinstalling its official requirements.
- Avoid downgrading the main `auralis` environment just to support ChatTTS.
- If CosyVoice2 zero-shot or instruct2 becomes abnormal after ChatTTS experiments, reinstalling the CosyVoice dependencies from `third_party/CosyVoice/requirements.txt` restored the validated setup in our tests.

If CosyVoice reports CUDA out of memory:

```bash
nvidia-smi
```

Close other GPU workloads, or select an idle GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=7 python auralis_lab/tts.py --engine cosyvoice --cosy-mode zero_shot --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/prompt.wav --prompt-text "这段参考音频对应的文字。" --text "你好，我是 Auralis。" --output outputs/cosyvoice-zero-shot.wav
```

If memory is still insufficient:

```bash
python auralis_lab/tts.py --engine cosyvoice --cosy-mode instruct2 --cosy-device cpu --model models/tts/cosyvoice/CosyVoice2-0.5B --prompt-audio samples/tts/prompt.wav --instruct-text "用自然流畅的普通话朗读。<|endofprompt|>" --text "你好，我是 Auralis。" --output outputs/cosyvoice-instruct2-cpu.wav
```

CPU inference is slow, but it is useful as a quality fallback.

Do not use `--no-cosy-fp16` as the first fix for CosyVoice2. In the validated environment, disabling fp16 can cause Qwen2 dtype mismatch errors such as `mat1 and mat2 must have the same dtype`.

## ChatTTS Compatibility

In our tests, ChatTTS could run after changing the `transformers` version, but those dependency changes disturbed CosyVoice2 in the current shared environment.

Decision:

- Avoid using ChatTTS in the main `auralis` environment.
- If ChatTTS is tested later, use a separate conda environment with a compatible `ChatTTS` and `transformers` pair.
- Keep ChatTTS as a non-primary exploratory candidate.

## Direct Script Running And Debuggers

Problem:

Running `python auralis_lab/asr.py` directly may fail with:

```text
ModuleNotFoundError: No module named 'auralis_lab'
```

Solution:

The script entry points add the project root to `sys.path` when run directly. Both forms are supported:

```bash
python -m auralis_lab.asr --engine faster_whisper --audio samples/asr/Tencent_test.wav
python auralis_lab/asr.py --engine faster_whisper --audio samples/asr/Tencent_test.wav
```
