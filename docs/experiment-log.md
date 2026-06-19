# Experiment Log

Use one row per model run.

| Date | Task | Engine | Model | Device | Input | Latency | Result Summary | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ASR | faster-whisper | small | CPU | samples/asr/test.wav |  |  |  |
|  | ASR | faster-whisper | medium | GPU | samples/asr/test.wav |  |  |  |
|  | ASR | faster-whisper | large-v3 | GPU | samples/asr/test.wav |  |  |  |
|  | ASR | funasr | paraformer-zh + ct-punc | GPU/CPU | samples/asr/test.wav |  |  |  |
|  | ASR | sensevoice | small | GPU/CPU | samples/asr/test.wav |  |  |  |
|  | ASR | sherpa-onnx | sensevoice onnx | CPU | samples/asr/test.wav |  |  |  |
|  | TTS | edge-tts | zh-CN-XiaoxiaoNeural |  | text prompt |  |  |  |
|  | LLM | ollama | qwen3:8b |  | prompt |  |  |  |
