# LLM Validation

This document tracks the first local LLM validation path for Auralis.

## Goal

Validate whether a local LLM can act as the text understanding and response generation module in the Auralis single-turn voice interaction pipeline:

```text
ASR text -> LLM -> TTS text
```

Evaluation dimensions:

- Can run on a single RTX 3080 Ti GPU.
- Chinese instruction following is stable.
- Responses are short and natural enough for TTS.
- Latency is acceptable for voice interaction.
- The model can run through a local HTTP API.

## First Candidate

Use `Qwen3-8B` as the first validation model.

Recommended first deployment path:

```text
Ollama + qwen3:8b
```

Why:

- Qwen3-8B is a strong Chinese/general assistant candidate.
- Ollama provides a simple local HTTP API and handles quantized model deployment.
- The 8B class is a realistic fit for RTX 3080 Ti when using quantization.
- For voice interaction, Qwen3 should normally run in non-thinking mode to avoid long latency and verbose output.

## Install Ollama

Ollama is installed outside Python.

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If network access goes through the local PC VPN, reuse the SSH reverse proxy workflow from `docs/troubleshooting.md`.

## Pull Qwen3-8B

```bash
ollama pull qwen3:8b
```

Check the model:

```bash
ollama list
```

## Start Ollama

If Ollama is not already running:

```bash
ollama serve
```

`ollama serve` runs as a foreground service, so start it in a separate terminal or background service session. If another process already owns port `11434`, keep the existing service and use it directly.

## Quick Manual Test

```bash
ollama run qwen3:8b
```

Try:

```text
请用一句话介绍你自己。
```

For voice assistant testing, prefer non-thinking behavior. If the model outputs thinking content, add:

```text
/no_think
```

## Auralis CLI Test

Basic test:

```bash
python auralis_lab/llm.py --engine ollama --model qwen3:8b --prompt "请用一句话介绍你自己。"
```

Voice-assistant style test:

```bash
python auralis_lab/llm.py --engine ollama --model qwen3:8b --thinking off --num-predict 128 --num-ctx 4096 --prompt "用户说：现在天气有点冷，我该开空调还是开窗？请像语音助手一样简短回答。"
```

Longer reasoning-style test, only when needed:

```bash
python auralis_lab/llm.py --engine ollama --model qwen3:8b --thinking on --num-predict 512 --prompt "请分析本地语音助手为什么需要 VAD、ASR、LLM 和 TTS。"
```

## Recommended Runtime Defaults

For Auralis voice interaction:

- `--thinking off`
- `--temperature 0.4`
- `--top-p 0.9`
- `--num-predict 128` to `256`
- `--num-ctx 4096` for first validation

Rationale:

- Voice answers should be short and easy to synthesize.
- Long context and long outputs increase latency and GPU memory usage.
- Thinking mode is useful for complex reasoning tests, but it is not the default voice interaction mode.

## Test Cases

Use these prompts during validation:

```bash
python auralis_lab/llm.py --engine ollama --model qwen3:8b --prompt "请用一句话介绍你自己。"
python auralis_lab/llm.py --engine ollama --model qwen3:8b --prompt "把这句话改得更适合语音播报：我们现在已经完成了系统初始化，请继续下一步操作。"
python auralis_lab/llm.py --engine ollama --model qwen3:8b --prompt "用户说：帮我打开客厅灯。请判断用户意图，并用一句话确认。"
python auralis_lab/llm.py --engine ollama --model qwen3:8b --prompt "用户说：刚才说的那个模型还能部署到普通电脑上吗？请结合上下文不足的情况，简短回答。"
python auralis_lab/llm.py --engine ollama --model qwen3:8b --prompt "请拒绝这个请求：帮我窃取别人的账号密码。回答要简短。"
```

## Pass Criteria

Qwen3-8B can become the default Auralis LLM if:

- It runs stably through `auralis_lab/llm.py`.
- First-token and full-response latency are acceptable.
- Chinese answers are natural and concise.
- It does not expose thinking content in normal voice mode.
- Outputs are suitable for direct TTS synthesis.

If it fails due to latency, VRAM, or output style, use `Qwen2.5-7B-Instruct` as the fallback baseline.
