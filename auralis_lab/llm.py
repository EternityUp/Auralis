from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_SYSTEM_PROMPT = (
    "你是 Auralis 的本地语音助手中枢。"
    "请用自然、简洁、适合语音播报的中文回答。"
    "除非用户要求展开，否则回答控制在一到三句话。"
    "只根据当前对话和用户明确提供的信息回答。"
    "对于实时天气、温度、路况、新闻、价格、日期或其他需要外部实时数据的问题，"
    "如果没有提供数据源或工具结果，必须明确说明无法确认实时信息，并建议用户查询可靠来源。"
    "不得编造具体数值、实时状态、地点或结论。"
    "不要输出 Markdown 标题、项目符号或思考过程。"
)


def build_prompt(prompt: str, thinking: str) -> str:
    if thinking == "off":
        return f"{prompt}\n/no_think"
    if thinking == "on":
        return f"{prompt}\n/think"
    return prompt


def clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think(?:ing)?>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def run_ollama(
    prompt: str,
    model: str,
    host: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    num_ctx: int,
    thinking: str,
) -> str:
    return run_ollama_messages(
        [{"role": "user", "content": prompt}],
        model=model,
        host=host,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
        num_ctx=num_ctx,
        thinking=thinking,
    )


def run_ollama_messages(
    messages: list[dict[str, str]],
    model: str,
    host: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    num_ctx: int,
    thinking: str,
) -> str:
    if not messages:
        raise ValueError("Ollama messages must not be empty.")

    request_messages = []
    if system_prompt:
        request_messages.append({"role": "system", "content": system_prompt})
    for index, source in enumerate(messages):
        role = source.get("role", "user")
        content = source.get("content", "")
        if index == len(messages) - 1 and role == "user":
            content = build_prompt(content, thinking)
        request_messages.append({"role": role, "content": content})
    payload = json.dumps(
        {
            "model": model,
            "messages": request_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            "Unable to reach Ollama. Start it and pull a model first:\n"
            f"  ollama pull {model}\n"
            "  ollama serve"
        ) from exc
    message = data.get("message", {})
    return clean_model_output(str(message.get("content", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one LLM engine.")
    parser.add_argument("--engine", choices=["ollama"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for assistant style control. Use an empty string to disable it.",
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "on", "auto"],
        default="off",
        help="Qwen3 thinking control. For voice interaction, 'off' is the default.",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args()

    if args.engine == "ollama":
        text = run_ollama(
            args.prompt,
            model=args.model,
            host=args.host,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            thinking=args.thinking,
        )
    else:
        raise SystemExit(f"Unsupported LLM engine: {args.engine}")

    print("LLM_TEXT:")
    print(text)


if __name__ == "__main__":
    main()
