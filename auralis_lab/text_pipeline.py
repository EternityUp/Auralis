from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab import llm, tts


def synthesize_reply(args: argparse.Namespace, reply_text: str) -> None:
    if args.tts_engine == "edge_tts":
        asyncio.run(tts.run_edge_tts(reply_text, args.output, args.voice))
    elif args.tts_engine == "piper":
        tts.run_piper(
            reply_text,
            args.output,
            model=args.tts_model or tts.DEFAULT_PIPER_MODEL,
            executable=args.executable,
        )
    elif args.tts_engine == "cosyvoice":
        default_model = (
            tts.DEFAULT_COSYVOICE_SFT_MODEL
            if args.cosy_mode == "sft"
            else tts.DEFAULT_COSYVOICE_MODEL
        )
        tts.run_cosyvoice(
            reply_text,
            args.output,
            model=args.tts_model or default_model,
            mode=args.cosy_mode,
            speaker=args.speaker,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text,
            instruct_text=args.instruct_text,
            prompt_loader=args.cosy_prompt_loader,
            device=args.cosy_device,
            fp16=not args.no_cosy_fp16,
        )
    elif args.tts_engine == "voxcpm":
        default_model = (
            tts.DEFAULT_VOXCPM2_MODEL
            if args.voxcpm_variant == "voxcpm2"
            else tts.DEFAULT_VOXCPM_MODEL
        )
        tts.run_voxcpm(
            reply_text,
            args.output,
            model=args.tts_model or default_model,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text,
            cfg_value=args.voxcpm_cfg,
            inference_timesteps=args.voxcpm_timesteps,
            normalize=not args.no_voxcpm_normalize,
            denoise=args.voxcpm_denoise,
            device=args.voxcpm_device,
        )
    else:
        raise SystemExit(f"Unsupported TTS engine: {args.tts_engine}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text -> LLM -> TTS single-turn pipeline.")
    parser.add_argument("--text", required=True, help="User text input.")
    parser.add_argument("--output", default="outputs/text-pipeline-reply.wav")

    parser.add_argument("--llm-engine", choices=["ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--system-prompt", default=llm.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=4096)

    parser.add_argument(
        "--tts-engine",
        choices=["edge_tts", "piper", "cosyvoice", "voxcpm"],
        default="cosyvoice",
    )
    parser.add_argument("--tts-model", default=None)
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--speaker", default="中文女")
    parser.add_argument(
        "--cosy-mode",
        choices=["sft", "zero_shot", "instruct2"],
        default="sft",
    )
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--prompt-text", default="")
    parser.add_argument(
        "--cosy-prompt-loader",
        choices=["path", "soundfile"],
        default="path",
    )
    parser.add_argument(
        "--cosy-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--no-cosy-fp16", action="store_true")
    parser.add_argument("--instruct-text", default="用自然流畅的普通话朗读。<|endofprompt|>")
    parser.add_argument("--voxcpm-variant", choices=["0.5b", "voxcpm2"], default="0.5b")
    parser.add_argument("--voxcpm-cfg", type=float, default=2.0)
    parser.add_argument("--voxcpm-timesteps", type=int, default=10)
    parser.add_argument("--no-voxcpm-normalize", action="store_true")
    parser.add_argument("--voxcpm-denoise", action="store_true")
    parser.add_argument(
        "--voxcpm-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--executable", default="piper")
    args = parser.parse_args()

    if args.llm_engine == "ollama":
        reply_text = llm.run_ollama(
            args.text,
            model=args.llm_model,
            host=args.llm_host,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            thinking=args.thinking,
        )
    else:
        raise SystemExit(f"Unsupported LLM engine: {args.llm_engine}")

    if not reply_text:
        raise SystemExit("LLM returned empty text.")

    synthesize_reply(args, reply_text)

    print("USER_TEXT:")
    print(args.text)
    print("LLM_TEXT:")
    print(reply_text)
    print(f"TTS_OUTPUT: {args.output}")


if __name__ == "__main__":
    main()
