from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab import llm
from auralis_lab.audio_pipeline import transcribe_audio
from auralis_lab.text_pipeline import synthesize_reply


def add_asr_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asr-engine",
        choices=["faster_whisper", "funasr", "sensevoice", "sherpa_onnx"],
        default="sherpa_onnx",
    )
    parser.add_argument("--asr-model", default=None)
    parser.add_argument(
        "--punc-model",
        default="models/asr/funasr-ct-punc",
        help="FunASR punctuation model path. Use an empty string to disable punctuation.",
    )
    parser.add_argument(
        "--sherpa-model-type",
        choices=["sensevoice", "paraformer", "zipformer_ctc"],
        default="sensevoice",
    )
    parser.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--initial-prompt",
        default="Mandarin transcript in simplified Chinese with Chinese punctuation.",
    )
    parser.add_argument("--text-script", choices=["simplified", "raw"], default="simplified")
    parser.add_argument("--allow-download", action="store_true")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-engine", choices=["ollama"], default="ollama")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--system-prompt", default=llm.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=4096)


def add_tts_args(parser: argparse.ArgumentParser) -> None:
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


def run_llm(args: argparse.Namespace, user_text: str) -> str:
    if args.llm_engine == "ollama":
        return llm.run_ollama(
            user_text,
            model=args.llm_model,
            host=args.llm_host,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            thinking=args.thinking,
        )
    raise SystemExit(f"Unsupported LLM engine: {args.llm_engine}")


def timed(label: str, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Auralis single-turn pipeline.")
    parser.add_argument(
        "--input-type",
        choices=["text", "audio"],
        required=True,
        help="Use text for manual text input, or audio for wav input.",
    )
    parser.add_argument("--text", default=None, help="Manual user text when --input-type text.")
    parser.add_argument("--audio", default=None, help="Input wav when --input-type audio.")
    parser.add_argument("--output", default="outputs/pipeline-reply.wav")

    add_asr_args(parser)
    add_llm_args(parser)
    add_tts_args(parser)
    args = parser.parse_args()

    if args.input_type == "text":
        if not args.text:
            raise SystemExit("--input-type text requires --text.")
        user_text = args.text
        asr_seconds = 0.0
    elif args.input_type == "audio":
        if not args.audio:
            raise SystemExit("--input-type audio requires --audio.")
        user_text, asr_seconds = timed("asr", transcribe_audio, args)
    else:
        raise SystemExit(f"Unsupported input type: {args.input_type}")

    if not user_text:
        raise SystemExit("Pipeline input text is empty.")

    reply_text, llm_seconds = timed("llm", run_llm, args, user_text)
    if not reply_text:
        raise SystemExit("LLM returned empty text.")

    _, tts_seconds = timed("tts", synthesize_reply, args, reply_text)
    total_seconds = asr_seconds + llm_seconds + tts_seconds

    print(f"PIPELINE_MODE: {args.input_type}")
    if args.input_type == "audio":
        print("AUDIO_INPUT:")
        print(args.audio)
        print("ASR_TEXT:")
        print(user_text)
    else:
        print("USER_TEXT:")
        print(user_text)
    print("LLM_TEXT:")
    print(reply_text)
    print(f"TTS_OUTPUT: {args.output}")
    print("LATENCY:")
    if args.input_type == "audio":
        print(f"  asr_seconds: {asr_seconds:.3f}")
    print(f"  llm_seconds: {llm_seconds:.3f}")
    print(f"  tts_seconds: {tts_seconds:.3f}")
    print(f"  total_seconds: {total_seconds:.3f}")


if __name__ == "__main__":
    main()
