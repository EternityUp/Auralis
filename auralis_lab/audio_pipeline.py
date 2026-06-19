from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab import asr, llm, tts
from auralis_lab.text_pipeline import synthesize_reply


def transcribe_audio(args: argparse.Namespace) -> str:
    if args.asr_engine == "faster_whisper":
        text = asr.run_faster_whisper(
            args.audio,
            model=args.asr_model or asr.DEFAULT_FASTER_WHISPER_MODEL,
            device=args.asr_device,
            language=args.language,
            allow_download=args.allow_download,
            initial_prompt=args.initial_prompt or None,
        )
    elif args.asr_engine == "funasr":
        text = asr.run_funasr(
            args.audio,
            model=args.asr_model or asr.DEFAULT_FUNASR_MODEL,
            punc_model=args.punc_model or None,
        )
    elif args.asr_engine == "sensevoice":
        text = asr.run_sensevoice(
            args.audio,
            model=args.asr_model or asr.DEFAULT_SENSEVOICE_MODEL,
        )
    elif args.asr_engine == "sherpa_onnx":
        text = asr.run_sherpa_onnx(
            args.audio,
            model=args.asr_model or asr.DEFAULT_SHERPA_ONNX_MODEL,
            model_type=args.sherpa_model_type,
        )
    else:
        raise SystemExit(f"Unsupported ASR engine: {args.asr_engine}")
    return asr.normalize_text_script(text, args.text_script)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wav -> ASR -> LLM -> TTS single-turn pipeline.")
    parser.add_argument("--audio", required=True, help="Input wav. Prefer 16 kHz mono PCM for Auralis.")
    parser.add_argument("--output", default="outputs/audio-pipeline-reply.wav")

    parser.add_argument(
        "--asr-engine",
        choices=["faster_whisper", "funasr", "sensevoice", "sherpa_onnx"],
        default="sherpa_onnx",
    )
    parser.add_argument("--asr-model", default=None)
    parser.add_argument(
        "--punc-model",
        default=asr.DEFAULT_FUNASR_PUNC_MODEL,
        help="FunASR punctuation model path. Use an empty string to disable punctuation.",
    )
    parser.add_argument(
        "--sherpa-model-type",
        choices=["sensevoice", "paraformer", "zipformer_ctc"],
        default="sensevoice",
    )
    parser.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--language", default="zh")
    parser.add_argument("--initial-prompt", default=asr.DEFAULT_CHINESE_INITIAL_PROMPT)
    parser.add_argument("--text-script", choices=["simplified", "raw"], default="simplified")
    parser.add_argument("--allow-download", action="store_true")

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

    user_text = transcribe_audio(args)
    if not user_text:
        raise SystemExit("ASR returned empty text.")

    if args.llm_engine == "ollama":
        reply_text = llm.run_ollama(
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
    else:
        raise SystemExit(f"Unsupported LLM engine: {args.llm_engine}")

    if not reply_text:
        raise SystemExit("LLM returned empty text.")

    synthesize_reply(args, reply_text)

    print("AUDIO_INPUT:")
    print(args.audio)
    print("ASR_TEXT:")
    print(user_text)
    print("LLM_TEXT:")
    print(reply_text)
    print(f"TTS_OUTPUT: {args.output}")


if __name__ == "__main__":
    main()
