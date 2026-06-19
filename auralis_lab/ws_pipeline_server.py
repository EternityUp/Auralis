from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab import asr, tts
from auralis_lab.pipeline import add_asr_args, add_llm_args, add_tts_args, run_llm, timed
from auralis_lab.runtime import CosyVoiceRuntimeTTS, SherpaOnnxRuntimeASR


def parse_json_message(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        return {
            "type": "bytes",
            "size": len(message),
        }
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        return {
            "type": message,
        }
    if not isinstance(parsed, dict):
        return {
            "type": "json",
            "payload": parsed,
        }
    return parsed


def save_uploaded_audio(output_dir: str, audio_bytes: bytes, suffix: str = ".wav") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"turn-input-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}{suffix}"
    target = output_path / filename
    target.write_bytes(audio_bytes)
    return str(target)


def output_path_for_input(input_path: str, output_dir: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem.replace("turn-input", "turn-reply")
    return str(output_path / f"{stem}.wav")


class PipelineRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.asr_engine != "sherpa_onnx":
            raise SystemExit("Persistent ws_pipeline_server currently supports --asr-engine sherpa_onnx only.")
        if args.tts_engine != "cosyvoice":
            raise SystemExit("Persistent ws_pipeline_server currently supports --tts-engine cosyvoice only.")

        print("Loading persistent ASR runtime...")
        self.asr_runtime = SherpaOnnxRuntimeASR(
            model=args.asr_model or asr.DEFAULT_SHERPA_ONNX_MODEL,
            model_type=args.sherpa_model_type,
            text_script=args.text_script,
        )

        default_tts_model = (
            tts.DEFAULT_COSYVOICE_SFT_MODEL
            if args.cosy_mode == "sft"
            else tts.DEFAULT_COSYVOICE_MODEL
        )
        print("Loading persistent TTS runtime...")
        self.tts_runtime = CosyVoiceRuntimeTTS(
            model=args.tts_model or default_tts_model,
            mode=args.cosy_mode,
            speaker=args.speaker,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text,
            instruct_text=args.instruct_text,
            prompt_loader=args.cosy_prompt_loader,
            device=args.cosy_device,
            fp16=not args.no_cosy_fp16,
        )
        self.lock = Lock()

    def transcribe(self, input_wav: str) -> str:
        return self.asr_runtime.transcribe(input_wav)

    def synthesize(self, text: str, output_wav: str) -> None:
        self.tts_runtime.synthesize(text, output_wav)


def run_offline_turn(args: argparse.Namespace, runtime: PipelineRuntime, input_wav: str) -> dict[str, Any]:
    output_wav = output_path_for_input(input_wav, args.reply_output_dir)

    # The model runtimes are intentionally shared across turns. Keep each full
    # turn serialized for now because CosyVoice and GPU memory are not treated
    # as concurrent resources in this prototype.
    with runtime.lock:
        user_text, asr_seconds = timed("asr", runtime.transcribe, input_wav)
        if not user_text:
            raise RuntimeError("ASR returned empty text.")

        reply_text, llm_seconds = timed("llm", run_llm, args, user_text)
        if not reply_text:
            raise RuntimeError("LLM returned empty text.")

        _, tts_seconds = timed("tts", runtime.synthesize, reply_text, output_wav)

    total_seconds = asr_seconds + llm_seconds + tts_seconds

    return {
        "input_wav": input_wav,
        "reply_wav": output_wav,
        "asr_text": user_text,
        "llm_text": reply_text,
        "latency": {
            "asr_seconds": asr_seconds,
            "llm_seconds": llm_seconds,
            "tts_seconds": tts_seconds,
            "total_seconds": total_seconds,
        },
    }


async def send_pipeline_reply(websocket: Any, result: dict[str, Any]) -> None:
    reply_path = Path(result["reply_wav"])
    if not reply_path.exists():
        raise RuntimeError(f"Reply wav was not generated: {reply_path}")

    metadata = {
        "type": "reply_audio",
        "server_time": time.time(),
        "format": "wav",
        "suffix": ".wav",
        "filename": reply_path.name,
        "bytes": reply_path.stat().st_size,
        "pipeline": result,
    }
    await websocket.send(json.dumps(metadata, ensure_ascii=False))
    await websocket.send(reply_path.read_bytes())


async def handle_connection(websocket: Any, args: argparse.Namespace, runtime: PipelineRuntime) -> None:
    peer = getattr(websocket, "remote_address", None)
    print(f"Client connected: {peer}")
    pending_audio_meta: dict[str, Any] | None = None
    try:
        async for message in websocket:
            if isinstance(message, bytes) and pending_audio_meta is not None:
                input_wav = save_uploaded_audio(
                    args.upload_output_dir,
                    message,
                    suffix=pending_audio_meta.get("suffix", ".wav"),
                )
                ack = {
                    "type": "audio_upload_ack",
                    "server_time": time.time(),
                    "saved_path": input_wav,
                    "bytes": len(message),
                    "metadata": pending_audio_meta,
                }
                await websocket.send(json.dumps(ack, ensure_ascii=False))

                print(f"Running offline turn for: {input_wav}")
                try:
                    result = await asyncio.to_thread(run_offline_turn, args, runtime, input_wav)
                    print("ASR_TEXT:", result["asr_text"])
                    print("LLM_TEXT:", result["llm_text"])
                    print("LATENCY:", result["latency"])
                    await send_pipeline_reply(websocket, result)
                except BaseException as exc:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "pipeline_error",
                                "server_time": time.time(),
                                "message": str(exc),
                            },
                            ensure_ascii=False,
                        )
                    )

                pending_audio_meta = None
                continue

            payload = parse_json_message(message)
            message_type = payload.get("type")
            if message_type == "ping":
                response = {
                    "type": "pong",
                    "server_time": time.time(),
                    "message": "Auralis pipeline server is reachable.",
                }
            elif message_type == "audio_upload":
                pending_audio_meta = payload
                response = {
                    "type": "ready_for_audio",
                    "server_time": time.time(),
                    "message": "Send audio bytes as the next websocket message.",
                }
            else:
                response = {
                    "type": "echo",
                    "server_time": time.time(),
                    "payload": payload,
                }
            await websocket.send(json.dumps(response, ensure_ascii=False))
    except Exception as exc:
        print(f"Client disconnected: {peer}, reason: {exc}")
    else:
        print(f"Client disconnected: {peer}")


async def run_server(args: argparse.Namespace) -> None:
    try:
        websockets = __import__("websockets")
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: websockets\nInstall it with:\n  python -m pip install websockets") from exc

    runtime = PipelineRuntime(args)

    if not args.no_warmup:
        warmup_output = str(Path(args.reply_output_dir) / "_warmup.wav")
        print("Warming up persistent TTS runtime...")
        runtime.synthesize(args.warmup_text, warmup_output)

    async def handler(websocket: Any, path: str | None = None) -> None:
        await handle_connection(websocket, args, runtime)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        print(f"Auralis WebSocket pipeline server listening on ws://{args.host}:{args.port}")
        print(f"Uploaded audio will be saved under: {args.upload_output_dir}")
        print(f"Reply audio will be saved under: {args.reply_output_dir}")
        print("Pipeline stack:")
        print(f"  ASR: {args.asr_engine}")
        print(f"  LLM: {args.llm_engine} / {args.llm_model}")
        print(f"  TTS: {args.tts_engine} / {args.cosy_mode if args.tts_engine == 'cosyvoice' else args.tts_engine}")
        print("Persistent ASR/TTS runtime is enabled.")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Auralis offline ASR->LLM->TTS WebSocket server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--upload-output-dir", default="outputs/ws_pipeline_uploads")
    parser.add_argument("--reply-output-dir", default="outputs/ws_pipeline_replies")
    parser.add_argument("--warmup-text", default="你好。")
    parser.add_argument("--no-warmup", action="store_true")

    add_asr_args(parser)
    add_llm_args(parser)
    add_tts_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
