from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab import asr, llm
from auralis_lab.runtime import SherpaOnnxRuntimeASR
from auralis_lab.ws_stream_asr_server import (
    build_asr,
    normalize_for_noise_match,
    should_drop_asr_text,
    transcribe_utterance,
)
from auralis_lab.ws_stream_vad_server import (
    DEFAULT_FUNASR_VAD_MODEL,
    DEFAULT_SILERO_VAD_MODEL,
    UtteranceSegmenter,
    build_vad,
    parse_json_message,
    save_pcm16_wav,
    vad_accepts_utterance,
)
from auralis_lab.common import AUDIO_SAMPLE_RATE


async def send_event(websocket: Any, payload: dict[str, Any], label: str) -> None:
    print(f"{label}:", payload)
    await websocket.send(json.dumps(payload, ensure_ascii=False))


async def send_asr_result(
    websocket: Any,
    utterance_index: int,
    audio_path: str,
    duration_seconds: float,
    asr_text: str,
    asr_seconds: float,
    vad_engine: str,
    asr_engine: str,
) -> None:
    await send_event(
        websocket,
        {
            "type": "asr_result",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "audio_path": audio_path,
            "duration_seconds": duration_seconds,
            "asr_text": asr_text,
            "asr_seconds": asr_seconds,
            "vad_engine": vad_engine,
            "asr_engine": asr_engine,
        },
        "ASR_RESULT",
    )


def run_llm_turn(args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[str, float]:
    start = time.perf_counter()
    reply = llm.run_ollama_messages(
        messages,
        model=args.llm_model,
        host=args.llm_host,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        thinking=args.thinking,
    )
    return reply, time.perf_counter() - start


def trim_history(history: list[dict[str, str]], max_history_turns: int) -> None:
    if max_history_turns <= 0:
        history.clear()
        return
    max_messages = max_history_turns * 2
    if len(history) > max_messages:
        del history[:-max_messages]


async def process_utterance(
    websocket: Any,
    args: argparse.Namespace,
    vad: Any,
    asr_runtime: SherpaOnnxRuntimeASR,
    history: list[dict[str, str]],
    utterance: list[bytes],
    utterance_index: int,
) -> bool:
    if not vad_accepts_utterance(vad, utterance):
        return False

    saved_path, duration_seconds, bytes_received = save_pcm16_wav(
        args.output_dir,
        f"utt-{utterance_index:03d}",
        utterance,
    )
    await send_event(
        websocket,
        {
            "type": "utterance_saved",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "saved_path": saved_path,
            "duration_seconds": duration_seconds,
            "bytes_received": bytes_received,
            "vad_engine": vad.name,
        },
        "UTTERANCE_SAVED",
    )

    asr_text, asr_seconds = await asyncio.to_thread(transcribe_utterance, asr_runtime, saved_path)
    drop_asr, drop_reason = should_drop_asr_text(asr_text, duration_seconds, args)
    if drop_asr:
        await send_event(
            websocket,
            {
                "type": "asr_filtered",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "audio_path": saved_path,
                "duration_seconds": duration_seconds,
                "asr_text": asr_text,
                "asr_seconds": asr_seconds,
                "reason": drop_reason,
                "vad_engine": vad.name,
                "asr_engine": args.asr_engine,
            },
            "ASR_FILTERED",
        )
        return True

    await send_asr_result(
        websocket,
        utterance_index=utterance_index,
        audio_path=saved_path,
        duration_seconds=duration_seconds,
        asr_text=asr_text,
        asr_seconds=asr_seconds,
        vad_engine=vad.name,
        asr_engine=args.asr_engine,
    )

    if args.max_history_turns <= 0:
        history.clear()
    llm_messages = [*history, {"role": "user", "content": asr_text}]
    try:
        reply_text, llm_seconds = await asyncio.to_thread(run_llm_turn, args, llm_messages)
    except BaseException as exc:
        await send_event(
            websocket,
            {
                "type": "llm_error",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "asr_text": asr_text,
                "message": str(exc),
            },
            "LLM_ERROR",
        )
        return True
    if not reply_text:
        await send_event(
            websocket,
            {
                "type": "llm_filtered",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "asr_text": asr_text,
                "reason": "empty_llm_text",
            },
            "LLM_FILTERED",
        )
        return True

    history.extend(
        [
            {"role": "user", "content": asr_text},
            {"role": "assistant", "content": reply_text},
        ]
    )
    trim_history(history, args.max_history_turns)
    await send_event(
        websocket,
        {
            "type": "llm_result",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "asr_text": asr_text,
            "llm_text": reply_text,
            "llm_seconds": llm_seconds,
            "history_turns": len(history) // 2,
            "llm_model": args.llm_model,
        },
        "LLM_RESULT",
    )
    return True


async def handle_connection(
    websocket: Any,
    args: argparse.Namespace,
    vad: Any,
    asr_runtime: SherpaOnnxRuntimeASR,
) -> None:
    peer = getattr(websocket, "remote_address", None)
    print(f"Client connected: {peer}")
    streaming = False
    segmenter: UtteranceSegmenter | None = None
    frame_count = 0
    utterance_index = 0
    history: list[dict[str, str]] = []
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if not streaming or segmenter is None:
                    await websocket.send(json.dumps({"type": "stream_error", "message": "Received audio before stream_start."}))
                    continue
                frame_count += 1
                utterance = segmenter.accept(message)
                if utterance is not None:
                    utterance_index += 1
                    accepted = await process_utterance(
                        websocket, args, vad, asr_runtime, history, utterance, utterance_index
                    )
                    if not accepted:
                        utterance_index -= 1
                continue

            payload = parse_json_message(message)
            message_type = payload.get("type")
            if message_type == "stream_start":
                if payload.get("format") != "pcm_s16le":
                    await websocket.send(json.dumps({"type": "stream_error", "message": "Expected pcm_s16le."}))
                    continue
                if int(payload.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE or int(payload.get("channels", 0)) != 1:
                    await websocket.send(json.dumps({"type": "stream_error", "message": "Expected 16 kHz mono PCM16."}))
                    continue
                frame_ms = int(payload.get("frame_ms", args.frame_ms))
                segmenter = UtteranceSegmenter(
                    vad=vad,
                    frame_ms=frame_ms,
                    speech_start_ms=args.speech_start_ms,
                    speech_end_ms=args.speech_end_ms,
                    pre_speech_ms=args.pre_speech_ms,
                    min_utterance_ms=args.min_utterance_ms,
                    max_utterance_ms=args.max_utterance_ms,
                )
                streaming = True
                frame_count = 0
                utterance_index = 0
                history.clear()
                await send_event(
                    websocket,
                    {
                        "type": "stream_llm_ready",
                        "server_time": time.time(),
                        "vad_engine": vad.name,
                        "asr_engine": args.asr_engine,
                        "llm_model": args.llm_model,
                        "max_history_turns": args.max_history_turns,
                        "message": "Send PCM16 frames; valid utterances will be transcribed and sent to the LLM.",
                    },
                    "STREAM_LLM_READY",
                )
            elif message_type == "stream_stop":
                if streaming and segmenter is not None:
                    utterance = segmenter.flush()
                    if utterance is not None:
                        utterance_index += 1
                        accepted = await process_utterance(
                            websocket, args, vad, asr_runtime, history, utterance, utterance_index
                        )
                        if not accepted:
                            utterance_index -= 1
                await send_event(
                    websocket,
                    {
                        "type": "stream_llm_stopped",
                        "server_time": time.time(),
                        "frames_received": frame_count,
                        "utterances_processed": utterance_index,
                        "history_turns": len(history) // 2,
                        "client_stop": payload,
                    },
                    "STREAM_LLM_STOPPED",
                )
                streaming = False
                segmenter = None
            elif message_type == "ping":
                await websocket.send(json.dumps({"type": "pong", "server_time": time.time()}))
            else:
                await websocket.send(json.dumps({"type": "echo", "payload": payload, "server_time": time.time()}))
    except Exception as exc:
        print(f"Client disconnected: {peer}, reason: {exc}")
    else:
        print(f"Client disconnected: {peer}")


async def run_server(args: argparse.Namespace) -> None:
    try:
        websockets = __import__("websockets")
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: websockets\nInstall it with:\n  python -m pip install websockets") from exc

    vad = build_vad(args)
    asr_runtime = build_asr(args)

    async def handler(websocket: Any, path: str | None = None) -> None:
        await handle_connection(websocket, args, vad, asr_runtime)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        print(f"Auralis stream LLM server listening on ws://{args.host}:{args.port}")
        print(f"VAD engine: {vad.name}")
        print(f"ASR engine: {args.asr_engine} / {args.asr_model}")
        print(f"LLM: ollama / {args.llm_model}")
        print(f"Utterance wav files will be saved under: {args.output_dir}")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming VAD, ASR, and LLM over WebSocket.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--output-dir", default="outputs/ws_stream_llm_utterances")

    parser.add_argument("--vad-engine", choices=["energy", "funasr_fsmn", "silero", "webrtc"], default="silero")
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--speech-start-ms", type=int, default=200)
    parser.add_argument("--speech-end-ms", type=int, default=700)
    parser.add_argument("--pre-speech-ms", type=int, default=300)
    parser.add_argument("--min-utterance-ms", type=int, default=500)
    parser.add_argument("--max-utterance-ms", type=int, default=15000)
    parser.add_argument("--energy-threshold", type=float, default=0.012)
    parser.add_argument("--silero-threshold", type=float, default=0.5)
    parser.add_argument("--funasr-vad-model", default=str(DEFAULT_FUNASR_VAD_MODEL))
    parser.add_argument("--silero-vad-model", default=str(DEFAULT_SILERO_VAD_MODEL))
    parser.add_argument("--webrtc-aggressiveness", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--webrtc-frame-ms", type=int, choices=[10, 20, 30], default=20)

    parser.add_argument("--asr-engine", choices=["sherpa_onnx"], default="sherpa_onnx")
    parser.add_argument("--asr-model", default=asr.DEFAULT_SHERPA_ONNX_MODEL)
    parser.add_argument("--sherpa-model-type", choices=["sensevoice", "paraformer"], default="sensevoice")
    parser.add_argument("--text-script", choices=["simplified", "raw"], default="simplified")
    parser.add_argument("--keep-empty-asr", dest="drop_empty_asr", action="store_false")
    parser.set_defaults(drop_empty_asr=True)
    parser.add_argument("--min-asr-text-chars", type=int, default=2)
    parser.add_argument("--min-asr-duration-seconds", type=float, default=0.8)
    parser.add_argument(
        "--asr-noise-phrases",
        default="the,yeah,yes,no,um,uh,ah,er,嗯,呃,啊",
        help="Comma-separated normalized ASR texts to drop before the LLM stage.",
    )

    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--system-prompt", default=llm.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-history-turns", type=int, default=4)
    args = parser.parse_args()
    args.asr_noise_phrases = {
        normalize_for_noise_match(item)
        for item in str(args.asr_noise_phrases).split(",")
        if normalize_for_noise_match(item)
    }
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
