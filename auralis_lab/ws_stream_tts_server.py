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

from auralis_lab import asr, llm, tts
from auralis_lab.pipeline import add_tts_args
from auralis_lab.runtime import CosyVoiceRuntimeTTS, SherpaOnnxRuntimeASR
from auralis_lab.ws_stream_asr_server import (
    build_asr,
    normalize_for_noise_match,
    should_drop_asr_text,
    transcribe_utterance,
)
from auralis_lab.ws_stream_llm_server import run_llm_turn, trim_history
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


class StreamingTtsRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.tts_engine != "cosyvoice":
            raise SystemExit("Streaming TTS server currently supports --tts-engine cosyvoice only.")
        default_model = tts.DEFAULT_COSYVOICE_SFT_MODEL if args.cosy_mode == "sft" else tts.DEFAULT_COSYVOICE_MODEL
        self.runtime = CosyVoiceRuntimeTTS(
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
        self.lock = Lock()

    def synthesize(self, text: str, output_wav: str) -> float:
        start = time.perf_counter()
        with self.lock:
            self.runtime.synthesize(text, output_wav)
        return time.perf_counter() - start


async def send_event(websocket: Any, payload: dict[str, Any], label: str) -> None:
    print(f"{label}:", payload)
    await websocket.send(json.dumps(payload, ensure_ascii=False))


async def send_reply_audio(
    websocket: Any,
    reply_path: str,
    utterance_index: int,
    reply_text: str,
    tts_seconds: float,
) -> None:
    path = Path(reply_path)
    if not path.exists():
        raise RuntimeError(f"TTS output was not generated: {reply_path}")
    audio_bytes = path.read_bytes()
    await send_event(
        websocket,
        {
            "type": "tts_result",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "tts_output": str(path),
            "tts_seconds": tts_seconds,
            "llm_text": reply_text,
        },
        "TTS_RESULT",
    )
    metadata = {
        "type": "reply_audio",
        "server_time": time.time(),
        "utterance_index": utterance_index,
        "format": "wav",
        "suffix": ".wav",
        "filename": path.name,
        "bytes": len(audio_bytes),
        "tts_seconds": tts_seconds,
    }
    print("REPLY_AUDIO:", metadata)
    await websocket.send(json.dumps(metadata, ensure_ascii=False))
    await websocket.send(audio_bytes)


def reply_output_path(output_dir: str, utterance_index: int) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"reply-{utterance_index:03d}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
    return str(output_path / filename)


async def process_utterance(
    websocket: Any,
    args: argparse.Namespace,
    vad: Any,
    asr_runtime: SherpaOnnxRuntimeASR,
    tts_runtime: StreamingTtsRuntime,
    history: list[dict[str, str]],
    utterance: list[bytes],
    utterance_index: int,
) -> bool:
    if not vad_accepts_utterance(vad, utterance):
        return False

    # The client switches to half-duplex as soon as an utterance is accepted.
    # This prevents microphone frames captured during LLM/TTS work from
    # accumulating and being interpreted as the next user turn.
    await send_event(
        websocket,
        {
            "type": "turn_started",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "message": "Utterance accepted; pause microphone capture until reply playback completes.",
        },
        "TURN_STARTED",
    )

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

    await send_event(
        websocket,
        {
            "type": "asr_result",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "audio_path": saved_path,
            "duration_seconds": duration_seconds,
            "asr_text": asr_text,
            "asr_seconds": asr_seconds,
            "vad_engine": vad.name,
            "asr_engine": args.asr_engine,
        },
        "ASR_RESULT",
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

    reply_path = reply_output_path(args.reply_output_dir, utterance_index)
    try:
        tts_seconds = await asyncio.to_thread(tts_runtime.synthesize, reply_text, reply_path)
        await send_reply_audio(websocket, reply_path, utterance_index, reply_text, tts_seconds)
    except BaseException as exc:
        await send_event(
            websocket,
            {
                "type": "tts_error",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "llm_text": reply_text,
                "message": str(exc),
            },
            "TTS_ERROR",
        )
    return True


async def handle_connection(
    websocket: Any,
    args: argparse.Namespace,
    vad: Any,
    asr_runtime: SherpaOnnxRuntimeASR,
    tts_runtime: StreamingTtsRuntime,
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
                        websocket,
                        args,
                        vad,
                        asr_runtime,
                        tts_runtime,
                        history,
                        utterance,
                        utterance_index,
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
                        "type": "stream_tts_ready",
                        "server_time": time.time(),
                        "vad_engine": vad.name,
                        "asr_engine": args.asr_engine,
                        "llm_model": args.llm_model,
                        "tts_engine": args.tts_engine,
                        "cosy_mode": args.cosy_mode,
                        "max_history_turns": args.max_history_turns,
                        "message": "Send PCM16 frames; valid utterances will receive synthesized reply audio.",
                    },
                    "STREAM_TTS_READY",
                )
            elif message_type == "stream_stop":
                if streaming and segmenter is not None:
                    utterance = segmenter.flush()
                    if utterance is not None:
                        utterance_index += 1
                        accepted = await process_utterance(
                            websocket,
                            args,
                            vad,
                            asr_runtime,
                            tts_runtime,
                            history,
                            utterance,
                            utterance_index,
                        )
                        if not accepted:
                            utterance_index -= 1
                await send_event(
                    websocket,
                    {
                        "type": "stream_tts_stopped",
                        "server_time": time.time(),
                        "frames_received": frame_count,
                        "utterances_processed": utterance_index,
                        "history_turns": len(history) // 2,
                        "client_stop": payload,
                    },
                    "STREAM_TTS_STOPPED",
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
    print("Loading persistent TTS runtime...")
    tts_runtime = StreamingTtsRuntime(args)
    if not args.no_warmup:
        warmup_path = str(Path(args.reply_output_dir) / "_warmup.wav")
        print("Warming up persistent TTS runtime...")
        tts_runtime.synthesize(args.warmup_text, warmup_path)

    async def handler(websocket: Any, path: str | None = None) -> None:
        await handle_connection(websocket, args, vad, asr_runtime, tts_runtime)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        print(f"Auralis stream TTS server listening on ws://{args.host}:{args.port}")
        print(f"VAD engine: {vad.name}")
        print(f"ASR engine: {args.asr_engine} / {args.asr_model}")
        print(f"LLM: ollama / {args.llm_model}")
        print(f"TTS: {args.tts_engine} / {args.cosy_mode}")
        print(f"Utterance wav files will be saved under: {args.output_dir}")
        print(f"Reply wav files will be saved under: {args.reply_output_dir}")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming VAD, ASR, LLM, and TTS over WebSocket.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--output-dir", default="outputs/ws_stream_tts_utterances")
    parser.add_argument("--reply-output-dir", default="outputs/ws_stream_tts_replies")
    parser.add_argument("--warmup-text", default="你好。")
    parser.add_argument("--no-warmup", action="store_true")

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
    add_tts_args(parser)
    args = parser.parse_args()
    args.asr_noise_phrases = {
        normalize_for_noise_match(item)
        for item in str(args.asr_noise_phrases).split(",")
        if normalize_for_noise_match(item)
    }
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
