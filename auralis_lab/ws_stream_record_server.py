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

from auralis_lab.common import AUDIO_SAMPLE_RATE


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
    if isinstance(parsed, dict):
        return parsed
    return {
        "type": "json",
        "payload": parsed,
    }


def save_pcm16_wav(output_dir: str, pcm_bytes: bytes) -> str:
    import wave

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"stream-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
    target = output_path / filename
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(AUDIO_SAMPLE_RATE)
        writer.writeframes(pcm_bytes)
    return str(target)


async def handle_connection(websocket: Any, output_dir: str) -> None:
    peer = getattr(websocket, "remote_address", None)
    print(f"Client connected: {peer}")
    streaming = False
    stream_meta: dict[str, Any] | None = None
    chunks: list[bytes] = []
    frame_count = 0
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if not streaming:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "stream_error",
                                "message": "Received binary audio before stream_start.",
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue
                chunks.append(message)
                frame_count += 1
                continue

            payload = parse_json_message(message)
            message_type = payload.get("type")
            if message_type == "ping":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "pong",
                            "server_time": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )
            elif message_type == "stream_start":
                if payload.get("format") != "pcm_s16le":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "stream_error",
                                "message": f"Unsupported stream format: {payload.get('format')}",
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue
                if int(payload.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE or int(payload.get("channels", 0)) != 1:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "stream_error",
                                "message": "Expected 16 kHz mono PCM16 stream.",
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue
                streaming = True
                stream_meta = payload
                chunks = []
                frame_count = 0
                await websocket.send(
                    json.dumps(
                        {
                            "type": "stream_ready",
                            "server_time": time.time(),
                            "message": "Send PCM16 frames as websocket binary messages.",
                        },
                        ensure_ascii=False,
                    )
                )
            elif message_type == "stream_stop":
                if not streaming:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "stream_error",
                                "message": "Received stream_stop before stream_start.",
                            },
                            ensure_ascii=False,
                        )
                    )
                    continue
                pcm_bytes = b"".join(chunks)
                saved_path = save_pcm16_wav(output_dir, pcm_bytes)
                duration_seconds = len(pcm_bytes) / 2 / AUDIO_SAMPLE_RATE
                ack = {
                    "type": "stream_saved",
                    "server_time": time.time(),
                    "saved_path": saved_path,
                    "frames_received": frame_count,
                    "bytes_received": len(pcm_bytes),
                    "duration_seconds": duration_seconds,
                    "metadata": stream_meta,
                    "client_stop": payload,
                }
                print("STREAM_SAVED:", ack)
                await websocket.send(json.dumps(ack, ensure_ascii=False))
                streaming = False
                stream_meta = None
                chunks = []
                frame_count = 0
            else:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "echo",
                            "server_time": time.time(),
                            "payload": payload,
                        },
                        ensure_ascii=False,
                    )
                )
    except Exception as exc:
        print(f"Client disconnected: {peer}, reason: {exc}")
    else:
        print(f"Client disconnected: {peer}")


async def run_server(host: str, port: int, output_dir: str) -> None:
    try:
        websockets = __import__("websockets")
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: websockets\nInstall it with:\n  python -m pip install websockets") from exc

    async def handler(websocket: Any, path: str | None = None) -> None:
        await handle_connection(websocket, output_dir)

    async with websockets.serve(handler, host, port, max_size=None):
        print(f"Auralis stream record server listening on ws://{host}:{port}")
        print(f"Stream wav files will be saved under: {output_dir}")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record client-streamed PCM16 frames as wav files.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-dir", default="outputs/ws_stream_records")
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port, args.output_dir))


if __name__ == "__main__":
    main()
