from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any


def parse_message(message: str | bytes) -> dict[str, Any]:
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


def save_uploaded_audio(output_dir: str, audio_bytes: bytes, suffix: str = ".wav") -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"upload-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}{suffix}"
    target = output_path / filename
    target.write_bytes(audio_bytes)
    return str(target)


def reply_audio_metadata(reply_wav: str) -> dict[str, Any]:
    path = Path(reply_wav)
    return {
        "type": "reply_audio",
        "server_time": time.time(),
        "format": "wav",
        "suffix": ".wav",
        "filename": path.name,
        "bytes": path.stat().st_size,
    }


async def handle_connection(
    websocket: Any,
    path: str | None = None,
    output_dir: str = "outputs/ws_uploads",
    reply_wav: str | None = None,
) -> None:
    peer = getattr(websocket, "remote_address", None)
    print(f"Client connected: {peer}")
    pending_audio_meta: dict[str, Any] | None = None
    try:
        async for message in websocket:
            if isinstance(message, bytes) and pending_audio_meta is not None:
                saved_path = save_uploaded_audio(
                    output_dir,
                    message,
                    suffix=pending_audio_meta.get("suffix", ".wav"),
                )
                response = {
                    "type": "audio_upload_ack",
                    "server_time": time.time(),
                    "saved_path": saved_path,
                    "bytes": len(message),
                    "metadata": pending_audio_meta,
                }
                pending_audio_meta = None
                await websocket.send(json.dumps(response, ensure_ascii=False))
                if reply_wav:
                    reply_path = Path(reply_wav)
                    if not reply_path.exists():
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "reply_audio_error",
                                    "server_time": time.time(),
                                    "message": f"Reply wav was not found: {reply_wav}",
                                },
                                ensure_ascii=False,
                            )
                        )
                    else:
                        await websocket.send(json.dumps(reply_audio_metadata(reply_wav), ensure_ascii=False))
                        await websocket.send(reply_path.read_bytes())
                continue

            payload = parse_message(message)
            message_type = payload.get("type")
            if message_type == "ping":
                response = {
                    "type": "pong",
                    "server_time": time.time(),
                    "client_time": payload.get("client_time"),
                    "message": "Auralis server websocket is reachable.",
                }
            elif message_type == "audio_upload":
                pending_audio_meta = payload
                response = {
                    "type": "ready_for_audio",
                    "server_time": time.time(),
                    "message": "Send audio bytes as the next websocket message.",
                }
            elif message_type == "bytes":
                response = {
                    "type": "bytes_ack",
                    "server_time": time.time(),
                    "size": payload["size"],
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


async def run_server(host: str, port: int, output_dir: str, reply_wav: str | None) -> None:
    try:
        websockets = __import__("websockets")
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: websockets\nInstall it with:\n  python -m pip install websockets") from exc

    async def handler(websocket: Any, path: str | None = None) -> None:
        await handle_connection(websocket, path, output_dir=output_dir, reply_wav=reply_wav)

    async with websockets.serve(handler, host, port, max_size=None):
        print(f"Auralis WebSocket ping server listening on ws://{host}:{port}")
        print("Use ws://<server-ip>:8765 from the Windows client.")
        print(f"Uploaded audio will be saved under: {output_dir}")
        if reply_wav:
            print(f"Reply wav simulation enabled: {reply_wav}")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal Auralis WebSocket ping/pong server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", default="outputs/ws_uploads")
    parser.add_argument("--reply-wav", default=None, help="Optional wav file to send back after each audio upload.")
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port, args.output_dir, args.reply_wav))


if __name__ == "__main__":
    main()
