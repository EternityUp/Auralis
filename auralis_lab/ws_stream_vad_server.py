from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab.common import AUDIO_SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUNASR_VAD_MODEL = REPO_ROOT / "models" / "vad" / "funasr-fsmn-vad"
DEFAULT_SILERO_VAD_MODEL = REPO_ROOT / "models" / "vad" / "silero-vad"


class VadEngine(Protocol):
    name: str

    def is_speech(self, pcm16: np.ndarray) -> bool:
        ...


@dataclass
class EnergyVad:
    threshold: float = 0.012
    name: str = "energy"

    def is_speech(self, pcm16: np.ndarray) -> bool:
        if pcm16.size == 0:
            return False
        audio = pcm16.astype("float32") / 32768.0
        rms = math.sqrt(float(np.mean(audio * audio)))
        return rms >= self.threshold


class FunasrFsmnVad:
    name = "funasr_fsmn"

    def __init__(self, model: str, threshold: float) -> None:
        model_path = Path(model)
        if not model_path.exists():
            raise SystemExit(
                "FunASR FSMN-VAD model directory was not found.\n\n"
                f"Expected local model path:\n  {model_path}\n\n"
                "Download it with:\n"
                f"  modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch --local_dir {model_path}\n"
            )
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: funasr\nInstall it with:\n  python -m pip install funasr modelscope") from exc
        self.model = AutoModel(model=str(model_path), disable_update=True)
        self.energy_fallback = EnergyVad(threshold=threshold)

    def is_speech(self, pcm16: np.ndarray) -> bool:
        # FunASR's FSMN-VAD is primarily used after an utterance buffer is formed in
        # this prototype. Frame-level endpointing keeps a deterministic energy gate
        # so the streaming protocol can be validated independent of model latency.
        return self.energy_fallback.is_speech(pcm16)

    def has_speech_in_utterance(self, pcm16: np.ndarray) -> bool:
        if pcm16.size == 0:
            return False
        try:
            result = self.model.generate(input=pcm16.astype("float32") / 32768.0, fs=AUDIO_SAMPLE_RATE)
        except Exception:
            return True
        return bool(result and result[0].get("value"))


class SileroVad:
    name = "silero"

    def __init__(self, model: str, threshold: float) -> None:
        model_path = Path(model)
        if not model_path.exists():
            raise SystemExit(
                "Silero VAD local repository was not found.\n\n"
                f"Expected local model/repo path:\n  {model_path}\n\n"
                "Download it with:\n"
                f"  git clone https://github.com/snakers4/silero-vad {model_path}\n"
            )
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: torch\nInstall PyTorch before using --vad-engine silero.") from exc
        self.torch = torch
        try:
            model, _ = torch.hub.load(
                repo_or_dir=str(model_path),
                model="silero_vad",
                source="local",
                trust_repo=True,
            )
        except Exception as exc:
            raise SystemExit(
                "Unable to load Silero VAD from the local repository. "
                "Check that models/vad/silero-vad is a complete snakers4/silero-vad checkout."
            ) from exc
        self.model = model
        self.threshold = threshold
        self.window_samples = 512
        self.pending = np.zeros(0, dtype="int16")

    def is_speech(self, pcm16: np.ndarray) -> bool:
        if pcm16.size == 0:
            return False
        audio = np.concatenate([self.pending, pcm16.astype("int16", copy=False)])
        usable = audio.size - (audio.size % self.window_samples)
        if usable <= 0:
            self.pending = audio
            return False
        windows = audio[:usable].reshape(-1, self.window_samples)
        self.pending = audio[usable:]

        speech_windows = 0
        with self.torch.no_grad():
            for window in windows:
                tensor = self.torch.from_numpy(window.astype("float32") / 32768.0)
                score = float(self.model(tensor, AUDIO_SAMPLE_RATE).item())
                if score >= self.threshold:
                    speech_windows += 1
        return speech_windows > 0


class WebRtcVad:
    name = "webrtc"

    def __init__(self, aggressiveness: int, frame_ms: int) -> None:
        try:
            import webrtcvad
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: webrtcvad\nInstall it with:\n  python -m pip install webrtcvad") from exc
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.samples_per_frame = AUDIO_SAMPLE_RATE * frame_ms // 1000
        self.bytes_per_frame = self.samples_per_frame * 2

    def is_speech(self, pcm16: np.ndarray) -> bool:
        if pcm16.size < self.samples_per_frame:
            return False
        pcm_bytes = pcm16.astype("<i2", copy=False).tobytes()
        voiced = 0
        total = 0
        for start in range(0, len(pcm_bytes) - self.bytes_per_frame + 1, self.bytes_per_frame):
            total += 1
            if self.vad.is_speech(pcm_bytes[start : start + self.bytes_per_frame], AUDIO_SAMPLE_RATE):
                voiced += 1
        return total > 0 and voiced / total >= 0.5


@dataclass
class UtteranceSegmenter:
    vad: VadEngine
    frame_ms: int
    speech_start_ms: int
    speech_end_ms: int
    pre_speech_ms: int
    min_utterance_ms: int
    max_utterance_ms: int

    def __post_init__(self) -> None:
        self.pre_roll_frames = max(0, self.pre_speech_ms // self.frame_ms)
        self.start_frames = max(1, self.speech_start_ms // self.frame_ms)
        self.end_frames = max(1, self.speech_end_ms // self.frame_ms)
        self.min_frames = max(1, self.min_utterance_ms // self.frame_ms)
        self.max_frames = max(self.min_frames, self.max_utterance_ms // self.frame_ms)
        self.pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self.current: list[bytes] = []
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0

    def accept(self, frame: bytes) -> list[bytes] | None:
        pcm16 = np.frombuffer(frame, dtype="<i2")
        speech = self.vad.is_speech(pcm16)
        if not self.in_speech:
            if speech:
                self.speech_run += 1
            else:
                self.speech_run = 0
            self.pre_roll.append(frame)
            if self.speech_run >= self.start_frames:
                self.in_speech = True
                self.current = list(self.pre_roll)
                self.pre_roll.clear()
                self.silence_run = 0
            return None

        self.current.append(frame)
        if speech:
            self.silence_run = 0
        else:
            self.silence_run += 1

        too_long = len(self.current) >= self.max_frames
        ended = self.silence_run >= self.end_frames
        if not too_long and not ended:
            return None

        utterance = self.current
        self.current = []
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.pre_roll.clear()
        if len(utterance) < self.min_frames:
            return None
        return utterance

    def flush(self) -> list[bytes] | None:
        if not self.current:
            return None
        utterance = self.current
        self.current = []
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.pre_roll.clear()
        if len(utterance) < self.min_frames:
            return None
        return utterance


def parse_json_message(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        return {"type": "bytes", "size": len(message)}
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        return {"type": message}
    return parsed if isinstance(parsed, dict) else {"type": "json", "payload": parsed}


def save_pcm16_wav(output_dir: str, prefix: str, frames: list[bytes]) -> tuple[str, float, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pcm_bytes = b"".join(frames)
    filename = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
    target = output_path / filename
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(AUDIO_SAMPLE_RATE)
        writer.writeframes(pcm_bytes)
    duration_seconds = len(pcm_bytes) / 2 / AUDIO_SAMPLE_RATE
    return str(target), duration_seconds, len(pcm_bytes)


def vad_accepts_utterance(vad: VadEngine, frames: list[bytes]) -> bool:
    checker = getattr(vad, "has_speech_in_utterance", None)
    if checker is None:
        return True
    pcm16 = np.frombuffer(b"".join(frames), dtype="<i2")
    return bool(checker(pcm16))


def build_vad(args: argparse.Namespace) -> VadEngine:
    if args.vad_engine == "energy":
        return EnergyVad(threshold=args.energy_threshold)
    if args.vad_engine == "funasr_fsmn":
        return FunasrFsmnVad(model=args.funasr_vad_model, threshold=args.energy_threshold)
    if args.vad_engine == "silero":
        return SileroVad(model=args.silero_vad_model, threshold=args.silero_threshold)
    if args.vad_engine == "webrtc":
        return WebRtcVad(aggressiveness=args.webrtc_aggressiveness, frame_ms=args.webrtc_frame_ms)
    raise ValueError(f"Unsupported VAD engine: {args.vad_engine}")


async def send_utterance_ack(
    websocket: Any,
    saved_path: str,
    duration_seconds: float,
    bytes_received: int,
    utterance_index: int,
    vad_engine: str,
) -> None:
    payload = {
        "type": "utterance_saved",
        "server_time": time.time(),
        "utterance_index": utterance_index,
        "saved_path": saved_path,
        "duration_seconds": duration_seconds,
        "bytes_received": bytes_received,
        "vad_engine": vad_engine,
    }
    print("UTTERANCE_SAVED:", payload)
    await websocket.send(json.dumps(payload, ensure_ascii=False))


async def handle_connection(websocket: Any, args: argparse.Namespace, vad: VadEngine) -> None:
    peer = getattr(websocket, "remote_address", None)
    print(f"Client connected: {peer}")
    streaming = False
    segmenter: UtteranceSegmenter | None = None
    frame_count = 0
    utterance_index = 0
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if not streaming or segmenter is None:
                    await websocket.send(json.dumps({"type": "stream_error", "message": "Received audio before stream_start."}))
                    continue
                frame_count += 1
                utterance = segmenter.accept(message)
                if utterance is not None:
                    if not vad_accepts_utterance(vad, utterance):
                        continue
                    utterance_index += 1
                    saved_path, duration_seconds, bytes_received = save_pcm16_wav(
                        args.output_dir,
                        f"utt-{utterance_index:03d}",
                        utterance,
                    )
                    await send_utterance_ack(
                        websocket,
                        saved_path,
                        duration_seconds,
                        bytes_received,
                        utterance_index,
                        vad.name,
                    )
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
                await websocket.send(
                    json.dumps(
                        {
                            "type": "stream_vad_ready",
                            "server_time": time.time(),
                            "vad_engine": vad.name,
                            "message": "Send PCM16 frames; utterances will be saved when speech endpoints are detected.",
                        },
                        ensure_ascii=False,
                    )
                )
            elif message_type == "stream_stop":
                if streaming and segmenter is not None:
                    utterance = segmenter.flush()
                    if utterance is not None:
                        if not vad_accepts_utterance(vad, utterance):
                            utterance = None
                    if utterance is not None:
                        utterance_index += 1
                        saved_path, duration_seconds, bytes_received = save_pcm16_wav(
                            args.output_dir,
                            f"utt-{utterance_index:03d}",
                            utterance,
                        )
                        await send_utterance_ack(
                            websocket,
                            saved_path,
                            duration_seconds,
                            bytes_received,
                            utterance_index,
                            vad.name,
                        )
                ack = {
                    "type": "stream_vad_stopped",
                    "server_time": time.time(),
                    "frames_received": frame_count,
                    "utterances_saved": utterance_index,
                    "client_stop": payload,
                }
                await websocket.send(json.dumps(ack, ensure_ascii=False))
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

    async def handler(websocket: Any, path: str | None = None) -> None:
        await handle_connection(websocket, args, vad)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        print(f"Auralis stream VAD server listening on ws://{args.host}:{args.port}")
        print(f"VAD engine: {vad.name}")
        print(f"Utterance wav files will be saved under: {args.output_dir}")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment client-streamed PCM16 audio into utterance wav files.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--output-dir", default="outputs/ws_stream_utterances")
    parser.add_argument("--vad-engine", choices=["energy", "funasr_fsmn", "silero", "webrtc"], default="energy")
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
    args = parser.parse_args()

    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
