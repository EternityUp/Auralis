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
from auralis_lab.pipeline import add_tts_args
from auralis_lab.runtime import SherpaOnnxRuntimeASR
from auralis_lab.ws_stream_asr_server import (
    build_asr,
    normalize_for_noise_match,
    should_drop_asr_text,
    transcribe_utterance,
)
from auralis_lab.ws_stream_llm_server import run_llm_turn, trim_history
from auralis_lab.ws_stream_tts_server import StreamingTtsRuntime
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


BARGE_IN_CLASSIFIER_PROMPT = """You classify speech received while a voice assistant is responding.
Return exactly one lowercase token and nothing else:
- interrupt: the user is asking a question, making a request, correcting the assistant, or explicitly asking it to stop.
- continue: a brief acknowledgement such as yes, okay, mm-hmm, or thanks that should not stop the reply.
- ignore: noise, an invalid transcription, or speech not directed at the assistant.
Treat short explicit stop commands as interrupt.
"""


def reply_output_path(output_dir: str, utterance_index: int) -> str:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"reply-{utterance_index:03d}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.wav"
    return str(directory / filename)


def classify_barge_in(args: argparse.Namespace, asr_text: str, assistant_text: str) -> tuple[str, float, str]:
    prompt = (
        "Assistant response or state:\n"
        f"{assistant_text or '[The assistant is still preparing a response.]'}\n\n"
        "New user speech transcription:\n"
        f"{asr_text}\n\n"
        "Classification:"
    )
    start = time.perf_counter()
    raw = llm.run_ollama_messages(
        [{"role": "user", "content": prompt}],
        model=args.llm_model,
        host=args.llm_host,
        system_prompt=BARGE_IN_CLASSIFIER_PROMPT,
        temperature=0.0,
        top_p=1.0,
        num_predict=args.barge_in_num_predict,
        num_ctx=args.num_ctx,
        thinking="off",
    )
    normalized = raw.strip().lower()
    if "interrupt" in normalized:
        decision = "interrupt"
    elif "continue" in normalized:
        decision = "continue"
    else:
        decision = "ignore"
    return decision, time.perf_counter() - start, raw


class RealtimeSession:
    def __init__(
        self,
        websocket: Any,
        args: argparse.Namespace,
        vad: Any,
        asr_runtime: SherpaOnnxRuntimeASR,
        tts_runtime: StreamingTtsRuntime,
    ) -> None:
        self.websocket = websocket
        self.args = args
        self.vad = vad
        self.asr_runtime = asr_runtime
        self.tts_runtime = tts_runtime
        self.segmenter: UtteranceSegmenter | None = None
        self.streaming = False
        self.closed = False
        self.frame_count = 0
        self.utterance_index = 0
        self.history: list[dict[str, str]] = []
        self.last_assistant_text = ""
        self._response_token = 0
        self.active_response_token: int | None = None
        self.send_lock = asyncio.Lock()
        self.asr_lock = asyncio.Lock()
        self.llm_lock = asyncio.Lock()
        self.history_lock = asyncio.Lock()
        self.tasks: set[asyncio.Task[None]] = set()

    @property
    def assistant_active(self) -> bool:
        return self.active_response_token is not None

    async def send_event(self, payload: dict[str, Any], label: str) -> None:
        if self.closed:
            return
        print(f"{label}:", payload)
        async with self.send_lock:
            await self.websocket.send(json.dumps(payload, ensure_ascii=False))

    async def send_reply_audio(
        self,
        reply_path: str,
        utterance_index: int,
        reply_text: str,
        tts_seconds: float,
        response_token: int,
    ) -> None:
        path = Path(reply_path)
        if not path.exists():
            raise RuntimeError(f"TTS output was not generated: {reply_path}")
        audio_bytes = path.read_bytes()
        reply_id = f"reply-{utterance_index:03d}-{response_token}"
        async with self.send_lock:
            if self.closed or not self.is_response_current(response_token):
                return
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "tts_result",
                        "server_time": time.time(),
                        "utterance_index": utterance_index,
                        "tts_output": str(path),
                        "tts_seconds": tts_seconds,
                        "llm_text": reply_text,
                        "response_token": response_token,
                    },
                    ensure_ascii=False,
                )
            )
            metadata = {
                "type": "reply_audio",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "response_token": response_token,
                "reply_id": reply_id,
                "format": "wav",
                "suffix": ".wav",
                "filename": path.name,
                "bytes": len(audio_bytes),
                "tts_seconds": tts_seconds,
            }
            print("TTS_RESULT:", {"utterance_index": utterance_index, "tts_seconds": tts_seconds})
            print("REPLY_AUDIO:", metadata)
            await self.websocket.send(json.dumps(metadata, ensure_ascii=False))
            await self.websocket.send(audio_bytes)

    def begin_response(self) -> int:
        self._response_token += 1
        self.active_response_token = self._response_token
        return self._response_token

    def is_response_current(self, response_token: int) -> bool:
        return not self.closed and self.active_response_token == response_token

    def invalidate_response(self) -> int | None:
        previous = self.active_response_token
        self.active_response_token = None
        return previous

    def complete_playback(self, response_token: int) -> bool:
        if self.active_response_token != response_token:
            return False
        self.active_response_token = None
        return True

    def schedule(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)

        def report_result(completed: asyncio.Task[None]) -> None:
            self.tasks.discard(completed)
            if completed.cancelled():
                return
            try:
                completed.result()
            except Exception as exc:
                print(f"Realtime turn task failed: {exc}")

        task.add_done_callback(report_result)

    async def history_messages(self, user_text: str) -> list[dict[str, str]]:
        async with self.history_lock:
            return [*self.history, {"role": "user", "content": user_text}]

    async def append_history(self, user_text: str, assistant_text: str, response_token: int) -> bool:
        async with self.history_lock:
            if not self.is_response_current(response_token):
                return False
            self.history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            trim_history(self.history, self.args.max_history_turns)
            self.last_assistant_text = assistant_text
            return True

    async def history_turn_count(self) -> int:
        async with self.history_lock:
            return len(self.history) // 2

    async def close(self) -> None:
        self.closed = True
        self.invalidate_response()
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


async def send_asr_event(
    session: RealtimeSession,
    event_type: str,
    utterance_index: int,
    saved_path: str,
    duration_seconds: float,
    bytes_received: int,
    asr_text: str = "",
    asr_seconds: float = 0.0,
    reason: str = "",
) -> None:
    payload: dict[str, Any] = {
        "type": event_type,
        "server_time": time.time(),
        "utterance_index": utterance_index,
        "audio_path": saved_path,
        "duration_seconds": duration_seconds,
        "vad_engine": session.vad.name,
    }
    if event_type == "utterance_saved":
        payload["saved_path"] = saved_path
        payload["bytes_received"] = bytes_received
    else:
        payload.update(
            {
                "asr_text": asr_text,
                "asr_seconds": asr_seconds,
                "asr_engine": session.args.asr_engine,
            }
        )
        if reason:
            payload["reason"] = reason
    await session.send_event(payload, event_type.upper())


async def process_valid_user_turn(
    session: RealtimeSession,
    utterance_index: int,
    asr_text: str,
) -> None:
    response_token = session.begin_response()
    messages = await session.history_messages(asr_text)
    try:
        async with session.llm_lock:
            reply_text, llm_seconds = await asyncio.to_thread(run_llm_turn, session.args, messages)
    except BaseException as exc:
        if session.is_response_current(response_token):
            session.invalidate_response()
            await session.send_event(
                {
                    "type": "llm_error",
                    "server_time": time.time(),
                    "utterance_index": utterance_index,
                    "asr_text": asr_text,
                    "message": str(exc),
                },
                "LLM_ERROR",
            )
        return

    if not session.is_response_current(response_token):
        return
    if not reply_text:
        session.invalidate_response()
        await session.send_event(
            {
                "type": "llm_filtered",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "asr_text": asr_text,
                "reason": "empty_llm_text",
            },
            "LLM_FILTERED",
        )
        return
    if not await session.append_history(asr_text, reply_text, response_token):
        return
    await session.send_event(
        {
            "type": "llm_result",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "asr_text": asr_text,
            "llm_text": reply_text,
            "llm_seconds": llm_seconds,
            "history_turns": await session.history_turn_count(),
            "llm_model": session.args.llm_model,
            "response_token": response_token,
        },
        "LLM_RESULT",
    )

    reply_path = reply_output_path(session.args.reply_output_dir, utterance_index)
    try:
        tts_seconds = await asyncio.to_thread(session.tts_runtime.synthesize, reply_text, reply_path)
    except BaseException as exc:
        if session.is_response_current(response_token):
            session.invalidate_response()
            await session.send_event(
                {
                    "type": "tts_error",
                    "server_time": time.time(),
                    "utterance_index": utterance_index,
                    "llm_text": reply_text,
                    "message": str(exc),
                },
                "TTS_ERROR",
            )
        return
    if session.is_response_current(response_token):
        await session.send_reply_audio(reply_path, utterance_index, reply_text, tts_seconds, response_token)


async def process_utterance(session: RealtimeSession, utterance: list[bytes], utterance_index: int) -> None:
    if not vad_accepts_utterance(session.vad, utterance):
        return
    saved_path, duration_seconds, bytes_received = save_pcm16_wav(
        session.args.output_dir,
        f"utt-{utterance_index:03d}",
        utterance,
    )
    await send_asr_event(session, "utterance_saved", utterance_index, saved_path, duration_seconds, bytes_received)
    async with session.asr_lock:
        asr_text, asr_seconds = await asyncio.to_thread(transcribe_utterance, session.asr_runtime, saved_path)
    drop_asr, drop_reason = should_drop_asr_text(asr_text, duration_seconds, session.args)
    if drop_asr:
        await send_asr_event(
            session,
            "asr_filtered",
            utterance_index,
            saved_path,
            duration_seconds,
            bytes_received,
            asr_text,
            asr_seconds,
            drop_reason,
        )
        return
    await send_asr_event(
        session,
        "asr_result",
        utterance_index,
        saved_path,
        duration_seconds,
        bytes_received,
        asr_text,
        asr_seconds,
    )

    if not session.assistant_active:
        await process_valid_user_turn(session, utterance_index, asr_text)
        return

    active_token = session.active_response_token
    await session.send_event(
        {
            "type": "barge_in_candidate",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "asr_text": asr_text,
            "active_response_token": active_token,
        },
        "BARGE_IN_CANDIDATE",
    )
    try:
        async with session.llm_lock:
            decision, classifier_seconds, classifier_raw = await asyncio.to_thread(
                classify_barge_in,
                session.args,
                asr_text,
                session.last_assistant_text,
            )
    except BaseException as exc:
        await session.send_event(
            {
                "type": "barge_in_decision",
                "server_time": time.time(),
                "utterance_index": utterance_index,
                "asr_text": asr_text,
                "decision": "ignore",
                "reason": f"classifier_error: {exc}",
            },
            "BARGE_IN_DECISION",
        )
        return
    response_still_active = active_token is not None and session.active_response_token == active_token
    decision_reason = ""
    if decision == "interrupt" and not response_still_active:
        # Playback may naturally finish while the short classifier is running.
        # The utterance is then a normal next turn, not an interruption.
        decision = "continue"
        decision_reason = "assistant_reply_completed_before_classifier_decision"
    await session.send_event(
        {
            "type": "barge_in_decision",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "asr_text": asr_text,
            "decision": decision,
            "classifier_seconds": classifier_seconds,
            "classifier_raw": classifier_raw,
            "reason": decision_reason,
        },
        "BARGE_IN_DECISION",
    )
    if decision != "interrupt":
        if not session.assistant_active:
            await process_valid_user_turn(session, utterance_index, asr_text)
        return

    invalidated_token = session.invalidate_response()
    await session.send_event(
        {
            "type": "barge_in",
            "server_time": time.time(),
            "utterance_index": utterance_index,
            "asr_text": asr_text,
            "interrupted_response_token": invalidated_token,
            "message": "Validated user interruption. Stop local reply playback.",
        },
        "BARGE_IN",
    )
    await process_valid_user_turn(session, utterance_index, asr_text)


async def handle_connection(
    websocket: Any,
    args: argparse.Namespace,
    vad: Any,
    asr_runtime: SherpaOnnxRuntimeASR,
    tts_runtime: StreamingTtsRuntime,
) -> None:
    peer = getattr(websocket, "remote_address", None)
    print(f"Client connected: {peer}")
    session = RealtimeSession(websocket, args, vad, asr_runtime, tts_runtime)
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                if not session.streaming or session.segmenter is None:
                    await session.send_event({"type": "stream_error", "message": "Received audio before stream_start."}, "STREAM_ERROR")
                    continue
                session.frame_count += 1
                utterance = session.segmenter.accept(message)
                if utterance is not None:
                    session.utterance_index += 1
                    session.schedule(process_utterance(session, utterance, session.utterance_index))
                continue

            payload = parse_json_message(message)
            message_type = payload.get("type")
            if message_type == "stream_start":
                if payload.get("format") != "pcm_s16le":
                    await session.send_event({"type": "stream_error", "message": "Expected pcm_s16le."}, "STREAM_ERROR")
                    continue
                if int(payload.get("sample_rate", 0)) != AUDIO_SAMPLE_RATE or int(payload.get("channels", 0)) != 1:
                    await session.send_event({"type": "stream_error", "message": "Expected 16 kHz mono PCM16."}, "STREAM_ERROR")
                    continue
                frame_ms = int(payload.get("frame_ms", args.frame_ms))
                session.segmenter = UtteranceSegmenter(
                    vad=vad,
                    frame_ms=frame_ms,
                    speech_start_ms=args.speech_start_ms,
                    speech_end_ms=args.speech_end_ms,
                    pre_speech_ms=args.pre_speech_ms,
                    min_utterance_ms=args.min_utterance_ms,
                    max_utterance_ms=args.max_utterance_ms,
                )
                session.streaming = True
                await session.send_event(
                    {
                        "type": "stream_realtime_ready",
                        "server_time": time.time(),
                        "vad_engine": vad.name,
                        "asr_engine": args.asr_engine,
                        "llm_model": args.llm_model,
                        "tts_engine": args.tts_engine,
                        "cosy_mode": args.cosy_mode,
                        "message": "Send PCM16 continuously. Valid barge-in requires VAD, ASR, and LLM classification.",
                    },
                    "STREAM_REALTIME_READY",
                )
            elif message_type == "playback_started":
                print("PLAYBACK_STARTED:", payload)
            elif message_type == "playback_completed":
                response_token = int(payload.get("response_token", -1))
                completed = session.complete_playback(response_token)
                print(f"PLAYBACK_COMPLETED: response_token={response_token}, active_cleared={completed}")
            elif message_type == "barge_in_ack":
                print("BARGE_IN_ACK:", payload)
            elif message_type == "stream_stop":
                if session.streaming and session.segmenter is not None:
                    utterance = session.segmenter.flush()
                    if utterance is not None:
                        session.utterance_index += 1
                        session.schedule(process_utterance(session, utterance, session.utterance_index))
                history_turns = await session.history_turn_count()
                await session.send_event(
                    {
                        "type": "stream_realtime_stopped",
                        "server_time": time.time(),
                        "frames_received": session.frame_count,
                        "utterances_detected": session.utterance_index,
                        "history_turns": history_turns,
                        "client_stop": payload,
                    },
                    "STREAM_REALTIME_STOPPED",
                )
                break
            elif message_type == "ping":
                await session.send_event({"type": "pong", "server_time": time.time()}, "PONG")
            else:
                await session.send_event({"type": "echo", "payload": payload, "server_time": time.time()}, "ECHO")
    except Exception as exc:
        print(f"Client disconnected: {peer}, reason: {exc}")
    else:
        print(f"Client disconnected: {peer}")
    finally:
        await session.close()


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
        print(f"Auralis realtime barge-in server listening on ws://{args.host}:{args.port}")
        print(f"VAD: {vad.name}; ASR: {args.asr_engine}; LLM: {args.llm_model}; TTS: {args.tts_engine}/{args.cosy_mode}")
        print("Barge-in policy: VAD endpoint -> ASR filter -> LLM classifier -> interrupt event.")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuous streaming audio with LLM-confirmed barge-in.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--output-dir", default="outputs/ws_realtime_bargein_utterances")
    parser.add_argument("--reply-output-dir", default="outputs/ws_realtime_bargein_replies")
    parser.add_argument("--warmup-text", default="Auralis ready.")
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
    parser.add_argument("--asr-noise-phrases", default="the,yeah,yes,no,um,uh,ah,er")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--system-prompt", default=llm.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--barge-in-num-predict", type=int, default=8)
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
