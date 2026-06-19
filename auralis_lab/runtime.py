from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab import asr, llm, tts
from auralis_lab.common import AUDIO_SAMPLE_RATE, ensure_parent, require_module


class SherpaOnnxRuntimeASR:
    def __init__(self, model: str, model_type: str, text_script: str) -> None:
        self.text_script = text_script
        sherpa_onnx = require_module("sherpa_onnx", "python -m pip install sherpa-onnx")
        model_dir = Path(asr.resolve_model_dir(model, "sherpa-onnx"))
        tokens = asr._first_existing_file(model_dir, ["tokens.txt"])

        if model_type == "sensevoice":
            model_file = asr._first_existing_file(model_dir, ["model.int8.onnx", "model.onnx"])
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_file),
                tokens=str(tokens),
                num_threads=4,
                sample_rate=AUDIO_SAMPLE_RATE,
                feature_dim=80,
                language="auto",
                use_itn=True,
            )
        elif model_type == "paraformer":
            model_file = asr._first_existing_file(model_dir, ["model.int8.onnx", "model.onnx"])
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(model_file),
                tokens=str(tokens),
                num_threads=4,
                sample_rate=AUDIO_SAMPLE_RATE,
                feature_dim=80,
            )
        else:
            raise SystemExit(f"Runtime currently supports sherpa model type sensevoice/paraformer, got: {model_type}")

    def transcribe(self, audio_path: str) -> str:
        samples, sample_rate = asr.read_mono_pcm16_wave(audio_path)
        if sample_rate != AUDIO_SAMPLE_RATE:
            raise SystemExit(
                f"sherpa-onnx input wav must be {AUDIO_SAMPLE_RATE} Hz for Auralis, got {sample_rate} Hz: {audio_path}"
            )
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.recognizer.decode_stream(stream)
        text = asr.normalize_asr_spacing(stream.result.text)
        return asr.normalize_text_script(text, self.text_script)


class CosyVoiceRuntimeTTS:
    def __init__(
        self,
        model: str,
        mode: str,
        speaker: str,
        prompt_audio: str | None,
        prompt_text: str,
        instruct_text: str,
        prompt_loader: str,
        device: str,
        fp16: bool,
    ) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.mode = mode
        self.speaker = speaker
        self.prompt_audio = prompt_audio
        self.prompt_text = prompt_text
        self.instruct_text = instruct_text
        self.prompt_loader = prompt_loader

        self.torch = require_module(
            "torch",
            "Install PyTorch first. For CUDA 12.8:\n"
            "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
        )
        require_module("torchaudio", "Install PyTorch and torchaudio first.")
        self.numpy = require_module("numpy", "python -m pip install numpy")
        self.soundfile = require_module("soundfile", "python -m pip install soundfile")
        cosyvoice_module = require_module(
            "cosyvoice.cli.cosyvoice",
            "Install CosyVoice from its official repository, then make sure the repository is on PYTHONPATH.",
        )

        model_path = Path(model)
        if not model_path.exists():
            raise SystemExit(f"CosyVoice model directory was not found: {model}")

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        model_cls = getattr(cosyvoice_module, "AutoModel")
        try:
            self.model = model_cls(
                model_dir=str(model_path),
                load_jit=False,
                load_trt=False,
                fp16=self.torch.cuda.is_available() and fp16,
            )
        except TypeError:
            self.model = model_cls(model_dir=str(model_path))

        if mode == "sft":
            available_spks = self.model.list_available_spks()
            if speaker not in available_spks:
                raise SystemExit(f"CosyVoice speaker was not found: {speaker}. Available: {available_spks}")
        elif mode in ("zero_shot", "instruct2"):
            if not prompt_audio:
                raise SystemExit(f"CosyVoice {mode} runtime requires --prompt-audio.")
        else:
            raise SystemExit(f"Unsupported CosyVoice runtime mode: {mode}")

    def synthesize(self, text: str, output: str) -> None:
        output_path = ensure_parent(output)
        if self.mode == "sft":
            result_iter = self.model.inference_sft(text, self.speaker, stream=False)
        elif self.mode == "zero_shot":
            if not self.prompt_text:
                raise SystemExit("CosyVoice zero_shot runtime requires --prompt-text.")
            prompt_speech = tts.resolve_cosyvoice_prompt_audio(
                self.prompt_audio or "",
                self.prompt_loader,
                self.torch,
            )
            result_iter = self.model.inference_zero_shot(text, self.prompt_text, prompt_speech, stream=False)
        elif self.mode == "instruct2":
            prompt_speech = tts.resolve_cosyvoice_prompt_audio(
                self.prompt_audio or "",
                self.prompt_loader,
                self.torch,
            )
            result_iter = self.model.inference_instruct2(text, self.instruct_text, prompt_speech, stream=False)
        else:
            raise SystemExit(f"Unsupported CosyVoice runtime mode: {self.mode}")

        audio_chunks = []
        for result in result_iter:
            speech = result["tts_speech"]
            audio = speech.detach().cpu().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()
            audio_chunks.append(audio)
        if not audio_chunks:
            raise SystemExit("CosyVoice did not return any audio.")

        if len(audio_chunks) == 1:
            merged_audio = audio_chunks[0]
        else:
            pause = self.numpy.zeros(int(self.model.sample_rate * 0.08), dtype=audio_chunks[0].dtype)
            merged = []
            for index, chunk in enumerate(audio_chunks):
                if index:
                    merged.append(pause)
                merged.append(chunk)
            merged_audio = self.numpy.concatenate(merged)
        edge_silence = self.numpy.zeros(int(self.model.sample_rate * 0.05), dtype=merged_audio.dtype)
        merged_audio = self.numpy.concatenate([edge_silence, merged_audio, edge_silence])
        self.soundfile.write(str(output_path), merged_audio, self.model.sample_rate)


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def run_llm(args: argparse.Namespace, user_text: str) -> str:
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


def output_path_for_turn(output_dir: str, turn_index: int) -> str:
    return str(Path(output_dir) / f"runtime-reply-{turn_index:03d}.wav")


def process_turn(
    args: argparse.Namespace,
    tts_runtime: CosyVoiceRuntimeTTS,
    asr_runtime: SherpaOnnxRuntimeASR | None,
    mode: str,
    value: str,
    turn_index: int,
) -> None:
    if mode == "text":
        user_text = value
        asr_seconds = 0.0
    elif mode == "audio":
        if asr_runtime is None:
            raise SystemExit("Audio turn requires ASR runtime.")
        user_text, asr_seconds = timed(asr_runtime.transcribe, value)
    else:
        raise SystemExit(f"Unsupported turn mode: {mode}")

    if not user_text:
        raise SystemExit("Runtime input text is empty.")

    reply_text, llm_seconds = timed(run_llm, args, user_text)
    if not reply_text:
        raise SystemExit("LLM returned empty text.")

    output = output_path_for_turn(args.output_dir, turn_index)
    _, tts_seconds = timed(tts_runtime.synthesize, reply_text, output)
    total_seconds = asr_seconds + llm_seconds + tts_seconds

    print(f"TURN: {turn_index}")
    print(f"PIPELINE_MODE: {mode}")
    if mode == "audio":
        print("AUDIO_INPUT:")
        print(value)
        print("ASR_TEXT:")
        print(user_text)
    else:
        print("USER_TEXT:")
        print(user_text)
    print("LLM_TEXT:")
    print(reply_text)
    print(f"TTS_OUTPUT: {output}")
    print("LATENCY:")
    if mode == "audio":
        print(f"  asr_seconds: {asr_seconds:.3f}")
    print(f"  llm_seconds: {llm_seconds:.3f}")
    print(f"  tts_seconds: {tts_seconds:.3f}")
    print(f"  total_seconds: {total_seconds:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Auralis as a persistent interactive runtime.")
    parser.add_argument("--output-dir", default="outputs/runtime")
    parser.add_argument("--asr-engine", choices=["sherpa_onnx"], default="sherpa_onnx")
    parser.add_argument("--asr-model", default=asr.DEFAULT_SHERPA_ONNX_MODEL)
    parser.add_argument("--sherpa-model-type", choices=["sensevoice", "paraformer"], default="sensevoice")
    parser.add_argument("--text-script", choices=["simplified", "raw"], default="simplified")

    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--system-prompt", default=llm.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=4096)

    parser.add_argument("--tts-engine", choices=["cosyvoice"], default="cosyvoice")
    parser.add_argument("--tts-model", default=tts.DEFAULT_COSYVOICE_SFT_MODEL)
    parser.add_argument("--cosy-mode", choices=["sft", "zero_shot", "instruct2"], default="sft")
    parser.add_argument("--speaker", default="中文女")
    parser.add_argument("--prompt-audio", default=None)
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--instruct-text", default="用自然流畅的普通话朗读。<|endofprompt|>")
    parser.add_argument("--cosy-prompt-loader", choices=["path", "soundfile"], default="path")
    parser.add_argument("--cosy-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-cosy-fp16", action="store_true")
    parser.add_argument("--warmup-text", default="你好。")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading ASR runtime...")
    asr_runtime = SherpaOnnxRuntimeASR(
        model=args.asr_model,
        model_type=args.sherpa_model_type,
        text_script=args.text_script,
    )
    print("Loading TTS runtime...")
    tts_runtime = CosyVoiceRuntimeTTS(
        model=args.tts_model,
        mode=args.cosy_mode,
        speaker=args.speaker,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        instruct_text=args.instruct_text,
        prompt_loader=args.cosy_prompt_loader,
        device=args.cosy_device,
        fp16=not args.no_cosy_fp16,
    )

    if not args.no_warmup:
        print("Warming up LLM/TTS...")
        _ = run_llm(args, args.warmup_text)
        warmup_output = str(Path(args.output_dir) / "_warmup.wav")
        tts_runtime.synthesize(args.warmup_text, warmup_output)

    print("Auralis runtime is ready.")
    print("Commands:")
    print("  text <user text>")
    print("  audio <wav path>")
    print("  quit")

    turn_index = 1
    while True:
        try:
            command = input("auralis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not command:
            continue
        if command in ("quit", "exit", "q"):
            break
        if command.startswith("text "):
            process_turn(args, tts_runtime, asr_runtime, "text", command[5:].strip(), turn_index)
            turn_index += 1
        elif command.startswith("audio "):
            process_turn(args, tts_runtime, asr_runtime, "audio", command[6:].strip(), turn_index)
            turn_index += 1
        else:
            print("Unknown command. Use: text <user text>, audio <wav path>, or quit.")


if __name__ == "__main__":
    main()
