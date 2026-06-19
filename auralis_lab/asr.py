from __future__ import annotations

import argparse
import re
import sys
import wave
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab.common import AUDIO_SAMPLE_RATE, require_file, require_module


DEFAULT_FASTER_WHISPER_MODEL = "models/asr/faster-whisper-small"
DEFAULT_FUNASR_MODEL = "models/asr/funasr-paraformer-zh"
DEFAULT_FUNASR_PUNC_MODEL = "models/asr/funasr-ct-punc"
DEFAULT_SENSEVOICE_MODEL = "models/asr/sensevoice-small"
DEFAULT_SHERPA_ONNX_MODEL = "models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
DEFAULT_CHINESE_INITIAL_PROMPT = (
    "Mandarin transcript in simplified Chinese with Chinese punctuation."
)


def resolve_faster_whisper_model(model: str, allow_download: bool) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    if allow_download:
        return model
    raise SystemExit(
        "faster-whisper model directory was not found.\n\n"
        f"Expected local model path:\n  {model}\n\n"
        "Download the model on a machine with working network access, then copy it to the server.\n"
        "Example download command:\n"
        "  hf download Systran/faster-whisper-small --local-dir models/asr/faster-whisper-small\n\n"
        "Then run:\n"
        "  python auralis_lab/asr.py --engine faster_whisper "
        "--model models/asr/faster-whisper-small --audio samples/asr/Tencent_test.wav\n\n"
        "If you explicitly want this script to download from Hugging Face, add --allow-download."
    )


def resolve_model_dir(model: str, label: str) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    raise SystemExit(
        f"{label} model directory was not found.\n\n"
        f"Expected local model path:\n  {model}\n\n"
        "Auralis keeps ASR models under models/asr/ by default."
    )


def resolve_optional_model_dir(model: str | None, label: str) -> str | None:
    if not model:
        return None
    return resolve_model_dir(model, label)


def run_faster_whisper(
    audio_path: str,
    model: str,
    device: str,
    language: str | None,
    allow_download: bool,
    initial_prompt: str | None,
) -> str:
    fw = require_module(
        "faster_whisper",
        "python -m pip install -r requirements/asr.txt",
    )
    model_ref = resolve_faster_whisper_model(model, allow_download=allow_download)
    whisper_model = fw.WhisperModel(model_ref, device=device)
    segments, info = whisper_model.transcribe(
        str(require_file(audio_path)),
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=True,
    )
    text = "".join(segment.text for segment in segments).strip()
    print(f"language={info.language} probability={info.language_probability:.3f}")
    return text


def normalize_text_script(text: str, script: str) -> str:
    if script == "raw":
        return text
    if script != "simplified":
        raise SystemExit(f"Unsupported text script: {script}")

    opencc = require_module(
        "opencc",
        "python -m pip install opencc-python-reimplemented",
    )
    converter = opencc.OpenCC("t2s")
    return converter.convert(text)


def normalize_asr_spacing(text: str) -> str:
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def run_funasr(audio_path: str, model: str, punc_model: str | None) -> str:
    funasr = require_module(
        "funasr",
        "python -m pip install funasr modelscope",
    )
    model_ref = resolve_model_dir(model, "FunASR")
    punc_ref = resolve_optional_model_dir(punc_model, "FunASR punctuation")
    kwargs = {"model": model_ref, "disable_update": True}
    if punc_ref:
        kwargs["punc_model"] = punc_ref
    auto_model = funasr.AutoModel(**kwargs)
    result = auto_model.generate(input=str(require_file(audio_path)))
    if isinstance(result, list) and result:
        return normalize_asr_spacing(str(result[0].get("text", "")).strip())
    return normalize_asr_spacing(str(result).strip())


def run_sensevoice(audio_path: str, model: str) -> str:
    funasr = require_module(
        "funasr",
        "python -m pip install funasr modelscope",
    )
    postprocess = require_module(
        "funasr.utils.postprocess_utils",
        "python -m pip install funasr modelscope",
    )
    model_ref = resolve_model_dir(model, "SenseVoice")
    auto_model = funasr.AutoModel(
        model=model_ref,
        trust_remote_code=False,
        disable_update=True,
    )
    result = auto_model.generate(
        input=str(require_file(audio_path)),
        language="auto",
        use_itn=True,
    )
    if isinstance(result, list) and result:
        text = str(result[0].get("text", "")).strip()
    else:
        text = str(result).strip()
    if hasattr(postprocess, "rich_transcription_postprocess"):
        text = postprocess.rich_transcription_postprocess(text)
    return normalize_asr_spacing(text)


def _first_existing_file(model_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = model_dir / candidate
        if path.exists():
            return path
    raise SystemExit(
        f"Unable to find any of these files under {model_dir}:\n  "
        + "\n  ".join(candidates)
    )


def read_mono_pcm16_wave(audio_path: str) -> tuple[object, int]:
    numpy = require_module(
        "numpy",
        "python -m pip install numpy",
    )
    path = require_file(audio_path)
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise SystemExit(
            f"sherpa-onnx wav reader only supports 16-bit PCM wav, got {sample_width * 8}-bit: {path}"
        )

    samples = numpy.frombuffer(frames, dtype=numpy.int16).astype(numpy.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def run_sherpa_onnx(audio_path: str, model: str, model_type: str) -> str:
    sherpa_onnx = require_module(
        "sherpa_onnx",
        "python -m pip install sherpa-onnx",
    )
    model_dir = Path(resolve_model_dir(model, "sherpa-onnx"))
    tokens = _first_existing_file(model_dir, ["tokens.txt"])

    if model_type == "sensevoice":
        model_file = _first_existing_file(model_dir, ["model.int8.onnx", "model.onnx"])
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_file),
            tokens=str(tokens),
            num_threads=4,
            sample_rate=AUDIO_SAMPLE_RATE,
            feature_dim=80,
            language="auto",
            use_itn=True,
        )
    elif model_type == "paraformer":
        model_file = _first_existing_file(model_dir, ["model.int8.onnx", "model.onnx"])
        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(model_file),
            tokens=str(tokens),
            num_threads=4,
            sample_rate=AUDIO_SAMPLE_RATE,
            feature_dim=80,
        )
    elif model_type == "zipformer_ctc":
        model_file = _first_existing_file(
            model_dir,
            ["model.int8.onnx", "model.onnx", "ctc-epoch-99-avg-1.int8.onnx"],
        )
        recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=str(model_file),
            tokens=str(tokens),
            num_threads=4,
            sample_rate=AUDIO_SAMPLE_RATE,
            feature_dim=80,
        )
    else:
        raise SystemExit(f"Unsupported sherpa-onnx model type: {model_type}")

    samples, sample_rate = read_mono_pcm16_wave(audio_path)
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise SystemExit(
            f"sherpa-onnx input wav must be {AUDIO_SAMPLE_RATE} Hz for Auralis, got {sample_rate} Hz: {audio_path}"
        )
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return normalize_asr_spacing(stream.result.text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Validate one ASR engine. Auralis uses {AUDIO_SAMPLE_RATE} Hz audio."
    )
    parser.add_argument(
        "--engine",
        choices=["faster_whisper", "funasr", "sensevoice", "sherpa_onnx"],
        required=True,
    )
    parser.add_argument(
        "--audio",
        required=True,
        help=f"Path to an audio file. Prefer {AUDIO_SAMPLE_RATE} Hz mono wav for Auralis.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Engine-specific local model path. Defaults: "
            f"faster-whisper -> {DEFAULT_FASTER_WHISPER_MODEL}; "
            f"funasr -> {DEFAULT_FUNASR_MODEL}; "
            f"sensevoice -> {DEFAULT_SENSEVOICE_MODEL}; "
            f"sherpa-onnx -> {DEFAULT_SHERPA_ONNX_MODEL}."
        ),
    )
    parser.add_argument(
        "--punc-model",
        default=DEFAULT_FUNASR_PUNC_MODEL,
        help=(
            "FunASR punctuation model path. Use an empty string to disable punctuation. "
            f"Default: {DEFAULT_FUNASR_PUNC_MODEL}."
        ),
    )
    parser.add_argument(
        "--sherpa-model-type",
        choices=["sensevoice", "paraformer", "zipformer_ctc"],
        default="sensevoice",
        help="Which sherpa-onnx offline recognizer type to use.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--language", default="zh", help="Language hint for faster-whisper.")
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_CHINESE_INITIAL_PROMPT,
        help="Initial prompt for faster-whisper decoding. Use an empty string to disable it.",
    )
    parser.add_argument(
        "--text-script",
        choices=["simplified", "raw"],
        default="simplified",
        help="Post-process ASR text script. Default converts Chinese text to simplified Chinese.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the engine to download a model if --model is not a local path.",
    )
    args = parser.parse_args()

    if args.engine == "faster_whisper":
        text = run_faster_whisper(
            args.audio,
            model=args.model or DEFAULT_FASTER_WHISPER_MODEL,
            device=args.device,
            language=args.language,
            allow_download=args.allow_download,
            initial_prompt=args.initial_prompt or None,
        )
    elif args.engine == "funasr":
        text = run_funasr(
            args.audio,
            model=args.model or DEFAULT_FUNASR_MODEL,
            punc_model=args.punc_model or None,
        )
    elif args.engine == "sensevoice":
        text = run_sensevoice(args.audio, model=args.model or DEFAULT_SENSEVOICE_MODEL)
    elif args.engine == "sherpa_onnx":
        text = run_sherpa_onnx(
            args.audio,
            model=args.model or DEFAULT_SHERPA_ONNX_MODEL,
            model_type=args.sherpa_model_type,
        )
    else:
        raise SystemExit(f"Unsupported ASR engine: {args.engine}")

    text = normalize_text_script(text, args.text_script)
    print("ASR_TEXT:")
    print(text)


if __name__ == "__main__":
    main()
