from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auralis_lab.common import ensure_parent, require_module


DEFAULT_PIPER_MODEL = "models/tts/piper/zh_CN-huayan-medium.onnx"
DEFAULT_COSYVOICE_MODEL = "models/tts/cosyvoice/CosyVoice2-0.5B"
DEFAULT_COSYVOICE_SFT_MODEL = "models/tts/cosyvoice/CosyVoice-300M-SFT"
DEFAULT_CHATTTS_MODEL = "models/tts/chattts"
DEFAULT_VOXCPM_MODEL = "models/tts/voxcpm/VoxCPM-0.5B"
DEFAULT_VOXCPM2_MODEL = "models/tts/voxcpm/VoxCPM2"


async def run_edge_tts(text: str, output: str, voice: str) -> None:
    edge_tts = require_module(
        "edge_tts",
        "python -m pip install -r requirements/tts.txt",
    )
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(ensure_parent(output)))


def run_piper(text: str, output: str, model: str, executable: str) -> None:
    model_path = Path(model)
    if not model_path.exists():
        raise SystemExit(
            "Piper model file was not found.\n\n"
            f"Expected local model path:\n  {model}\n\n"
            "Download a Piper voice model and place both the .onnx file and its .json config "
            "under models/tts/piper/."
        )
    output_path = ensure_parent(output)
    command = [
        executable,
        "--model",
        str(model_path),
        "--output_file",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            check=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Piper executable was not found.\n"
            "Install it with:\n  python -m pip install piper-tts\n"
            "Or pass the executable path with --executable."
        ) from exc


def run_cosyvoice(
    text: str,
    output: str,
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

    torch = require_module(
        "torch",
        "Install PyTorch first. For CUDA 12.8:\n"
        "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CosyVoice was asked to use CUDA, but torch.cuda.is_available() is False.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torchaudio = require_module(
        "torchaudio",
        "Install PyTorch and torchaudio first.",
    )
    soundfile = require_module(
        "soundfile",
        "python -m pip install soundfile",
    )
    cosyvoice_module = require_module(
        "cosyvoice.cli.cosyvoice",
        "Install CosyVoice from its official repository, then make sure the repository is on PYTHONPATH.",
    )

    model_path = Path(model)
    if not model_path.exists():
        raise SystemExit(
            "CosyVoice model directory was not found.\n\n"
            f"Expected local model path:\n  {model}\n\n"
            "Download a CosyVoice model to models/tts/cosyvoice/ before running this engine."
        )

    model_cls = getattr(cosyvoice_module, "AutoModel")
    try:
        tts_model = model_cls(
            model_dir=str(model_path),
            load_jit=False,
            load_trt=False,
            fp16=torch.cuda.is_available() and fp16,
        )
    except TypeError:
        tts_model = model_cls(model_dir=str(model_path))

    output_path = ensure_parent(output)
    if mode == "sft":
        available_spks = tts_model.list_available_spks()
        if speaker not in available_spks:
            raise SystemExit(
                f"CosyVoice speaker was not found: {speaker}\n\n"
                f"Available speakers:\n  {available_spks}\n\n"
                "Use a CosyVoice SFT model such as models/tts/cosyvoice/CosyVoice-300M-SFT, "
                "or switch to --cosy-mode zero_shot for CosyVoice2."
            )
        result_iter = tts_model.inference_sft(text, speaker, stream=False)
    elif mode == "zero_shot":
        if not prompt_audio or not prompt_text:
            raise SystemExit("--cosy-mode zero_shot requires --prompt-audio and --prompt-text.")
        prompt_speech = resolve_cosyvoice_prompt_audio(prompt_audio, prompt_loader, torch)
        result_iter = tts_model.inference_zero_shot(
            text,
            prompt_text,
            prompt_speech,
            stream=False,
        )
    elif mode == "instruct2":
        if not prompt_audio or not instruct_text:
            raise SystemExit("--cosy-mode instruct2 requires --prompt-audio and --instruct-text.")
        if not hasattr(tts_model, "inference_instruct2"):
            raise SystemExit("The selected CosyVoice model does not support inference_instruct2.")
        prompt_speech = resolve_cosyvoice_prompt_audio(prompt_audio, prompt_loader, torch)
        result_iter = tts_model.inference_instruct2(
            text,
            instruct_text,
            prompt_speech,
            stream=False,
        )
    else:
        raise SystemExit(f"Unsupported CosyVoice mode: {mode}")

    audio_chunks = []
    for result in result_iter:
        speech = result["tts_speech"]
        audio = speech.detach().cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        audio_chunks.append(audio)
    if not audio_chunks:
        raise SystemExit("CosyVoice did not return any audio.")

    numpy = require_module("numpy", "python -m pip install numpy")
    if len(audio_chunks) == 1:
        merged_audio = audio_chunks[0]
    else:
        pause = numpy.zeros(int(tts_model.sample_rate * 0.08), dtype=audio_chunks[0].dtype)
        merged = []
        for index, chunk in enumerate(audio_chunks):
            if index:
                merged.append(pause)
            merged.append(chunk)
        merged_audio = numpy.concatenate(merged)
    edge_silence = numpy.zeros(int(tts_model.sample_rate * 0.05), dtype=merged_audio.dtype)
    merged_audio = numpy.concatenate([edge_silence, merged_audio, edge_silence])
    soundfile.write(str(output_path), merged_audio, tts_model.sample_rate)


def resolve_cosyvoice_prompt_audio(audio_path: str, prompt_loader: str, torch_module: object) -> object:
    path = Path(audio_path)
    if not path.exists():
        raise SystemExit(f"Prompt audio was not found: {audio_path}")
    if prompt_loader == "path":
        return str(path)
    if prompt_loader == "soundfile":
        return load_prompt_audio_16k(audio_path, torch_module)
    raise SystemExit(f"Unsupported CosyVoice prompt loader: {prompt_loader}")


def load_prompt_audio_16k(audio_path: str, torch_module: object) -> object:
    numpy = require_module("numpy", "python -m pip install numpy")
    soundfile = require_module("soundfile", "python -m pip install soundfile")
    scipy_signal = require_module("scipy.signal", "python -m pip install scipy")

    path = Path(audio_path)
    if not path.exists():
        raise SystemExit(f"Prompt audio was not found: {audio_path}")

    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        target_len = int(round(len(audio) * 16000 / sample_rate))
        audio = scipy_signal.resample(audio, target_len).astype("float32")
    audio = numpy.asarray(audio, dtype="float32")
    return torch_module.from_numpy(audio).unsqueeze(0)


def run_chattts(text: str, output: str, model: str, refine_text: bool) -> None:
    numpy = require_module("numpy", "python -m pip install numpy")
    soundfile = require_module("soundfile", "python -m pip install soundfile")
    chattts = require_module(
        "ChatTTS",
        "Install ChatTTS from its official package or repository, then retry.",
    )

    chat = chattts.Chat()
    model_path = Path(model)
    if model_path.exists():
        try:
            chat.load(source="local", custom_path=str(model_path), compile=False)
        except TypeError:
            chat.load(source="local", custom_path=str(model_path))
    else:
        try:
            chat.load(compile=False)
        except TypeError:
            chat.load()

    try:
        wavs = chat.infer([text], skip_refine_text=not refine_text)
    except AttributeError as exc:
        if "DynamicCache" in str(exc):
            raise SystemExit(
                "ChatTTS is incompatible with the installed transformers DynamicCache API.\n"
                "This can happen during text refinement or audio-code generation.\n"
                "Install a ChatTTS/transformers version pair that is compatible."
            ) from exc
        raise
    except TypeError as exc:
        raise SystemExit(
            "This ChatTTS version does not accept the skip_refine_text argument used by Auralis.\n"
            "Please share the installed ChatTTS version or install a newer ChatTTS release."
        ) from exc
    if not wavs:
        raise SystemExit("ChatTTS did not return any audio.")

    wav = numpy.asarray(wavs[0])
    if wav.ndim > 1:
        wav = wav.squeeze()
    soundfile.write(str(ensure_parent(output)), wav, 24000)


def run_voxcpm(
    text: str,
    output: str,
    model: str,
    prompt_audio: str | None,
    prompt_text: str,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise: bool,
    device: str,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    numpy = require_module("numpy", "python -m pip install numpy")
    soundfile = require_module("soundfile", "python -m pip install soundfile")
    torch = require_module(
        "torch",
        "Install PyTorch first. For CUDA 12.8:\n"
        "  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("VoxCPM was asked to use CUDA, but torch.cuda.is_available() is False.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    voxcpm_module = require_module(
        "voxcpm",
        "python -m pip install voxcpm",
    )

    model_path = Path(model)
    if not model_path.exists():
        raise SystemExit(
            "VoxCPM model directory was not found.\n\n"
            f"Expected local model path:\n  {model}\n\n"
            "Download a VoxCPM model (e.g. openbmb/VoxCPM-0.5B or openbmb/VoxCPM2) "
            "to models/tts/voxcpm/ before running this engine."
        )

    voxcpm_cls = getattr(voxcpm_module, "VoxCPM")
    try:
        tts_model = voxcpm_cls.from_pretrained(str(model_path), load_denoiser=denoise)
    except TypeError:
        tts_model = voxcpm_cls.from_pretrained(str(model_path))

    if prompt_audio:
        prompt_path = Path(prompt_audio)
        if not prompt_path.exists():
            raise SystemExit(f"Prompt audio was not found: {prompt_audio}")
        prompt_audio = str(prompt_path)

    # VoxCPM-0.5B exposes prompt_wav_path/prompt_text. VoxCPM2 supports
    # reference_wav_path for controllable cloning and can use prompt_wav_path +
    # prompt_text together for ultimate cloning. Try VoxCPM2-style arguments
    # first and fall back to the VoxCPM-0.5B signature.
    common_kwargs = {
        "text": text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
    }
    if prompt_audio and prompt_text:
        try:
            wav = tts_model.generate(
                prompt_wav_path=prompt_audio,
                prompt_text=prompt_text,
                reference_wav_path=prompt_audio,
                **common_kwargs,
            )
        except TypeError:
            wav = tts_model.generate(
                prompt_wav_path=prompt_audio,
                prompt_text=prompt_text,
                normalize=normalize,
                denoise=denoise,
                retry_badcase=True,
                **common_kwargs,
            )
    elif prompt_audio:
        try:
            wav = tts_model.generate(
                reference_wav_path=prompt_audio,
                **common_kwargs,
            )
        except TypeError:
            wav = tts_model.generate(
                prompt_wav_path=prompt_audio,
                prompt_text=prompt_text or None,
                normalize=normalize,
                denoise=denoise,
                retry_badcase=True,
                **common_kwargs,
            )
    else:
        try:
            wav = tts_model.generate(**common_kwargs)
        except TypeError:
            wav = tts_model.generate(
                normalize=normalize,
                denoise=denoise,
                retry_badcase=True,
                **common_kwargs,
            )

    sample_rate = getattr(getattr(tts_model, "tts_model", None), "sample_rate", 16000)

    wav = numpy.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.squeeze()
    if wav.size == 0:
        raise SystemExit("VoxCPM did not return any audio.")
    soundfile.write(str(ensure_parent(output)), wav, int(sample_rate))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one TTS engine.")
    parser.add_argument(
        "--engine",
        choices=["edge_tts", "piper", "cosyvoice", "chattts", "voxcpm"],
        required=True,
    )
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="outputs/tts.wav")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--model", default=None)
    parser.add_argument("--speaker", default="中文女")
    parser.add_argument(
        "--cosy-mode",
        choices=["sft", "zero_shot", "instruct2"],
        default="sft",
        help="CosyVoice inference mode.",
    )
    parser.add_argument("--prompt-audio", default=None, help="Prompt/reference wav for CosyVoice or VoxCPM cloning.")
    parser.add_argument("--prompt-text", default="", help="Transcript of --prompt-audio for zero-shot or ultimate cloning.")
    parser.add_argument(
        "--cosy-prompt-loader",
        choices=["path", "soundfile"],
        default="path",
        help=(
            "How to pass CosyVoice prompt audio. Use 'path' for native CosyVoice/torchcodec loading; "
            "use 'soundfile' only for experimental local waveform loading."
        ),
    )
    parser.add_argument(
        "--cosy-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="CosyVoice device preference. Use cpu as a slow fallback when CUDA runs out of memory.",
    )
    parser.add_argument(
        "--no-cosy-fp16",
        action="store_true",
        help=(
            "Disable fp16 for CosyVoice. Use only for diagnosis; CosyVoice2 may require fp16/bfloat16-compatible "
            "execution and can fail with dtype mismatch when this is enabled."
        ),
    )
    parser.add_argument(
        "--instruct-text",
        default="用自然流畅的普通话朗读。",
        help=(
            "Instruction text for CosyVoice2 instruct2 mode. "
            "Must be a short Chinese-style directive (e.g. '用四川话说' or '语速慢一些'); "
            "the model was not trained on English system-prompt formats."
        ),
    )
    parser.add_argument(
        "--refine-text",
        action="store_true",
        help="Enable ChatTTS text refinement. Disabled by default to avoid transformers DynamicCache incompatibilities.",
    )
    parser.add_argument(
        "--voxcpm-variant",
        choices=["0.5b", "voxcpm2"],
        default="0.5b",
        help="Which VoxCPM model family to default to when --model is not given.",
    )
    parser.add_argument(
        "--voxcpm-cfg",
        type=float,
        default=2.0,
        help="VoxCPM cfg_value: LM guidance on LocDiT. Higher follows the prompt more closely.",
    )
    parser.add_argument(
        "--voxcpm-timesteps",
        type=int,
        default=10,
        help="VoxCPM LocDiT inference timesteps. Higher for better quality, lower for speed.",
    )
    parser.add_argument(
        "--no-voxcpm-normalize",
        action="store_true",
        help="Disable VoxCPM external text normalization (VoxCPM-0.5B only).",
    )
    parser.add_argument(
        "--voxcpm-denoise",
        action="store_true",
        help="Enable VoxCPM denoiser/prompt denoising. Disabled by default.",
    )
    parser.add_argument(
        "--voxcpm-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="VoxCPM device preference. Use cpu as a slow fallback when CUDA runs out of memory.",
    )
    parser.add_argument("--executable", default="piper")
    args = parser.parse_args()

    if args.engine == "edge_tts":
        asyncio.run(run_edge_tts(args.text, args.output, args.voice))
    elif args.engine == "piper":
        run_piper(
            args.text,
            args.output,
            model=args.model or DEFAULT_PIPER_MODEL,
            executable=args.executable,
        )
    elif args.engine == "cosyvoice":
        default_model = DEFAULT_COSYVOICE_SFT_MODEL if args.cosy_mode == "sft" else DEFAULT_COSYVOICE_MODEL
        run_cosyvoice(
            args.text,
            args.output,
            model=args.model or default_model,
            mode=args.cosy_mode,
            speaker=args.speaker,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text,
            instruct_text=args.instruct_text,
            prompt_loader=args.cosy_prompt_loader,
            device=args.cosy_device,
            fp16=not args.no_cosy_fp16,
        )
    elif args.engine == "chattts":
        run_chattts(
            args.text,
            args.output,
            model=args.model or DEFAULT_CHATTTS_MODEL,
            refine_text=args.refine_text,
        )
    elif args.engine == "voxcpm":
        default_model = DEFAULT_VOXCPM2_MODEL if args.voxcpm_variant == "voxcpm2" else DEFAULT_VOXCPM_MODEL
        run_voxcpm(
            args.text,
            args.output,
            model=args.model or default_model,
            prompt_audio=args.prompt_audio,
            prompt_text=args.prompt_text,
            cfg_value=args.voxcpm_cfg,
            inference_timesteps=args.voxcpm_timesteps,
            normalize=not args.no_voxcpm_normalize,
            denoise=args.voxcpm_denoise,
            device=args.voxcpm_device,
        )
    else:
        raise SystemExit(f"Unsupported TTS engine: {args.engine}")

    print(f"TTS_OUTPUT: {args.output}")


if __name__ == "__main__":
    main()
