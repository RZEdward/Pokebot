"""Offline speech-to-skill prototype using a small Vosk vocabulary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import shutil
import tempfile
import time
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hitmonlee_sim.commands import SkillDispatch, dispatch_intent, parse_transcript

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
SAMPLE_RATE = 16_000
COMMAND_GRAMMAR = (
    "stop",
    "halt",
    "freeze",
    "emergency stop",
    "stand",
    "stand up",
    "hold still",
    "balance",
    "walk forward",
    "move forward",
    "go forward",
    "walk backward",
    "move backward",
    "go backward",
    "turn left",
    "rotate left",
    "turn right",
    "rotate right",
    "do not stop",
    "do not stand",
    "do not walk forward",
    "don't stop",
    "don't stand",
    "don't walk forward",
)


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    confidence: float
    source: str


def _vosk() -> Any:
    try:
        import vosk
    except ImportError as exc:
        raise RuntimeError('Install the speech extra with: pip install -e ".[speech]"') from exc
    vosk.SetLogLevel(-1)
    return vosk


def _load_recognizer(model_dir: Path, sample_rate: int = SAMPLE_RATE) -> tuple[Any, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Vosk model not found at {model_dir}. Run `python -m hitmonlee_sim.voice --download-model`."
        )
    vosk = _vosk()
    model = vosk.Model(str(model_dir))
    grammar = json.dumps([*COMMAND_GRAMMAR, "[unk]"])
    recognizer = vosk.KaldiRecognizer(model, sample_rate, grammar)
    recognizer.SetWords(True)
    return model, recognizer


def _result_to_transcript(result_json: str, *, source: str) -> Transcript:
    result = json.loads(result_json)
    words = result.get("result", [])
    confidence = sum(float(word.get("conf", 0.0)) for word in words) / len(words) if words else 0.0
    return Transcript(str(result.get("text", "")).strip(), confidence, source)


def _combine_transcripts(results: list[dict[str, Any]], *, source: str) -> Transcript:
    texts = [str(result.get("text", "")).strip() for result in results]
    words = [word for result in results for word in result.get("result", [])]
    confidence = sum(float(word.get("conf", 0.0)) for word in words) / len(words) if words else 0.0
    return Transcript(" ".join(text for text in texts if text), confidence, source)


def transcribe_wav(audio_path: Path, model_dir: Path) -> Transcript:
    """Transcribe mono 16 kHz signed 16-bit PCM WAV locally with Vosk."""
    _model, recognizer = _load_recognizer(model_dir)
    decoded: list[dict[str, Any]] = []
    with wave.open(str(audio_path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getframerate() != SAMPLE_RATE
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError("Expected mono 16 kHz 16-bit PCM WAV audio.")
        while chunk := audio.readframes(4_000):
            if recognizer.AcceptWaveform(chunk):
                decoded.append(json.loads(recognizer.Result()))
    decoded.append(json.loads(recognizer.FinalResult()))
    return _combine_transcripts(decoded, source=str(audio_path))


def listen_once(
    model_dir: Path, *, timeout_seconds: float = 8.0, device: str | None = None
) -> Transcript:
    """Listen to the default microphone once, then return the final transcript."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and greater than zero")
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError('Install the speech extra with: pip install -e ".[speech]"') from exc

    _model, recognizer = _load_recognizer(model_dir)
    audio_blocks: queue.Queue[bytes] = queue.Queue(maxsize=32)

    def receive_audio(indata: bytes, frames: int, timing: Any, status: Any) -> None:
        del frames, timing, status
        try:
            audio_blocks.put_nowait(bytes(indata))
        except queue.Full:
            pass

    final_transcript = Transcript("", 0.0, "microphone")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=4_000,
        dtype="int16",
        channels=1,
        device=device,
        callback=receive_audio,
    ):
        print(f"Listening for a command for up to {timeout_seconds:g} seconds…")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                block = audio_blocks.get(timeout=0.25)
            except queue.Empty:
                continue
            if recognizer.AcceptWaveform(block):
                final_transcript = _result_to_transcript(recognizer.Result(), source="microphone")
                if final_transcript.text:
                    break
        if not final_transcript.text:
            final_transcript = _result_to_transcript(recognizer.FinalResult(), source="microphone")
    return final_transcript


def download_small_english_model(model_dir: Path) -> Path:
    """Download and extract the official compact US-English Vosk model."""
    model_dir = model_dir.resolve()
    if model_dir.exists():
        raise FileExistsError(f"Model destination already exists: {model_dir}")
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hitmonlee-vosk-", dir=model_dir.parent) as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / f"{MODEL_NAME}.zip"
        digest = hashlib.sha256()
        print(f"Downloading {MODEL_URL}", flush=True)
        with (
            urllib.request.urlopen(MODEL_URL, timeout=60) as response,
            archive_path.open("wb") as output,
        ):
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded_bytes = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                downloaded_bytes += len(chunk)
                output.flush()
                if downloaded_bytes % (10 * 1024 * 1024) < len(chunk):
                    progress = (
                        f"{downloaded_bytes / 1024 / 1024:.0f} MiB"
                        if not total_bytes
                        else f"{downloaded_bytes / total_bytes:.0%}"
                    )
                    print(f"Downloaded {progress}", flush=True)
        print(f"Downloaded archive SHA-256: {digest.hexdigest()}", flush=True)

        staging = temp_root / "extracted"
        staging.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                target = (staging / entry.filename).resolve()
                if staging.resolve() not in target.parents and target != staging.resolve():
                    raise ValueError(f"Unsafe path in model archive: {entry.filename}")
            archive.extractall(staging)

        extracted_model = staging / MODEL_NAME
        if not extracted_model.is_dir():
            raise ValueError(f"Expected {MODEL_NAME} in the downloaded archive")
        shutil.move(str(extracted_model), str(model_dir))

    provenance = (
        "Model: "
        + MODEL_NAME
        + "\nSource: "
        + MODEL_URL
        + "\nModel license: Apache-2.0 (per official Vosk model list)"
        + "\nArchive SHA-256: "
        + digest.hexdigest()
        + "\nDownloaded: "
        + "official model archive; see https://alphacephei.com/vosk/models.html\n"
    )
    (model_dir / "SOURCE.txt").write_text(provenance, encoding="utf-8")
    return model_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--download-model", action="store_true")
    operation.add_argument(
        "--audio", type=Path, help="Transcribe a mono 16 kHz 16-bit PCM WAV file"
    )
    operation.add_argument(
        "--listen", action="store_true", help="Listen to one microphone utterance"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("runs/asr") / MODEL_NAME,
        help="Local Vosk model directory",
    )
    parser.add_argument(
        "--timeout", type=float, default=8.0, help="Maximum microphone listening time"
    )
    parser.add_argument(
        "--device", default=None, help="Optional sounddevice microphone name or index"
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.75)
    return parser.parse_args()


def _print_result(transcript: Transcript, dispatch: SkillDispatch) -> None:
    print(
        json.dumps(
            {
                "transcript": transcript.text,
                "confidence": transcript.confidence,
                "source": transcript.source,
                "dispatch": {
                    "status": dispatch.status,
                    "skill": dispatch.skill,
                    "reason": dispatch.reason,
                },
            },
            indent=2,
        )
    )


def main() -> None:
    args = _parse_args()
    if args.download_model:
        print(f"Model ready at {download_small_english_model(args.model_dir)}")
        return

    transcript = (
        transcribe_wav(args.audio, args.model_dir)
        if args.audio is not None
        else listen_once(args.model_dir, timeout_seconds=args.timeout, device=args.device)
    )
    intent = parse_transcript(
        transcript.text,
        confidence=transcript.confidence,
        source=transcript.source,
    )
    dispatch = dispatch_intent(intent, minimum_confidence=args.minimum_confidence)
    _print_result(transcript, dispatch)


if __name__ == "__main__":
    main()
