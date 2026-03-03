from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import numpy as np
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI


SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value.strip()
    return default


def _normalize_base_url(raw_url: str) -> str:
    url = raw_url.strip().strip("\"'").rstrip("/")
    if not url:
        return url
    parsed = urlparse(url)
    path = parsed.path or ""
    suffixes = ["/chat/completions", "/responses", "/v1/chat/completions", "/v1/responses"]
    lowered = path.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            path = path[: -len(suffix)] + ("/v1" if suffix.startswith("/v1/") else "")
            break
    return urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), parsed.params, parsed.query, parsed.fragment))


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    asr_model: str
    asr_model_path: str | None
    asr_download_root: str | None
    asr_local_files_only: bool
    asr_beam_size: int
    device: str
    compute_type: str
    source_language: str
    target_language: str
    translation_model: str
    api_key: str
    base_url: str | None
    ca_cert_path: str | None
    use_responses_api: bool
    vad_aggressiveness: int
    min_segment_ms: int
    glossary: str
    hf_endpoint: str | None
    hf_token: str | None

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        base_url = _env("OFFICETOOL_OPENAI_BASE_URL", "OFFCIATOOL_OPENAI_BASE_URL", "OPENAI_BASE_URL")
        ca_cert = _env("OFFICETOOL_CA_CERT_PATH", "OFFCIATOOL_CA_CERT_PATH", "SSL_CERT_FILE")
        asr_model_path = _env("RT_ASR_MODEL_DIR", "RT_WHISPER_MODEL_PATH")
        asr_download_root = _env("RT_ASR_HF_CACHE_DIR", "RT_HF_CACHE_DIR")
        return cls(
            asr_model=_env("RT_ASR_MODEL_NAME", "RT_WHISPER_MODEL", default="small"),
            asr_model_path=str(Path(asr_model_path).expanduser()) if asr_model_path else None,
            asr_download_root=str(Path(asr_download_root).expanduser()) if asr_download_root else None,
            asr_local_files_only=_truthy(_env("RT_ASR_HF_LOCAL_ONLY", "RT_HF_LOCAL_FILES_ONLY", default="false")),
            asr_beam_size=max(1, int(_env("RT_ASR_BEAM_SIZE", "RT_BEAM_SIZE", default="1"))),
            device=_env("RT_ASR_DEVICE", "RT_DEVICE", default="auto"),
            compute_type=_env("RT_ASR_COMPUTE_TYPE", "RT_COMPUTE_TYPE", default="int8"),
            source_language=_env("RT_SOURCE_LANGUAGE", default="zh"),
            target_language=_env("RT_TARGET_LANGUAGE", default="en"),
            translation_model=_env("RT_TRANSLATION_MODEL", default="gpt-5.1"),
            api_key=_env("OPENAI_API_KEY"),
            base_url=_normalize_base_url(base_url) if base_url else None,
            ca_cert_path=ca_cert or None,
            use_responses_api=_truthy(_env("OFFICETOOL_USE_RESPONSES_API", "OFFCIATOOL_USE_RESPONSES_API", default="false")),
            vad_aggressiveness=max(0, min(3, int(_env("RT_VAD_AGGRESSIVENESS", default="2")))),
            min_segment_ms=max(300, int(_env("RT_MIN_SEGMENT_MS", default="700"))),
            glossary=_env("RT_GLOSSARY", default=""),
            hf_endpoint=_env("RT_ASR_HF_ENDPOINT", "RT_HF_ENDPOINT", "HF_ENDPOINT") or None,
            hf_token=_env("RT_ASR_HF_TOKEN", "RT_HF_TOKEN", "HF_TOKEN") or None,
        )


def _configure_network_env(settings: Settings) -> None:
    if settings.ca_cert_path:
        # Apply corporate CA before any OpenAI/Hugging Face network calls.
        os.environ.setdefault("SSL_CERT_FILE", settings.ca_cert_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", settings.ca_cert_path)
    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)


def _resolve_asr_model_source(settings: Settings) -> str:
    if not settings.asr_model_path:
        return settings.asr_model
    model_path = Path(settings.asr_model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"RT_ASR_MODEL_DIR does not exist: {model_path}")
    return str(model_path)


def _build_model_load_error(settings: Settings, model_source: str, exc: Exception) -> RuntimeError:
    details = [
        f"Failed to load faster-whisper model: {model_source}",
        f"Original error: {exc}",
        "Hints:",
        "- If Hugging Face is blocked on your company network, set RT_ASR_MODEL_DIR to a local converted model directory.",
        "- If you need cached/offline mode, set RT_ASR_HF_LOCAL_ONLY=true and RT_ASR_HF_CACHE_DIR to your model cache.",
        "- If you need a mirror, set RT_ASR_HF_ENDPOINT.",
        "- If the company proxy uses a custom CA, set OFFICETOOL_CA_CERT_PATH.",
        "- RT_ASR_MODEL_DIR must point to a faster-whisper/CTranslate2 model directory, not the original OpenAI Whisper PyTorch files.",
    ]
    if settings.asr_model_path:
        details.append(f"- Current RT_ASR_MODEL_DIR: {settings.asr_model_path}")
    if settings.asr_download_root:
        details.append(f"- Current RT_ASR_HF_CACHE_DIR: {settings.asr_download_root}")
    return RuntimeError("\n".join(details))


class TranslatorClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.api_key)
        self._use_responses_api = settings.use_responses_api
        if not self.enabled:
            self.client = None
            return

        kwargs: dict[str, object] = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**kwargs)

    def translate(self, text: str) -> str:
        if not text.strip() or not self.enabled or self.client is None:
            return text

        system_prompt = (
            "You are a professional meeting interpreter. "
            f"Translate from {self.settings.source_language} to {self.settings.target_language}. "
            "Keep original meaning, names, and technical terms. "
            "Do not add explanations."
        )
        if self.settings.glossary:
            system_prompt += f" Apply this glossary strictly when relevant:\n{self.settings.glossary}"

        try:
            if self._use_responses_api:
                translated = self._translate_with_responses(system_prompt, text)
            else:
                translated = self._translate_with_chat(system_prompt, text)
            return translated or text
        except Exception as exc:
            if self._is_method_not_allowed(exc):
                fallback = not self._use_responses_api
                print(
                    f"[translator] 405 detected, retry with use_responses_api={str(fallback).lower()}",
                    file=sys.stderr,
                )
                self._use_responses_api = fallback
                if self._use_responses_api:
                    translated = self._translate_with_responses(system_prompt, text)
                else:
                    translated = self._translate_with_chat(system_prompt, text)
                return translated or text
            raise

    def _translate_with_responses(self, system_prompt: str, text: str) -> str:
        response = self.client.responses.create(
            model=self.settings.translation_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        return (response.output_text or "").strip()

    def _translate_with_chat(self, system_prompt: str, text: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.settings.translation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        message = completion.choices[0].message.content
        return (message or "").strip()

    @staticmethod
    def _is_method_not_allowed(exc: Exception) -> bool:
        text = str(exc).lower()
        return "405" in text or "method not allowed" in text


class Segmenter:
    def __init__(self, aggressiveness: int, min_segment_ms: int):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.ring_size = 10
        self.ring: deque[tuple[bytes, bool]] = deque(maxlen=self.ring_size)
        self.voiced_frames: list[bytes] = []
        self.triggered = False
        self.min_segment_frames = max(1, min_segment_ms // FRAME_MS)

    def consume(self, frame: bytes) -> Iterable[bytes]:
        if len(frame) != FRAME_BYTES:
            return []
        is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
        segments: list[bytes] = []

        if not self.triggered:
            self.ring.append((frame, is_speech))
            voiced_count = sum(1 for _, voiced in self.ring if voiced)
            if voiced_count >= int(0.8 * self.ring.maxlen):
                self.triggered = True
                self.voiced_frames.extend(chunk for chunk, _ in self.ring)
                self.ring.clear()
            return segments

        self.voiced_frames.append(frame)
        self.ring.append((frame, is_speech))
        unvoiced_count = sum(1 for _, voiced in self.ring if not voiced)
        if (
            unvoiced_count >= int(0.9 * self.ring.maxlen)
            and len(self.voiced_frames) >= self.min_segment_frames
        ):
            segments.append(b"".join(self.voiced_frames))
            self.voiced_frames = []
            self.ring.clear()
            self.triggered = False
        return segments

    def flush(self) -> bytes | None:
        if len(self.voiced_frames) >= self.min_segment_frames:
            segment = b"".join(self.voiced_frames)
            self.voiced_frames = []
            self.ring.clear()
            self.triggered = False
            return segment
        return None


class RealtimeMeetingTranslator:
    def __init__(self, settings: Settings):
        self.settings = settings
        _configure_network_env(settings)
        model_source = _resolve_asr_model_source(settings)
        try:
            model_kwargs: dict[str, object] = {
                "device": settings.device,
                "compute_type": settings.compute_type,
                "local_files_only": settings.asr_local_files_only,
            }
            if settings.asr_download_root:
                model_kwargs["download_root"] = settings.asr_download_root
            self.model = WhisperModel(model_source, **model_kwargs)
        except Exception as exc:
            raise _build_model_load_error(settings, model_source, exc) from exc
        self.translator = TranslatorClient(settings)
        self.segmenter = Segmenter(settings.vad_aggressiveness, settings.min_segment_ms)
        self.audio_queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=1200)
        self.stop_event = threading.Event()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[audio warning] {status}", file=sys.stderr)
        if self.stop_event.is_set():
            return
        data = bytes(indata)
        for i in range(0, len(data), FRAME_BYTES):
            frame = data[i : i + FRAME_BYTES]
            if len(frame) == FRAME_BYTES:
                try:
                    self.audio_queue.put_nowait(frame)
                except queue.Full:
                    # Drop old audio under pressure to keep realtime behavior.
                    _ = self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(frame)

    def _segment_to_text(self, pcm16le: bytes) -> str:
        pcm = np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            pcm,
            language=self.settings.source_language,
            beam_size=self.settings.asr_beam_size,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=True,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        return text

    def _process_segment(self, pcm16le: bytes) -> None:
        source_text = self._segment_to_text(pcm16le)
        if not source_text:
            return

        started = time.perf_counter()
        translated = self.translator.translate(source_text)
        spent_ms = int((time.perf_counter() - started) * 1000)

        now = time.strftime("%H:%M:%S")
        print(f"\n[{now}] {self.settings.source_language}> {source_text}")
        print(f"[{now}] {self.settings.target_language}> {translated} (translate {spent_ms} ms)")

    def run(self) -> None:
        print(
            "Realtime Meeting Translator started.\n"
            f"- ASR model: {_resolve_asr_model_source(self.settings)}\n"
            f"- ASR beam size: {self.settings.asr_beam_size}\n"
            f"- Source -> Target: {self.settings.source_language} -> {self.settings.target_language}\n"
            f"- Translation model: {self.settings.translation_model}\n"
            "Press Ctrl+C to stop.\n"
        )
        if not self.translator.enabled:
            print("OPENAI_API_KEY not set. Running ASR-only mode.\n")

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while not self.stop_event.is_set():
                try:
                    frame = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if frame is None:
                    break
                for segment in self.segmenter.consume(frame):
                    self._process_segment(segment)

        tail = self.segmenter.flush()
        if tail:
            self._process_segment(tail)

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.audio_queue.put_nowait(None)
        except queue.Full:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime meeting speech translation (local ASR + LLM translation)")
    parser.add_argument("--source-language", default=None, help="ASR source language, e.g. zh, en")
    parser.add_argument("--target-language", default=None, help="Target translation language, e.g. en, zh")
    parser.add_argument("--asr-model", default=None, help="faster-whisper model name, e.g. small, medium")
    parser.add_argument("--translation-model", default=None, help="LLM model for translation, e.g. gpt-5.1")
    return parser.parse_args()


def main() -> int:
    settings = Settings.load()
    args = parse_args()
    if args.source_language:
        settings.source_language = args.source_language
    if args.target_language:
        settings.target_language = args.target_language
    if args.asr_model:
        settings.asr_model = args.asr_model
    if args.translation_model:
        settings.translation_model = args.translation_model

    runner = RealtimeMeetingTranslator(settings)

    def _handle_signal(signum, frame):  # noqa: ANN001
        _ = (signum, frame)
        runner.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
