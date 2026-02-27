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

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        base_url = _env("OFFICETOOL_OPENAI_BASE_URL", "OFFCIATOOL_OPENAI_BASE_URL", "OPENAI_BASE_URL")
        ca_cert = _env("OFFICETOOL_CA_CERT_PATH", "OFFCIATOOL_CA_CERT_PATH", "SSL_CERT_FILE")
        return cls(
            asr_model=_env("RT_WHISPER_MODEL", default="large-v3"),
            device=_env("RT_DEVICE", default="auto"),
            compute_type=_env("RT_COMPUTE_TYPE", default="int8"),
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
        )


class TranslatorClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.api_key)
        if not self.enabled:
            self.client = None
            return

        kwargs: dict[str, object] = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.client = OpenAI(**kwargs)

        if settings.ca_cert_path:
            os.environ.setdefault("SSL_CERT_FILE", settings.ca_cert_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", settings.ca_cert_path)

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

        if self.settings.use_responses_api:
            response = self.client.responses.create(
                model=self.settings.translation_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            translated = (response.output_text or "").strip()
            return translated or text

        completion = self.client.chat.completions.create(
            model=self.settings.translation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        message = completion.choices[0].message.content
        translated = (message or "").strip()
        return translated or text


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
        self.model = WhisperModel(
            settings.asr_model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
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
            beam_size=5,
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
            f"- ASR model: {self.settings.asr_model}\n"
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
    parser.add_argument("--asr-model", default=None, help="faster-whisper model name, e.g. medium, large-v3")
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
