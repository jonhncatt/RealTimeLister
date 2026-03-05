from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import string
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse, urlunparse
import webbrowser

import numpy as np
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model as fw_download_model
from openai import OpenAI


SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional meeting interpreter.\n"
    "Translate from {source_language} to {target_language}.\n"
    "Speaker context: {speaker_label}.\n"
    "Keep original meaning, names, and technical terms.\n"
    "Do not add explanations.{glossary_block}"
)
TRANSLATION_PROMPT_PLACEHOLDERS = (
    "source_language",
    "target_language",
    "speaker_label",
    "speaker_id",
    "glossary",
    "glossary_block",
)


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


def _normalize_translation_prompt_template(raw: str) -> str:
    value = (raw or "").replace("\\n", "\n").strip()
    return value or DEFAULT_TRANSLATION_PROMPT_TEMPLATE


def _validate_translation_prompt_template(template: str) -> tuple[bool, str | None]:
    formatter = string.Formatter()
    try:
        fields = {
            name
            for _, name, _, _ in formatter.parse(template)
            if name is not None and name != ""
        }
    except ValueError as exc:
        return False, f"Prompt template format error: {exc}"

    unknown = sorted(name for name in fields if name not in TRANSLATION_PROMPT_PLACEHOLDERS)
    if unknown:
        allowed = ", ".join(f"{{{item}}}" for item in TRANSLATION_PROMPT_PLACEHOLDERS)
        unknown_text = ", ".join(f"{{{item}}}" for item in unknown)
        return False, f"Unsupported placeholder(s): {unknown_text}. Allowed placeholders: {allowed}"

    try:
        template.format(
            **{
                "source_language": "zh",
                "target_language": "en",
                "speaker_label": "Speaker 1",
                "speaker_id": "speaker-1",
                "glossary": "",
                "glossary_block": "",
            }
        )
    except Exception as exc:
        return False, f"Prompt template format error: {exc}"
    return True, None


@dataclass(slots=True)
class Settings:
    asr_model: str
    asr_model_path: str | None
    asr_download_root: str | None
    asr_local_files_only: bool
    asr_beam_size: int
    input_device: str
    device: str
    compute_type: str
    source_language: str
    target_language: str
    translation_model: str
    translation_prompt_template: str
    api_key: str
    base_url: str | None
    ca_cert_path: str | None
    use_responses_api: bool
    vad_aggressiveness: int
    min_segment_ms: int
    speaker_split_enabled: bool
    speaker_max_speakers: int
    speaker_match_threshold: float
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
        prompt_template = _normalize_translation_prompt_template(_env("RT_TRANSLATION_PROMPT_TEMPLATE", default=""))
        prompt_ok, prompt_error = _validate_translation_prompt_template(prompt_template)
        if not prompt_ok:
            print(
                f"[config] Invalid RT_TRANSLATION_PROMPT_TEMPLATE, fallback to default template. {prompt_error}",
                file=sys.stderr,
            )
            prompt_template = DEFAULT_TRANSLATION_PROMPT_TEMPLATE
        return cls(
            asr_model=_env("RT_ASR_MODEL_NAME", "RT_WHISPER_MODEL", default="small"),
            asr_model_path=str(Path(asr_model_path).expanduser()) if asr_model_path else None,
            asr_download_root=str(Path(asr_download_root).expanduser()) if asr_download_root else None,
            asr_local_files_only=_truthy(_env("RT_ASR_HF_LOCAL_ONLY", "RT_HF_LOCAL_FILES_ONLY", default="false")),
            asr_beam_size=max(1, int(_env("RT_ASR_BEAM_SIZE", "RT_BEAM_SIZE", default="1"))),
            input_device=_env("RT_AUDIO_INPUT_DEVICE", default="auto"),
            device=_env("RT_ASR_DEVICE", "RT_DEVICE", default="auto"),
            compute_type=_env("RT_ASR_COMPUTE_TYPE", "RT_COMPUTE_TYPE", default="int8"),
            source_language=_env("RT_SOURCE_LANGUAGE", default="zh"),
            target_language=_env("RT_TARGET_LANGUAGE", default="en"),
            translation_model=_env("RT_TRANSLATION_MODEL", default="gpt-5.1"),
            translation_prompt_template=prompt_template,
            api_key=_env("OPENAI_API_KEY"),
            base_url=_normalize_base_url(base_url) if base_url else None,
            ca_cert_path=ca_cert or None,
            use_responses_api=_truthy(_env("OFFICETOOL_USE_RESPONSES_API", "OFFCIATOOL_USE_RESPONSES_API", default="false")),
            vad_aggressiveness=max(0, min(3, int(_env("RT_VAD_AGGRESSIVENESS", default="2")))),
            min_segment_ms=max(300, int(_env("RT_MIN_SEGMENT_MS", default="700"))),
            speaker_split_enabled=_truthy(_env("RT_SPEAKER_SPLIT_ENABLED", default="true")),
            speaker_max_speakers=max(1, min(8, int(_env("RT_SPEAKER_MAX", default="4")))),
            speaker_match_threshold=max(0.02, min(1.0, float(_env("RT_SPEAKER_MATCH_THRESHOLD", default="0.12")))),
            glossary=_env("RT_GLOSSARY", default=""),
            hf_endpoint=_env("RT_ASR_HF_ENDPOINT", "RT_HF_ENDPOINT", "HF_ENDPOINT") or None,
            hf_token=_env("RT_ASR_HF_TOKEN", "RT_HF_TOKEN", "HF_TOKEN") or None,
        )


@dataclass(slots=True)
class TranscriptEntry:
    timestamp: str
    speaker_id: str
    speaker_label: str
    source_text: str
    translated_text: str
    translate_ms: int
    source_language: str
    target_language: str


@dataclass(slots=True)
class AsrModelStatus:
    level: str
    message: str


@dataclass(slots=True)
class AsrStrategy:
    key: str
    label: str


def _resolve_asr_strategy(settings: Settings) -> AsrStrategy:
    if settings.asr_model_path:
        return AsrStrategy(key="fixed_dir", label="Fixed Local Directory")
    if settings.asr_local_files_only:
        return AsrStrategy(key="offline_cache_only", label="Offline Cache Only")
    return AsrStrategy(key="online_auto", label="Model Name + Auto Download")


def _query_input_devices() -> tuple[list[dict[str, Any]], str | None, str | None]:
    try:
        devices = sd.query_devices()
        default_device = sd.default.device
    except Exception as exc:
        return [], None, f"Audio device query failed: {exc}"

    default_input_index: int | None = None
    if isinstance(default_device, (list, tuple)) and default_device:
        try:
            if int(default_device[0]) >= 0:
                default_input_index = int(default_device[0])
        except Exception:
            default_input_index = None
    elif isinstance(default_device, int) and default_device >= 0:
        default_input_index = default_device

    options: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        input_channels = int(device.get("max_input_channels") or 0)
        if input_channels <= 0:
            continue
        options.append(
            {
                "id": str(index),
                "name": str(device.get("name") or f"Input {index}"),
                "inputChannels": input_channels,
                "isDefault": default_input_index == index,
            }
        )
    return options, str(default_input_index) if default_input_index is not None else None, None


def _resolve_input_device(selection: str) -> tuple[int, str]:
    devices, default_input_id, error = _query_input_devices()
    if error:
        raise RuntimeError(error)
    if not devices:
        raise RuntimeError("No input device available. Connect a microphone or choose a different audio input device.")

    requested = (selection or "auto").strip()
    if not requested or requested.lower() == "auto":
        if default_input_id is not None:
            for item in devices:
                if item["id"] == default_input_id:
                    return int(item["id"]), item["name"]
        first = devices[0]
        return int(first["id"]), first["name"]

    if requested.lstrip("-").isdigit():
        for item in devices:
            if item["id"] == requested:
                return int(item["id"]), item["name"]
        raise RuntimeError(f"Audio input device {requested} is not available or has no input channels.")

    lowered = requested.casefold()
    for item in devices:
        if item["name"].casefold() == lowered:
            return int(item["id"]), item["name"]
    for item in devices:
        if lowered in item["name"].casefold():
            return int(item["id"]), item["name"]
    available = ", ".join(item["name"] for item in devices[:5])
    raise RuntimeError(f"Audio input device '{requested}' was not found. Available inputs: {available}")


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


def _inspect_asr_model_status(settings: Settings) -> AsrModelStatus:
    if settings.asr_model_path:
        model_path = Path(settings.asr_model_path).expanduser()
        if not model_path.exists():
            return AsrModelStatus(
                level="error",
                message=f"Configured local model directory does not exist: {model_path}",
            )
        required = ["config.json", "model.bin", "tokenizer.json"]
        missing = [name for name in required if not (model_path / name).exists()]
        if missing:
            return AsrModelStatus(
                level="error",
                message=(
                    f"Local model directory is incomplete: {model_path}. "
                    f"Missing files: {', '.join(missing)}"
                ),
            )
        return AsrModelStatus(
            level="ready",
            message=f"Using fixed local model directory: {model_path}",
        )

    if settings.asr_local_files_only:
        try:
            cached_path = fw_download_model(
                settings.asr_model,
                local_files_only=True,
                cache_dir=settings.asr_download_root,
            )
            return AsrModelStatus(
                level="ready",
                message=f"Using cached model for '{settings.asr_model}': {cached_path}",
            )
        except Exception:
            cache_hint = settings.asr_download_root or "default Hugging Face cache"
            return AsrModelStatus(
                level="error",
                message=(
                    f"Offline cache-only mode is enabled for '{settings.asr_model}', "
                    f"but the model was not found in {cache_hint}. "
                    "Set RT_ASR_MODEL_DIR to a copied local model directory or preload the cache."
                ),
            )

    cache_hint = settings.asr_download_root or "default Hugging Face cache"
    mirror_hint = " via RT_ASR_HF_ENDPOINT" if settings.hf_endpoint else ""
    return AsrModelStatus(
        level="info",
        message=(
            f"ASR will use model name '{settings.asr_model}'. "
            f"It will load from {cache_hint} first, and if missing it will try downloading from Hugging Face{mirror_hint}."
        ),
    )


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

    def translate(self, text: str, *, speaker_label: str = "Speaker 1", speaker_id: str = "speaker-1") -> str:
        if not text.strip() or not self.enabled or self.client is None:
            return text

        system_prompt = self._build_system_prompt(speaker_label=speaker_label, speaker_id=speaker_id)

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

    def _build_system_prompt(self, *, speaker_label: str, speaker_id: str) -> str:
        template = _normalize_translation_prompt_template(self.settings.translation_prompt_template)
        glossary_block = ""
        if self.settings.glossary:
            glossary_block = f"\nApply this glossary strictly when relevant:\n{self.settings.glossary}"
        try:
            return template.format(
                source_language=self.settings.source_language,
                target_language=self.settings.target_language,
                speaker_label=speaker_label,
                speaker_id=speaker_id,
                glossary=self.settings.glossary,
                glossary_block=glossary_block,
            )
        except Exception as exc:
            raise RuntimeError(
                "Invalid translation prompt template. "
                "Check placeholders in RT_TRANSLATION_PROMPT_TEMPLATE or the web prompt editor."
            ) from exc

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


@dataclass(slots=True)
class SpeakerProfile:
    speaker_id: str
    speaker_label: str
    centroid: np.ndarray
    seen_count: int


class PseudoSpeakerDiarizer:
    """Fast pseudo diarization using segment-level acoustic feature matching."""

    def __init__(self, enabled: bool, max_speakers: int, match_threshold: float):
        self.enabled = enabled
        self.max_speakers = max(1, max_speakers)
        self.match_threshold = match_threshold
        self._profiles: list[SpeakerProfile] = []
        self._last_index: int | None = None

    def assign(self, pcm16le: bytes) -> tuple[str, str]:
        if not self.enabled:
            return "speaker-1", "Speaker 1"
        features = self._extract_features(pcm16le)
        if features is None:
            if self._last_index is not None and self._last_index < len(self._profiles):
                profile = self._profiles[self._last_index]
                return profile.speaker_id, profile.speaker_label
            return "speaker-1", "Speaker 1"

        if not self._profiles:
            profile = self._create_profile(features)
            self._last_index = 0
            return profile.speaker_id, profile.speaker_label

        # Hysteresis: keep the previous speaker if still close enough.
        if self._last_index is not None and self._last_index < len(self._profiles):
            last_profile = self._profiles[self._last_index]
            last_distance = float(np.linalg.norm(last_profile.centroid - features))
            if last_distance <= self.match_threshold * 1.2:
                self._update_profile(last_profile, features)
                return last_profile.speaker_id, last_profile.speaker_label

        distances = [float(np.linalg.norm(profile.centroid - features)) for profile in self._profiles]
        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]

        if best_distance <= self.match_threshold or len(self._profiles) >= self.max_speakers:
            profile = self._profiles[best_index]
            self._update_profile(profile, features)
            self._last_index = best_index
            return profile.speaker_id, profile.speaker_label

        profile = self._create_profile(features)
        self._last_index = len(self._profiles) - 1
        return profile.speaker_id, profile.speaker_label

    @staticmethod
    def _extract_features(pcm16le: bytes) -> np.ndarray | None:
        pcm = np.frombuffer(pcm16le, dtype=np.int16)
        if pcm.size < 64:
            return None
        normalized = pcm.astype(np.float32) / 32768.0
        energy = float(np.sqrt(np.mean(np.square(normalized)) + 1e-9))
        abs_mean = float(np.mean(np.abs(normalized)))
        if normalized.size > 1:
            sign_changes = np.not_equal(np.signbit(normalized[1:]), np.signbit(normalized[:-1]))
            zcr = float(np.mean(sign_changes))
        else:
            zcr = 0.0
        return np.array([energy, abs_mean, zcr], dtype=np.float32)

    def _create_profile(self, features: np.ndarray) -> SpeakerProfile:
        speaker_number = len(self._profiles) + 1
        profile = SpeakerProfile(
            speaker_id=f"speaker-{speaker_number}",
            speaker_label=f"Speaker {speaker_number}",
            centroid=features.copy(),
            seen_count=1,
        )
        self._profiles.append(profile)
        return profile

    @staticmethod
    def _update_profile(profile: SpeakerProfile, features: np.ndarray) -> None:
        # Exponential moving average to keep profile stable while adapting to voice changes.
        profile.centroid = profile.centroid * 0.8 + features * 0.2
        profile.seen_count += 1


class RealtimeMeetingTranslator:
    def __init__(
        self,
        settings: Settings,
        *,
        on_result: Callable[[TranscriptEntry], None] | None = None,
        on_info: Callable[[str], None] | None = None,
        console_output: bool = True,
    ):
        self.settings = settings
        self.on_result = on_result
        self.on_info = on_info
        self.console_output = console_output
        self.input_device_index, self.input_device_name = _resolve_input_device(settings.input_device)
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
        self.speaker_diarizer = PseudoSpeakerDiarizer(
            enabled=settings.speaker_split_enabled,
            max_speakers=settings.speaker_max_speakers,
            match_threshold=settings.speaker_match_threshold,
        )
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
        speaker_id, speaker_label = self.speaker_diarizer.assign(pcm16le)
        source_text = self._segment_to_text(pcm16le)
        if not source_text:
            return

        started = time.perf_counter()
        translated = self.translator.translate(source_text, speaker_label=speaker_label, speaker_id=speaker_id)
        spent_ms = int((time.perf_counter() - started) * 1000)

        now = time.strftime("%H:%M:%S")
        entry = TranscriptEntry(
            timestamp=now,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            source_text=source_text,
            translated_text=translated,
            translate_ms=spent_ms,
            source_language=self.settings.source_language,
            target_language=self.settings.target_language,
        )
        if self.console_output:
            print(f"\n[{entry.timestamp}] {entry.speaker_label} {entry.source_language}> {entry.source_text}")
            print(
                f"[{entry.timestamp}] {entry.speaker_label} {entry.target_language}> "
                f"{entry.translated_text} (translate {entry.translate_ms} ms)"
            )
        if self.on_result:
            self.on_result(entry)

    def run(self) -> None:
        startup_message = (
            "Realtime Meeting Translator started.\n"
            f"- ASR model: {_resolve_asr_model_source(self.settings)}\n"
            f"- ASR beam size: {self.settings.asr_beam_size}\n"
            f"- Audio input: {self.input_device_name} ({self.input_device_index})\n"
            f"- Speaker split: {'on' if self.settings.speaker_split_enabled else 'off'} "
            f"(max={self.settings.speaker_max_speakers})\n"
            f"- Source -> Target: {self.settings.source_language} -> {self.settings.target_language}\n"
            f"- Translation model: {self.settings.translation_model}\n"
            "Press Ctrl+C to stop.\n"
        )
        if self.console_output:
            print(startup_message)
        if self.on_info:
            self.on_info(
                f"ASR ready: {_resolve_asr_model_source(self.settings)} | "
                f"{self.settings.source_language}->{self.settings.target_language} | "
                f"mic={self.input_device_name}"
            )
        if not self.translator.enabled:
            message = "OPENAI_API_KEY not set. Running ASR-only mode."
            if self.console_output:
                print(f"{message}\n")
            if self.on_info:
                self.on_info(message)

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            device=self.input_device_index,
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


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> "queue.Queue[dict[str, Any]]":
        subscriber: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue[dict[str, Any]]") -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    pass


class WebAppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.event_bus = EventBus()
        self._lock = threading.Lock()
        self._history: deque[TranscriptEntry] = deque(maxlen=120)
        self._runner: RealtimeMeetingTranslator | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._loading = False
        self._status_level = "idle"
        self._status_message = "Ready to start"
        self._last_info = ""
        self._last_error = ""
        self._stop_requested = False

    def subscribe(self) -> "queue.Queue[dict[str, Any]]":
        return self.event_bus.subscribe()

    def unsubscribe(self, subscriber: "queue.Queue[dict[str, Any]]") -> None:
        self.event_bus.unsubscribe(subscriber)

    def snapshot(self) -> dict[str, Any]:
        input_devices, default_input_id, input_error = _query_input_devices()
        model_status = _inspect_asr_model_status(self.settings)
        strategy = _resolve_asr_strategy(self.settings)
        selection = (self.settings.input_device or "auto").strip() or "auto"
        selected_label = "Auto"
        if selection.lower() != "auto":
            selected_label = selection
            for item in input_devices:
                if item["id"] == selection or item["name"] == selection:
                    selected_label = item["name"]
                    break
        else:
            if default_input_id is not None:
                for item in input_devices:
                    if item["id"] == default_input_id:
                        selected_label = f"Auto ({item['name']})"
                        break
            elif input_devices:
                selected_label = f"Auto ({input_devices[0]['name']})"
            elif input_error:
                selected_label = "Auto (unavailable)"
            else:
                selected_label = "Auto (no input device)"
        with self._lock:
            current = asdict(self._history[-1]) if self._history else None
            asr_source = self.settings.asr_model_path or self.settings.asr_model
            return {
                "running": self._running,
                "loading": self._loading,
                "statusLevel": self._status_level,
                "statusMessage": self._status_message,
                "lastInfo": self._last_info,
                "lastError": self._last_error,
                "sourceLanguage": self.settings.source_language,
                "targetLanguage": self.settings.target_language,
                "translationModel": self.settings.translation_model,
                "translationPromptTemplate": self.settings.translation_prompt_template,
                "translationPromptPlaceholders": list(TRANSLATION_PROMPT_PLACEHOLDERS),
                "translatorEnabled": bool(self.settings.api_key),
                "asrModelSource": asr_source,
                "asrResolutionMode": "directory" if self.settings.asr_model_path else "model_name",
                "asrStrategyKey": strategy.key,
                "asrStrategyName": strategy.label,
                "asrBeamSize": self.settings.asr_beam_size,
                "speakerSplitEnabled": self.settings.speaker_split_enabled,
                "speakerMaxSpeakers": self.settings.speaker_max_speakers,
                "speakerMatchThreshold": self.settings.speaker_match_threshold,
                "asrModelStatusLevel": model_status.level,
                "asrModelStatusMessage": model_status.message,
                "inputDevices": input_devices,
                "inputDevicesError": input_error,
                "selectedInputDevice": selection,
                "selectedInputDeviceLabel": selected_label,
                "history": [asdict(item) for item in self._history],
                "current": current,
            }

    def _publish(self, event: dict[str, Any]) -> None:
        self.event_bus.publish(event)

    def _publish_snapshot(self) -> None:
        self._publish({"type": "snapshot", "state": self.snapshot()})

    def _set_status(self, level: str, message: str) -> None:
        with self._lock:
            self._status_level = level
            self._status_message = message
            if level == "error":
                self._last_error = message
            elif level in {"running", "loading", "stopped", "idle"}:
                self._last_info = message
        self._publish({"type": "status", "level": level, "message": message, "timestamp": time.strftime("%H:%M:%S")})
        self._publish_snapshot()

    def _handle_result(self, entry: TranscriptEntry) -> None:
        with self._lock:
            self._history.append(entry)
        self._publish({"type": "transcript", "item": asdict(entry)})
        self._publish_snapshot()

    def _handle_info(self, message: str) -> None:
        with self._lock:
            self._last_info = message
        self._publish({"type": "info", "message": message, "timestamp": time.strftime("%H:%M:%S")})
        self._publish_snapshot()

    def start(self) -> tuple[bool, str]:
        model_status = _inspect_asr_model_status(self.settings)
        if model_status.level == "error":
            self._set_status("error", model_status.message)
            return False, model_status.message
        try:
            _resolve_input_device(self.settings.input_device)
        except Exception as exc:
            self._set_status("error", str(exc))
            return False, str(exc)
        with self._lock:
            if self._running or self._loading:
                return False, "Session is already running."
            self._loading = True
            self._stop_requested = False
            self._status_level = "loading"
            self._status_message = "Loading ASR model and preparing microphone..."
            thread = threading.Thread(target=self._run_session, name="realtime-lister-session", daemon=True)
            self._thread = thread
        self._publish_snapshot()
        thread.start()
        return True, "Session starting."

    def _run_session(self) -> None:
        try:
            runner = RealtimeMeetingTranslator(
                self.settings,
                on_result=self._handle_result,
                on_info=self._handle_info,
                console_output=False,
            )
            with self._lock:
                self._runner = runner
                self._running = True
                self._loading = False
                stop_requested = self._stop_requested
            if stop_requested:
                runner.stop()
                self._set_status("stopped", "Stop requested before microphone opened.")
            else:
                self._set_status("running", "Microphone live. Listening for speech...")
            runner.run()
            final_level = "stopped" if self._stop_requested else "idle"
            final_message = "Session stopped." if self._stop_requested else "Session finished."
            self._set_status(final_level, final_message)
        except Exception as exc:
            self._set_status("error", str(exc))
        finally:
            with self._lock:
                self._runner = None
                self._thread = None
                self._running = False
                self._loading = False
                self._stop_requested = False
            self._publish_snapshot()

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            runner = self._runner
            loading = self._loading
            running = self._running
            self._stop_requested = True
        if runner is not None:
            runner.stop()
            self._set_status("stopped", "Stopping microphone...")
            return True, "Stop requested."
        if loading or running:
            self._set_status("stopped", "Waiting for session to stop...")
            return True, "Stop requested."
        return False, "No active session."

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
        self._publish({"type": "cleared"})
        self._publish_snapshot()

    def set_input_device(self, value: str) -> tuple[bool, str]:
        normalized = (value or "auto").strip() or "auto"
        resolved_label = "auto"
        if normalized.lower() != "auto":
            try:
                _, resolved_label = _resolve_input_device(normalized)
            except Exception as exc:
                self._set_status("error", str(exc))
                return False, str(exc)
        with self._lock:
            if self._running or self._loading:
                return False, "Stop the current session before changing the input device."
            self.settings.input_device = normalized
        self._set_status("idle", f"Audio input set to {resolved_label}.")
        return True, "Audio input updated."

    def set_translation_prompt_template(self, value: str) -> tuple[bool, str]:
        normalized = _normalize_translation_prompt_template(value)
        ok, error = _validate_translation_prompt_template(normalized)
        if not ok:
            message = error or "Invalid translation prompt template."
            self._set_status("error", message)
            return False, message

        with self._lock:
            self.settings.translation_prompt_template = normalized
            self._last_info = "Translation prompt template updated."
        self._publish({"type": "info", "message": "Translation prompt template updated.", "timestamp": time.strftime("%H:%M:%S")})
        self._publish_snapshot()
        return True, "Translation prompt template updated."


class RealtimeHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], app_state: WebAppState):
        super().__init__(server_address, handler_class)
        self.app_state = app_state


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    return "application/octet-stream"


def _make_handler(static_dir: Path) -> type[BaseHTTPRequestHandler]:
    class RealtimeHandler(BaseHTTPRequestHandler):
        server: RealtimeHTTPServer

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._serve_static(static_dir / "index.html")
                return
            if path in {"/styles.css", "/app.js"}:
                self._serve_static(static_dir / path.lstrip("/"))
                return
            if path == "/api/state":
                self._send_json(self.server.app_state.snapshot())
                return
            if path == "/api/events":
                self._serve_events()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/start":
                ok, message = self.server.app_state.start()
                self._send_json({"ok": ok, "message": message}, status=HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
                return
            if path == "/api/stop":
                ok, message = self.server.app_state.stop()
                self._send_json({"ok": ok, "message": message}, status=HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
                return
            if path == "/api/clear":
                self.server.app_state.clear_history()
                self._send_json({"ok": True, "message": "History cleared."})
                return
            if path == "/api/device":
                payload = self._read_json()
                ok, message = self.server.app_state.set_input_device(str(payload.get("device") or "auto"))
                self._send_json({"ok": ok, "message": message}, status=HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
                return
            if path == "/api/translation-prompt":
                payload = self._read_json()
                ok, message = self.server.app_state.set_translation_prompt_template(str(payload.get("template") or ""))
                self._send_json({"ok": ok, "message": message}, status=HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_static(self, file_path: Path) -> None:
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _guess_content_type(file_path))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def _serve_events(self) -> None:
            subscriber = self.server.app_state.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 1000\n")
                self._write_event({"type": "snapshot", "state": self.server.app_state.snapshot()})
                while True:
                    try:
                        event = subscriber.get(timeout=12)
                        self._write_event(event)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                self.server.app_state.unsubscribe(subscriber)

        def _write_event(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return RealtimeHandler


def run_web_interface(settings: Settings, host: str, port: int, open_browser: bool) -> int:
    app_state = WebAppState(settings)
    handler_class = _make_handler(STATIC_DIR)
    server = RealtimeHTTPServer((host, port), handler_class, app_state)
    browser_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{port}"
    strategy = _resolve_asr_strategy(settings)
    print(
        "Realtime Meeting Translator Web UI\n"
        f"- URL: {url}\n"
        f"- ASR strategy: {strategy.label}\n"
        f"- ASR source: {settings.asr_model_path or settings.asr_model}\n"
        f"- Speaker split: {'on' if settings.speaker_split_enabled else 'off'} (max={settings.speaker_max_speakers})\n"
        f"- Translation model: {settings.translation_model}\n"
        "Use the browser controls to start or stop listening.\n"
    )
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    def _handle_signal(signum, frame):  # noqa: ANN001
        _ = (signum, frame)
        app_state.stop()
        server.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app_state.stop()
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime meeting speech translation (local ASR + LLM translation)")
    parser.add_argument("--source-language", default=None, help="ASR source language, e.g. zh, en")
    parser.add_argument("--target-language", default=None, help="Target translation language, e.g. en, zh")
    parser.add_argument("--asr-model", default=None, help="faster-whisper model name, e.g. small, medium")
    parser.add_argument("--input-device", default=None, help="Audio input device id or name, default auto")
    parser.add_argument("--translation-model", default=None, help="LLM model for translation, e.g. gpt-5.1")
    parser.add_argument(
        "--translation-prompt-template",
        default=None,
        help="Custom translation system prompt template. Supports placeholders like {source_language}, {target_language}, {speaker_label}.",
    )
    parser.add_argument("--terminal", action="store_true", help="Run in terminal mode instead of the local web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local web UI, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="Port for the local web UI, default 8080")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically in web mode")
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
    if args.input_device:
        settings.input_device = args.input_device
    if args.translation_model:
        settings.translation_model = args.translation_model
    if args.translation_prompt_template is not None:
        candidate = _normalize_translation_prompt_template(args.translation_prompt_template)
        ok, error = _validate_translation_prompt_template(candidate)
        if not ok:
            print(error or "Invalid translation prompt template.", file=sys.stderr)
            return 2
        settings.translation_prompt_template = candidate

    if not args.terminal:
        return run_web_interface(settings, args.host, args.port, open_browser=not args.no_browser)

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
