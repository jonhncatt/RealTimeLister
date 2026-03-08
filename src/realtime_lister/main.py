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

from huggingface_hub import snapshot_download
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "model"
DEFAULT_MODEL_CACHE_ROOT = DEFAULT_MODEL_ROOT / "cache"
ASR_MODEL_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")
ASR_MODEL_DOWNLOAD_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]
INTERACTIVE_CLI_HELP = (
    "COMMANDS\n"
    "  status     Show current ASR/runtime status\n"
    "  setup      Detect or download the ASR model into ./model\n"
    "  download   Re-download the ASR model into ./model\n"
    "  start      Launch the local Web UI\n"
    "  terminal   Start terminal transcription mode\n"
    "  panel      Open the retro control panel\n"
    "  help       Show this help\n"
    "  quit       Exit the CLI\n"
)
CLI_BANNER = r"""
   _____ _                   _      ____             __
  / ___/(_)___ _____  ____ _/ /     / __ \___  _____/ /__
  \__ \/ / __ `/ __ \/ __ `/ /_____/ / / / _ \/ ___/ //_/
 ___/ / / /_/ / / / / /_/ / /_____/ /_/ /  __/ /__/ ,<
/____/_/\__, /_/ /_/\__,_/_/     /_____/\___/\___/_/|_|
       /____/
"""
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
CLI_SOURCE_LANGUAGE_OPTIONS = (
    ("auto", "Auto Detect"),
    ("zh", "Chinese"),
    ("en", "English"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("de", "German"),
)
CLI_TARGET_LANGUAGE_OPTIONS = (
    ("en", "English"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("de", "German"),
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


def _normalize_source_language(raw: str, *, default: str = "auto") -> str:
    value = (raw or "").strip().lower()
    if not value:
        return default
    if value in {"auto", "automatic", "detect", "auto-detect"}:
        return "auto"
    return value


def _normalize_target_language(raw: str, *, default: str = "en") -> str:
    value = (raw or "").strip().lower()
    return value or default


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


def _repo_model_root() -> Path:
    return DEFAULT_MODEL_ROOT


def _repo_model_cache_root() -> Path:
    return DEFAULT_MODEL_CACHE_ROOT


def _resolve_hf_repo_id(model: str) -> str:
    candidate = (model or "small").strip()
    if "/" in candidate:
        return candidate
    return f"Systran/faster-whisper-{candidate}"


def _default_model_dir_name(model: str) -> str:
    repo_id = _resolve_hf_repo_id(model)
    return repo_id.rsplit("/", 1)[-1]


def _default_repo_model_dir(model: str) -> Path:
    return _repo_model_root() / _default_model_dir_name(model)


def _missing_asr_model_files(path: Path) -> list[str]:
    return [name for name in ASR_MODEL_REQUIRED_FILES if not (path / name).exists()]


def _is_complete_asr_model_dir(path: Path) -> bool:
    return path.exists() and not _missing_asr_model_files(path)


def _resolve_local_model_hint(model: str, configured_path: str | None) -> str | None:
    if configured_path:
        return str(Path(configured_path).expanduser())
    default_path = _default_repo_model_dir(model)
    if _is_complete_asr_model_dir(default_path):
        return str(default_path)
    return None


def _download_asr_model_to_dir(settings: "Settings", output_dir: Path) -> Path:
    target_dir = output_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    _configure_network_env(settings)
    repo_id = _resolve_hf_repo_id(settings.asr_model)
    print(f"[asr] Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=ASR_MODEL_DOWNLOAD_PATTERNS,
        token=settings.hf_token,
    )
    missing = _missing_asr_model_files(target_dir)
    if missing:
        raise RuntimeError(f"Downloaded model directory is incomplete: missing {', '.join(missing)}")
    return target_dir


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
        asr_model = _env("RT_ASR_MODEL_NAME", "RT_WHISPER_MODEL", default="small")
        asr_model_path = _env("RT_ASR_MODEL_DIR", "RT_WHISPER_MODEL_PATH")
        asr_download_root = _env("RT_ASR_HF_CACHE_DIR", "RT_HF_CACHE_DIR")
        local_model_hint = _resolve_local_model_hint(asr_model, asr_model_path)
        prompt_template = _normalize_translation_prompt_template(_env("RT_TRANSLATION_PROMPT_TEMPLATE", default=""))
        prompt_ok, prompt_error = _validate_translation_prompt_template(prompt_template)
        if not prompt_ok:
            print(
                f"[config] Invalid RT_TRANSLATION_PROMPT_TEMPLATE, fallback to default template. {prompt_error}",
                file=sys.stderr,
            )
            prompt_template = DEFAULT_TRANSLATION_PROMPT_TEMPLATE
        return cls(
            asr_model=asr_model,
            asr_model_path=local_model_hint,
            asr_download_root=str(Path(asr_download_root).expanduser()) if asr_download_root else str(_repo_model_cache_root()),
            asr_local_files_only=_truthy(_env("RT_ASR_HF_LOCAL_ONLY", "RT_HF_LOCAL_FILES_ONLY", default="false")),
            asr_beam_size=max(1, int(_env("RT_ASR_BEAM_SIZE", "RT_BEAM_SIZE", default="1"))),
            input_device=_env("RT_AUDIO_INPUT_DEVICE", default="auto"),
            device=_env("RT_ASR_DEVICE", "RT_DEVICE", default="auto"),
            compute_type=_env("RT_ASR_COMPUTE_TYPE", "RT_COMPUTE_TYPE", default="int8"),
            source_language=_normalize_source_language(_env("RT_SOURCE_LANGUAGE", default="auto")),
            target_language=_normalize_target_language(_env("RT_TARGET_LANGUAGE", default="en")),
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
        missing = _missing_asr_model_files(model_path)
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

    def translate(
        self,
        text: str,
        *,
        speaker_label: str = "Speaker 1",
        speaker_id: str = "speaker-1",
        source_language: str | None = None,
    ) -> str:
        if not text.strip() or not self.enabled or self.client is None:
            return text

        system_prompt = self._build_system_prompt(
            speaker_label=speaker_label,
            speaker_id=speaker_id,
            source_language=source_language,
        )

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

    def _build_system_prompt(self, *, speaker_label: str, speaker_id: str, source_language: str | None = None) -> str:
        template = _normalize_translation_prompt_template(self.settings.translation_prompt_template)
        glossary_block = ""
        if self.settings.glossary:
            glossary_block = f"\nApply this glossary strictly when relevant:\n{self.settings.glossary}"
        effective_source_language = _normalize_source_language(source_language or self.settings.source_language)
        if effective_source_language == "auto":
            effective_source_language = "detected speech language"
        try:
            return template.format(
                source_language=effective_source_language,
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

    def _segment_to_text(self, pcm16le: bytes) -> tuple[str, str]:
        pcm = np.frombuffer(pcm16le, dtype=np.int16).astype(np.float32) / 32768.0
        configured_source = _normalize_source_language(self.settings.source_language)
        language_hint = None if configured_source == "auto" else configured_source
        segments, info = self.model.transcribe(
            pcm,
            language=language_hint,
            beam_size=self.settings.asr_beam_size,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=True,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        detected = _normalize_source_language(getattr(info, "language", "") or "", default="")
        effective_source = detected or configured_source
        return text, effective_source

    def _process_segment(self, pcm16le: bytes) -> None:
        speaker_id, speaker_label = self.speaker_diarizer.assign(pcm16le)
        source_text, segment_source_language = self._segment_to_text(pcm16le)
        if not source_text:
            return

        started = time.perf_counter()
        translated = self.translator.translate(
            source_text,
            speaker_label=speaker_label,
            speaker_id=speaker_id,
            source_language=segment_source_language,
        )
        spent_ms = int((time.perf_counter() - started) * 1000)

        now = time.strftime("%H:%M:%S")
        entry = TranscriptEntry(
            timestamp=now,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            source_text=source_text,
            translated_text=translated,
            translate_ms=spent_ms,
            source_language=segment_source_language,
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
        source_display = "auto-detect" if _normalize_source_language(self.settings.source_language) == "auto" else self.settings.source_language
        startup_message = (
            "Realtime Meeting Translator started.\n"
            f"- ASR model: {_resolve_asr_model_source(self.settings)}\n"
            f"- ASR beam size: {self.settings.asr_beam_size}\n"
            f"- Audio input: {self.input_device_name} ({self.input_device_index})\n"
            f"- Speaker split: {'on' if self.settings.speaker_split_enabled else 'off'} "
            f"(max={self.settings.speaker_max_speakers})\n"
            f"- Source -> Target: {source_display} -> {self.settings.target_language}\n"
            f"- Translation model: {self.settings.translation_model}\n"
            "Press Q, S, X, Esc, or Ctrl+C to stop.\n"
        )
        if self.console_output:
            print(startup_message)
        if self.on_info:
            self.on_info(
                f"ASR ready: {_resolve_asr_model_source(self.settings)} | "
                f"{source_display}->{self.settings.target_language} | "
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
                "defaultTranslationPromptTemplate": DEFAULT_TRANSLATION_PROMPT_TEMPLATE,
                "translationPromptPlaceholders": list(TRANSLATION_PROMPT_PLACEHOLDERS),
                "glossary": self.settings.glossary,
                "glossaryLineCount": _glossary_line_count(self.settings.glossary),
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

    def set_glossary(self, value: str) -> tuple[bool, str]:
        normalized = (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        line_count = _glossary_line_count(normalized)
        with self._lock:
            self.settings.glossary = normalized
            self._last_info = f"Glossary updated ({line_count} lines)." if line_count else "Glossary cleared."
        self._publish({"type": "info", "message": self._last_info, "timestamp": time.strftime("%H:%M:%S")})
        self._publish_snapshot()
        return True, self._last_info

    def set_languages(self, source_language: str, target_language: str) -> tuple[bool, str]:
        normalized_source = _normalize_source_language(source_language)
        normalized_target = _normalize_target_language(target_language)
        with self._lock:
            if self._running or self._loading:
                return False, "Stop the current session before changing the language direction."
            self.settings.source_language = normalized_source
            self.settings.target_language = normalized_target
        source_label = "auto-detect" if normalized_source == "auto" else normalized_source
        self._set_status("idle", f"Language direction set to {source_label} -> {normalized_target}.")
        return True, "Language direction updated."


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
            if path == "/api/glossary":
                payload = self._read_json()
                ok, message = self.server.app_state.set_glossary(str(payload.get("glossary") or ""))
                self._send_json({"ok": ok, "message": message}, status=HTTPStatus.OK if ok else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/languages":
                payload = self._read_json()
                ok, message = self.server.app_state.set_languages(
                    str(payload.get("sourceLanguage") or "auto"),
                    str(payload.get("targetLanguage") or "en"),
                )
                self._send_json({"ok": ok, "message": message}, status=HTTPStatus.OK if ok else HTTPStatus.CONFLICT)
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
        "Press Q, S, X, or Esc in this terminal to stop the local server.\n"
    )
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    shortcut_done = _start_runtime_shortcut_listener(lambda: (app_state.stop(), server.shutdown()))

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
        if shortcut_done is not None:
            shortcut_done.set()
        app_state.stop()
        server.server_close()
    return 0


def run_terminal_session(settings: Settings) -> int:
    runner = RealtimeMeetingTranslator(settings)
    shortcut_done = _start_runtime_shortcut_listener(runner.stop)

    def _handle_signal(signum, frame):  # noqa: ANN001
        _ = (signum, frame)
        runner.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop()
    finally:
        if shortcut_done is not None:
            shortcut_done.set()
    return 0


def _prompt_text(label: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"{label}{suffix}: ").strip()
        except EOFError:
            print()
            return default or ""
        except KeyboardInterrupt:
            print()
            raise
        if raw:
            return raw
        if default is not None:
            return default


def _prompt_yes_no(question: str, *, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        answer = _prompt_text(f"{question}{suffix}", default="").lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _cli_supports_ansi() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _cli_paint(text: str, *codes: str) -> str:
    if not _cli_supports_ansi() or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _cli_rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _display_path(raw_path: str | None) -> str:
    if not raw_path:
        return "(not set)"
    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        return raw_path

    try:
        rel = resolved.relative_to(PROJECT_ROOT)
        return "." if str(rel) == "." else f"./{rel}"
    except ValueError:
        pass

    home = Path.home()
    try:
        rel_home = resolved.relative_to(home)
        return f"~/{rel_home}"
    except ValueError:
        return str(resolved)


def _compact_model_status(settings: "Settings", message: str) -> str:
    if settings.asr_model_path and message.startswith("Using fixed local model directory:"):
        return "ready :: fixed local directory"
    if message.startswith("ASR will use model name"):
        return "info :: model-name auto download"
    if message.startswith("Offline cache-only mode is enabled"):
        return "error :: offline cache missing model"
    return message


def _language_label(code: str, *, allow_auto: bool = True) -> str:
    options = CLI_SOURCE_LANGUAGE_OPTIONS if allow_auto else CLI_TARGET_LANGUAGE_OPTIONS
    for value, label in options:
        if value == code:
            return label
    return code.upper()


def _truncate_cli_text(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return f"{text[: width - 3]}..."


def _glossary_line_count(raw: str) -> int:
    return len([line for line in (raw or "").splitlines() if line.strip()])


def _clear_cli_screen() -> None:
    if _cli_supports_ansi():
        print("\033[2J\033[H", end="")


def _read_cli_keypress() -> str:
    if os.name == "nt":
        import msvcrt

        raw = msvcrt.getwch()
        if raw == "\x03":
            raise KeyboardInterrupt
        if raw in {"\r", "\n"}:
            return "enter"
        if raw in {"\x00", "\xe0"}:
            follow = msvcrt.getwch()
            return {
                "H": "up",
                "P": "down",
                "K": "left",
                "M": "right",
            }.get(follow, "")
        return {
            "k": "up",
            "j": "down",
            "h": "left",
            "l": "right",
            "w": "web",
            "t": "terminal",
            "s": "stop",
            "x": "stop",
            "q": "quit",
            "\x1b": "escape",
        }.get(raw.lower(), raw.lower())

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        raw = os.read(fd, 1).decode("utf-8", errors="ignore")
        if raw == "\x03":
            raise KeyboardInterrupt
        if raw in {"\r", "\n"}:
            return "enter"
        if raw == "\x1b":
            next_byte = os.read(fd, 1).decode("utf-8", errors="ignore")
            if next_byte == "[":
                final = os.read(fd, 1).decode("utf-8", errors="ignore")
                return {
                    "A": "up",
                    "B": "down",
                    "C": "right",
                    "D": "left",
                }.get(final, "escape")
            return "escape"
        return {
            "k": "up",
            "j": "down",
            "h": "left",
            "l": "right",
            "w": "web",
            "t": "terminal",
            "s": "stop",
            "x": "stop",
            "q": "quit",
        }.get(raw.lower(), raw.lower())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _cycle_value(options: list[str], current: str, step: int) -> str:
    if not options:
        return current
    try:
        index = options.index(current)
    except ValueError:
        index = 0
    return options[(index + step) % len(options)]


def _get_cli_input_device_options(settings: "Settings") -> list[tuple[str, str]]:
    devices, default_input_id, input_error = _query_input_devices()
    options: list[tuple[str, str]] = []
    auto_label = "Auto"
    if default_input_id is not None:
        for item in devices:
            if item["id"] == default_input_id:
                auto_label = f"Auto ({item['name']})"
                break
    elif input_error:
        auto_label = "Auto (unavailable)"
    options.append(("auto", auto_label))

    for item in devices:
        label = str(item["name"])
        if item["isDefault"]:
            label = f"{label} [default]"
        options.append((str(item["id"]), label))

    selected = (settings.input_device or "auto").strip() or "auto"
    if selected != "auto" and not any(value == selected for value, _ in options):
        options.append((selected, selected))
    return options


def _render_cli_control_panel(settings: "Settings", selected_index: int, footer: str) -> None:
    model_status = _inspect_asr_model_status(settings)
    direction_value = f"{_language_label(settings.source_language)} -> {_language_label(settings.target_language, allow_auto=False)}"
    device_options = _get_cli_input_device_options(settings)
    mic_label = next((label for value, label in device_options if value == settings.input_device), None)
    if not mic_label:
        mic_label = next((label for value, label in device_options if value == "auto"), "Auto")

    rows = [
        ("Direction", direction_value, "Common translation direction"),
        ("Mic", mic_label, "Active audio input"),
    ]

    _clear_cli_screen()
    _print_cli_banner()
    print()
    print(_cli_paint("CONTROL PANEL", "1", "33"))
    print(_cli_paint(_cli_rule(), "2"))
    print("  Use arrow keys or H/J/K/L to move. Left/Right changes values.")
    print("  Press W to launch Web UI, T for terminal mode, Enter for shell, Q to quit.")
    print()
    print("+------------------------------------------------------------------------------+")
    for index, (label, value, hint) in enumerate(rows):
        pointer = ">" if index == selected_index else " "
        label_text = f"{pointer} {label:<11}"
        value_text = f"[ {_truncate_cli_text(value, 28):<28} ]"
        hint_text = _truncate_cli_text(hint, 22)
        line = f"| {label_text} {value_text} {_truncate_cli_text(hint_text, 22):<22} |"
        if index == selected_index:
            line = _cli_paint(line, "1", "33")
        print(line)
    print("+------------------------------------------------------------------------------+")
    print(f"  Translator : {'enabled' if settings.api_key else 'ASR only'}")
    print(f"  ASR status  : {_compact_model_status(settings, model_status.message)}")
    print(f"  Model       : {_display_path(settings.asr_model_path) if settings.asr_model_path else settings.asr_model}")
    print()
    print(_cli_paint(f"  {footer}", "2"))


def _run_cli_control_panel(settings: "Settings") -> str:
    selected_index = 0
    footer = "Ready. Tune direction or microphone before launch."
    direction_options = [
        (source, target)
        for source, _ in CLI_SOURCE_LANGUAGE_OPTIONS
        for target, _ in CLI_TARGET_LANGUAGE_OPTIONS
    ]
    device_options = _get_cli_input_device_options(settings)

    while True:
        _render_cli_control_panel(settings, selected_index, footer)
        key = _read_cli_keypress()
        if key == "up":
            selected_index = (selected_index - 1) % 2
            footer = "Selection moved."
            continue
        if key == "down":
            selected_index = (selected_index + 1) % 2
            footer = "Selection moved."
            continue
        if key in {"left", "right"}:
            step = -1 if key == "left" else 1
            if selected_index == 0:
                current = (settings.source_language, settings.target_language)
                try:
                    idx = direction_options.index(current)
                except ValueError:
                    idx = 0
                new_source, new_target = direction_options[(idx + step) % len(direction_options)]
                settings.source_language = new_source
                settings.target_language = new_target
                footer = f"Direction set to {_language_label(new_source)} -> {_language_label(new_target, allow_auto=False)}."
            else:
                option_values = [value for value, _ in device_options]
                settings.input_device = _cycle_value(option_values, settings.input_device or "auto", step)
                selected_label = next((label for value, label in device_options if value == settings.input_device), settings.input_device)
                footer = f"Microphone set to {selected_label}."
            continue
        if key == "enter":
            _clear_cli_screen()
            return "shell"
        if key == "web":
            _clear_cli_screen()
            return "web"
        if key == "terminal":
            _clear_cli_screen()
            return "terminal"
        if key in {"quit", "escape", "stop"}:
            _clear_cli_screen()
            return "quit"


def _start_runtime_shortcut_listener(stop_action: Callable[[], None]) -> threading.Event | None:
    if not _cli_supports_ansi():
        return None

    done = threading.Event()

    def _worker() -> None:
        while not done.is_set():
            try:
                key = _read_cli_keypress()
            except Exception:
                return
            if key in {"quit", "escape", "stop"}:
                try:
                    stop_action()
                finally:
                    done.set()
                return

    thread = threading.Thread(target=_worker, name="realtime-shortcut-listener", daemon=True)
    thread.start()
    return done


def _print_cli_banner() -> None:
    print(_cli_paint(CLI_BANNER.rstrip(), "1", "33"))
    print(_cli_paint("  RealTimeLister CLI  //  local ASR  //  operator console", "2"))
    print(_cli_paint(_cli_rule("="), "2"))


def _print_cli_block(title: str, rows: list[tuple[str, str]]) -> None:
    print()
    print(_cli_paint(title.upper(), "1", "33"))
    print(_cli_paint(_cli_rule(), "2"))
    for label, value in rows:
        print(f"{_cli_paint(label + ':', '2'):24} {value}")


def _print_cli_status(settings: Settings) -> None:
    model_status = _inspect_asr_model_status(settings)
    source_display = "auto-detect" if settings.source_language == "auto" else settings.source_language
    _print_cli_block(
        "Status",
        [
            ("Project root", _display_path(str(PROJECT_ROOT))),
            ("Model root", _display_path(str(_repo_model_root()))),
            ("ASR model", settings.asr_model),
            ("ASR source", _display_path(settings.asr_model_path) if settings.asr_model_path else "(not fixed yet)"),
            ("ASR status", _compact_model_status(settings, model_status.message)),
            ("ASR cache root", _display_path(settings.asr_download_root) if settings.asr_download_root else "(default)"),
            ("Direction", f"{source_display} -> {settings.target_language}"),
            ("Translation", "enabled" if settings.api_key else "ASR only"),
        ],
    )


def _ensure_cli_asr_ready(settings: Settings, *, force_download: bool = False) -> bool:
    default_download_path = _default_repo_model_dir(settings.asr_model)
    preferred_path = Path(settings.asr_model_path).expanduser() if settings.asr_model_path else _default_repo_model_dir(settings.asr_model)
    if not force_download and _is_complete_asr_model_dir(preferred_path):
        settings.asr_model_path = str(preferred_path)
        print(f"{_cli_paint('[asr]', '1', '32')} Ready: {_display_path(str(preferred_path))}")
        return True

    if settings.asr_model_path and not force_download:
        configured_path = Path(settings.asr_model_path).expanduser()
        if _is_complete_asr_model_dir(configured_path):
            print(f"{_cli_paint('[asr]', '1', '32')} Ready: {_display_path(str(configured_path))}")
            return True
        if configured_path.exists():
            missing = _missing_asr_model_files(configured_path)
            print(
                f"{_cli_paint('[asr]', '1', '31')} Configured model directory is incomplete: "
                f"{_display_path(str(configured_path))} (missing: {', '.join(missing)})"
            )
        else:
            print(f"{_cli_paint('[asr]', '1', '31')} Configured model directory not found: {_display_path(str(configured_path))}")

    if not force_download:
        print(f"{_cli_paint('[asr]', '1', '33')} No ready local ASR model found for '{settings.asr_model}'.")
        print(f"{_cli_paint('[asr]', '1', '33')} Default download path: {_display_path(str(_default_repo_model_dir(settings.asr_model)))}")
        if not _prompt_yes_no("Download the ASR model now?", default=True):
            return False

    download_target = Path(
        _prompt_text(
            "ASR model directory",
            default=str(default_download_path),
        )
    ).expanduser()
    try:
        downloaded_dir = _download_asr_model_to_dir(settings, download_target)
    except Exception as exc:
        print(f"{_cli_paint('[asr]', '1', '31')} Download failed: {exc}")
        return False

    settings.asr_model_path = str(downloaded_dir)
    settings.asr_local_files_only = False
    print(f"{_cli_paint('[asr]', '1', '32')} Download complete. Using local model directory: {_display_path(str(downloaded_dir))}")
    return True


def run_interactive_cli(settings: Settings, host: str, port: int, open_browser: bool) -> int:
    _print_cli_banner()
    _print_cli_block(
        "Boot",
        [
            ("Project root", _display_path(str(PROJECT_ROOT))),
            ("Model root", _display_path(str(_repo_model_root()))),
            ("Hint", "Type help for commands"),
        ],
    )
    _print_cli_status(settings)
    ready = _ensure_cli_asr_ready(settings)
    if ready and _cli_supports_ansi():
        try:
            startup_action = _run_cli_control_panel(settings)
        except Exception:
            startup_action = ""
        if startup_action == "quit":
            return 0
        if startup_action == "web":
            try:
                run_web_interface(settings, host, port, open_browser=open_browser)
            except Exception as exc:
                print(f"[web] Failed to start: {exc}")
        elif startup_action == "terminal":
            try:
                run_terminal_session(settings)
            except Exception as exc:
                print(f"[terminal] Failed to start: {exc}")
        elif startup_action == "shell":
            _print_cli_banner()
            _print_cli_status(settings)
        elif _prompt_yes_no("ASR is ready. Launch the Web UI now?", default=True):
            try:
                run_web_interface(settings, host, port, open_browser=open_browser)
            except Exception as exc:
                print(f"[web] Failed to start: {exc}")
    elif ready and _prompt_yes_no("ASR is ready. Launch the Web UI now?", default=True):
        try:
            run_web_interface(settings, host, port, open_browser=open_browser)
        except Exception as exc:
            print(f"[web] Failed to start: {exc}")

    while True:
        try:
            raw = input(_cli_paint("rtl> ", "1", "33")).strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not raw:
            continue

        command = raw.lower()
        if command in {"quit", "exit"}:
            return 0
        if command == "help":
            print(INTERACTIVE_CLI_HELP)
            continue
        if command == "panel":
            if _cli_supports_ansi():
                try:
                    action = _run_cli_control_panel(settings)
                except Exception:
                    print("Retro control panel is unavailable in this terminal.")
                    continue
                if action == "quit":
                    return 0
                if action == "web":
                    try:
                        run_web_interface(settings, host, port, open_browser=open_browser)
                    except Exception as exc:
                        print(f"[web] Failed to start: {exc}")
                elif action == "terminal":
                    try:
                        run_terminal_session(settings)
                    except Exception as exc:
                        print(f"[terminal] Failed to start: {exc}")
                else:
                    _print_cli_banner()
                    _print_cli_status(settings)
            else:
                print("Retro control panel requires an ANSI-capable TTY.")
            continue
        if command == "status":
            _print_cli_status(settings)
            continue
        if command == "setup":
            _ensure_cli_asr_ready(settings, force_download=False)
            continue
        if command == "download":
            _ensure_cli_asr_ready(settings, force_download=True)
            continue
        if command == "start":
            if not _ensure_cli_asr_ready(settings, force_download=False):
                continue
            try:
                run_web_interface(settings, host, port, open_browser=open_browser)
            except Exception as exc:
                print(f"[web] Failed to start: {exc}")
            continue
        if command == "terminal":
            if not _ensure_cli_asr_ready(settings, force_download=False):
                continue
            try:
                run_terminal_session(settings)
            except Exception as exc:
                print(f"[terminal] Failed to start: {exc}")
            continue
        print("Unknown command. Type help.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime meeting speech translation (local ASR + LLM translation)")
    parser.add_argument("--source-language", default=None, help="ASR source language, e.g. zh, en, auto")
    parser.add_argument("--target-language", default=None, help="Target translation language, e.g. en, zh")
    parser.add_argument("--asr-model", default=None, help="faster-whisper model name, e.g. small, medium")
    parser.add_argument("--input-device", default=None, help="Audio input device id or name, default auto")
    parser.add_argument("--translation-model", default=None, help="LLM model for translation, e.g. gpt-5.1")
    parser.add_argument(
        "--translation-prompt-template",
        default=None,
        help="Custom translation system prompt template. Supports placeholders like {source_language}, {target_language}, {speaker_label}.",
    )
    parser.add_argument("--web", action="store_true", help="Run the local Web UI directly and skip the interactive CLI")
    parser.add_argument("--terminal", action="store_true", help="Run in terminal mode instead of the local web UI")
    parser.add_argument("--interactive", action="store_true", help="Force the interactive CLI even when flags are present")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local web UI, default 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="Port for the local web UI, default 8080")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically in web mode")
    return parser.parse_args()


def main() -> int:
    settings = Settings.load()
    args = parse_args()
    if args.web and args.terminal:
        print("Choose either --web or --terminal, not both.", file=sys.stderr)
        return 2
    if args.source_language:
        settings.source_language = _normalize_source_language(args.source_language)
    if args.target_language:
        settings.target_language = _normalize_target_language(args.target_language)
    if args.asr_model:
        settings.asr_model = args.asr_model
        configured_model_dir = _env("RT_ASR_MODEL_DIR", "RT_WHISPER_MODEL_PATH")
        settings.asr_model_path = _resolve_local_model_hint(settings.asr_model, configured_model_dir)
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

    use_interactive = args.interactive or (
        not args.web and not args.terminal and sys.stdin.isatty() and sys.stdout.isatty()
    )
    if use_interactive:
        return run_interactive_cli(settings, args.host, args.port, open_browser=not args.no_browser)

    if not args.terminal:
        return run_web_interface(settings, args.host, args.port, open_browser=not args.no_browser)

    return run_terminal_session(settings)


if __name__ == "__main__":
    raise SystemExit(main())
