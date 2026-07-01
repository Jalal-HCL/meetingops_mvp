import base64
import ctypes
import html
import io
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import wave
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

from agents.demo_data import format_demo_transcript
from agents.diarizer import transcribe_audio, format_diarized_transcript
from agents.language_handler import translate_segments, best_effort_english_fallback
from agents.extractor import extract_action_items, generate_meeting_summary
from utils.db_chroma import (
    close_action_item,
    delete_action_item,
    delete_meeting,
    get_open_action_items,
    get_pre_meeting_brief,
    list_action_items,
    list_meetings,
    save_action_items,
    save_meeting,
    storage_status,
    update_action_item,
    update_meeting,
)
import utils.calendar_mock as calendar_helpers

calendar_helpers = importlib.reload(calendar_helpers)
MicrosoftCalendarError = calendar_helpers.MicrosoftCalendarError
complete_device_login = calendar_helpers.complete_device_login
get_upcoming_meetings = calendar_helpers.get_upcoming_meetings
is_microsoft_calendar_configured = calendar_helpers.is_microsoft_calendar_configured
refresh_access_token = calendar_helpers.refresh_access_token
start_device_login = calendar_helpers.start_device_login
token_is_valid = calendar_helpers.token_is_valid

load_dotenv()

DEFAULT_LOGIN_USERNAME = os.getenv("MEETINGOPS_USERNAME", "admin")
DEFAULT_LOGIN_PASSWORD = os.getenv("MEETINGOPS_PASSWORD", "admin123")

HOME_PAGE = "Home"
LIVE_PAGE = "Live Meeting"
BRIEFING_PAGE = "Pre-Meeting Briefing"
ACTION_ITEMS_PAGE = "Action Items"
DATA_MANAGER_PAGE = "Data Manager"
PAGES = [HOME_PAGE, LIVE_PAGE, BRIEFING_PAGE, ACTION_ITEMS_PAGE, DATA_MANAGER_PAGE]
DEMO_SIMULATION_MODE = "Demo Simulation"
MICROPHONE_AUDIO_MODE = "Microphone / Audio Input"
SYSTEM_AUDIO_MODE = "Windows Loopback / Headset Output"


def supports_system_audio_mode() -> bool:
    return sys.platform.startswith("win")


def is_closed_action_status(status: str) -> bool:
    return str(status or "").strip().upper() in {"CLOSED", "DONE", "COMPLETE", "COMPLETED"}


class RecordedAudio:
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def audio_mime_type(name: str, default: str = "audio/wav") -> str:
    suffix = Path(name or "").suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
    }.get(suffix, default)


def show_recorded_audio(audio_file, mime_type: str = "audio/webm"):
    audio_bytes = audio_file.getvalue()
    audio_player_with_speed(audio_bytes, mime_type)
    st.caption("Fallback player")
    st.audio(audio_bytes, format=mime_type)


def transcript_segments_from_text(transcript: str):
    return [{"speaker": "Manual", "role": "Participant", "text": transcript.strip()}]


def format_english_transcript_segments(segments):
    lines = [
        f"[{s.get('speaker', 'Unknown')} - {s.get('role', '')}]: {s.get('translated_text', '')}"
        for s in segments
        if str(s.get("translated_text", "")).strip()
    ]
    return "\n".join(lines)


def collect_recognizer_texts(segments):
    texts = []
    for segment in segments:
        for key in ("english_hint", "text"):
            text = str(segment.get(key, "")).strip()
            if text and text not in texts:
                texts.append(text)
        for candidate in segment.get("transcription_candidates", []) or []:
            text = str(candidate.get("text", "")).strip()
            if text and text not in texts:
                texts.append(text)
    return texts


def best_recognizer_fallback(segments):
    texts = collect_recognizer_texts(segments)
    if not texts:
        return ""
    return max(texts, key=lambda value: len(value.split()))


def fallback_language_result(segments, fallback_text):
    english_text = best_effort_english_fallback(fallback_text)
    if not english_text:
        raise ValueError("Fallback recognizer text could not be converted to English.")

    return {
        "segments": [
            {
                "speaker": segments[0].get("speaker", "Unknown") if segments else "Unknown",
                "role": segments[0].get("role", "Participant") if segments else "Participant",
                "original_text": fallback_text,
                "translated_text": english_text,
                "detected_language": "Unknown",
                "is_mixed": False,
                "translation_notes": "Strict English translation failed; local English fallback applied.",
            }
        ],
        "language_summary": {
            "primary_language": "Unknown",
            "languages_detected": [],
            "code_switching_detected": False,
            "india_languages_present": [],
        },
        "english_transcript": f"[{segments[0].get('speaker', 'Unknown') if segments else 'Unknown'} - {segments[0].get('role', 'Participant') if segments else 'Participant'}]: {english_text}",
        "bilingual_display": [
            {
                "speaker": segments[0].get("speaker", "Unknown") if segments else "Unknown",
                "original": fallback_text,
                "english": english_text,
                "language": "Unknown",
            }
        ],
    }


def show_transcription_candidates(segments):
    candidates = []
    for segment in segments:
        candidates.extend(segment.get("transcription_candidates", []))
    if not candidates:
        return
    with st.expander("Recognizer candidates"):
        for candidate in candidates:
            st.caption(candidate.get("source", "candidate"))
            st.code(candidate.get("text", ""))


def show_transcription_debug(segments):
    st.warning("Debug: transcription/translation returned no clean English.")
    st.caption("Raw recognizer output")
    st.code(format_diarized_transcript(segments) or "[no raw segments]")
    candidates = []
    for segment in segments:
        candidates.extend(segment.get("transcription_candidates", []))
    if candidates:
        st.caption("Recognizer candidates")
        for candidate in candidates:
            st.caption(candidate.get("source", "candidate"))
            st.code(candidate.get("text", "") or "[empty]")


def audio_player_with_speed(audio_bytes: bytes, mime_type: str = "audio/webm"):
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    player_id = "recorded-audio-player"
    components.html(
        f"""
        <div style="font-family: sans-serif; margin: 0;">
            <audio id="{player_id}" controls style="width: 100%;">
                <source src="data:{mime_type};base64,{encoded_audio}" type="{mime_type}">
            </audio>
            <div style="display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;">
                <button type="button" onclick="document.getElementById('{player_id}').playbackRate=0.5">0.5x</button>
                <button type="button" onclick="document.getElementById('{player_id}').playbackRate=0.8">0.8x</button>
                <button type="button" onclick="document.getElementById('{player_id}').playbackRate=1">1x</button>
                <button type="button" onclick="document.getElementById('{player_id}').playbackRate=1.5">1.5x</button>
                <button type="button" onclick="document.getElementById('{player_id}').playbackRate=2">2x</button>
            </div>
        </div>
        """,
        height=96,
    )


def _import_soundcard_for_audio_thread():
    soundcard_already_loaded = "soundcard" in sys.modules or "soundcard.mediafoundation" in sys.modules
    try:
        import soundcard as sc
    except ImportError as exc:
        raise RuntimeError(
            "System audio recording needs the soundcard package. Run: py -3.13 -m pip install soundcard"
        ) from exc
    return sc, soundcard_already_loaded


def _system_loopback_devices(sc):
    speaker = sc.default_speaker()
    candidates = []
    seen = set()

    def add_candidate(loopback):
        device_id = getattr(loopback, "id", None) or getattr(loopback, "name", "")
        if device_id in seen:
            return
        seen.add(device_id)
        candidates.append(loopback)

    try:
        add_candidate(sc.get_microphone(id=speaker.name, include_loopback=True))
    except Exception:
        pass

    for mic in sc.all_microphones(include_loopback=True):
        if getattr(mic, "isloopback", False):
            add_candidate(mic)

    if not candidates:
        raise RuntimeError("No Windows loopback/system audio device was found.")
    return speaker, candidates


def _system_microphone_devices(sc):
    default_mic = sc.default_microphone()
    candidates = []
    seen = set()

    def add_candidate(mic):
        device_id = getattr(mic, "id", None) or getattr(mic, "name", "")
        if device_id in seen:
            return
        seen.add(device_id)
        candidates.append(mic)

    if default_mic is not None:
        add_candidate(default_mic)
    for mic in sc.all_microphones(include_loopback=False):
        if not getattr(mic, "isloopback", False):
            add_candidate(mic)
    return default_mic, candidates


def _matching_audio_device(devices, preferred_device: str, fallback=None):
    preferred_device = str(preferred_device or "").strip()
    if preferred_device:
        for device in devices:
            if preferred_device in {
                str(getattr(device, "name", "")),
                str(getattr(device, "id", "")),
            }:
                return device
    return fallback if fallback is not None else (devices[0] if devices else None)


def _friendly_audio_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if "0x100000001" in message:
        return "Windows audio was already initialized by another recorder. Start again; the app will now avoid that path."
    if "0x800401f0" in message:
        return "Windows audio was not initialized for this recorder thread."
    return message


SILENT_AUDIO_PEAK_THRESHOLD = 0.0001
SILENT_AUDIO_RMS_THRESHOLD = 0.00001


def _prepare_audio_samples(data):
    if data.size == 0:
        raise RuntimeError("System audio recording was empty.")

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] > 2:
        data = data[:, :2]
    return data


def _audio_signal_stats(data, sample_rate: int) -> dict:
    data = _prepare_audio_samples(data)
    abs_data = np.abs(data)
    peak = float(np.max(abs_data)) if abs_data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    nonzero_ratio = float(np.count_nonzero(abs_data > 0.00001) / abs_data.size) if abs_data.size else 0.0
    return {
        "duration_seconds": round(float(data.shape[0] / sample_rate), 2),
        "channels": int(data.shape[1]),
        "peak": peak,
        "rms": rms,
        "nonzero_ratio": nonzero_ratio,
    }


def _audio_is_silent(stats: dict) -> bool:
    return (
        float(stats.get("peak", 0.0)) < SILENT_AUDIO_PEAK_THRESHOLD
        and float(stats.get("rms", 0.0)) < SILENT_AUDIO_RMS_THRESHOLD
    )


def _chunk_has_audio_signal(chunk) -> bool:
    if chunk is None or not getattr(chunk, "size", 0):
        return False
    peak = float(np.max(np.abs(chunk)))
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    return peak >= SILENT_AUDIO_PEAK_THRESHOLD or rms >= SILENT_AUDIO_RMS_THRESHOLD


def _format_audio_signal_stats(stats: dict | None) -> str:
    if not stats:
        return "Signal level unavailable."
    return (
        f"Duration {stats.get('duration_seconds', 0):.1f}s, "
        f"peak {stats.get('peak', 0):.5f}, "
        f"RMS {stats.get('rms', 0):.5f}"
    )


def _wav_bytes_from_samples(data, sample_rate: int) -> bytes:
    data = _prepare_audio_samples(data)

    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _as_stereo_float(data):
    data = _prepare_audio_samples(data).astype(np.float32)
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2]


def _mix_audio_tracks(tracks):
    arrays = []
    for data, gain in tracks:
        if data is None or not getattr(data, "size", 0):
            continue
        arrays.append(_as_stereo_float(data) * float(gain))
    if not arrays:
        raise RuntimeError("No audio tracks were captured.")

    max_len = max(array.shape[0] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[0] < max_len:
            pad = np.zeros((max_len - array.shape[0], array.shape[1]), dtype=np.float32)
            array = np.vstack([array, pad])
        padded.append(array)

    mixed = np.sum(padded, axis=0)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def _initialize_com_for_audio_thread() -> bool:
    result = ctypes.windll.ole32.CoInitializeEx(None, 0)
    if result in (0, 1):
        return True
    if result == -2147417850:
        return False
    raise OSError(f"Windows audio initialization failed: 0x{result & 0xffffffff:08x}")


def get_system_audio_device_summary() -> dict:
    com_initialized = False
    try:
        sc, soundcard_already_loaded = _import_soundcard_for_audio_thread()
        if soundcard_already_loaded:
            com_initialized = _initialize_com_for_audio_thread()
        speaker, loopbacks = _system_loopback_devices(sc)
        default_mic, microphones = _system_microphone_devices(sc)
        return {
            "speaker": getattr(speaker, "name", "default speaker"),
            "loopbacks": [getattr(loopback, "name", "system audio") for loopback in loopbacks],
            "default_microphone": getattr(default_mic, "name", "default microphone") if default_mic else "",
            "microphones": [getattr(mic, "name", "microphone") for mic in microphones],
            "error": "",
        }
    except Exception as exc:
        return {"speaker": "", "loopbacks": [], "default_microphone": "", "microphones": [], "error": _friendly_audio_error(exc)}
    finally:
        if com_initialized:
            ctypes.windll.ole32.CoUninitialize()


def _record_loopback_track(sc, job: dict, sample_rate: int, chunk_frames: int) -> dict:
    speaker, loopbacks = _system_loopback_devices(sc)
    preferred_device = str(job.get("preferred_device", "")).strip()
    loopback = _matching_audio_device(loopbacks, preferred_device)
    errors = []
    chunks = None
    device_name = getattr(speaker, "name", "default speaker")

    if preferred_device or len(loopbacks) == 1:
        if loopback is None:
            return {"error": "No matching Windows output loopback device was found."}
        device_name = getattr(loopback, "name", device_name)
        job["device"] = device_name
        chunks = []
        try:
            while not job["stop_event"].is_set():
                chunk = loopback.record(samplerate=sample_rate, numframes=chunk_frames)
                chunks.append(chunk)
                job["loopback_peak"] = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                job["last_peak"] = job["loopback_peak"]
        except Exception as exc:
            errors.append(f"{device_name}: {_friendly_audio_error(exc)}")
            chunks = None
    else:
        scan_chunks_by_device = {}
        active_loopback = None
        active_chunks = []

        while not job["stop_event"].is_set() and active_loopback is None:
            for candidate in loopbacks:
                if job["stop_event"].is_set():
                    break
                candidate_name = getattr(candidate, "name", "Windows output")
                job["device"] = f"Auto-scanning: {candidate_name}"
                try:
                    chunk = candidate.record(samplerate=sample_rate, numframes=chunk_frames)
                except Exception as exc:
                    errors.append(f"{candidate_name}: {_friendly_audio_error(exc)}")
                    continue

                scan_chunks_by_device.setdefault(candidate_name, []).append(chunk)
                job["loopback_peak"] = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                job["last_peak"] = job["loopback_peak"]
                if _chunk_has_audio_signal(chunk):
                    active_loopback = candidate
                    active_chunks = [chunk]
                    device_name = candidate_name
                    job["device"] = device_name
                    break

        if active_loopback is not None:
            chunks = active_chunks
            try:
                while not job["stop_event"].is_set():
                    chunk = active_loopback.record(samplerate=sample_rate, numframes=chunk_frames)
                    chunks.append(chunk)
                    job["loopback_peak"] = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                    job["last_peak"] = job["loopback_peak"]
            except Exception as exc:
                errors.append(f"{device_name}: {_friendly_audio_error(exc)}")
                chunks = None
        elif scan_chunks_by_device:
            device_name, chunks = max(
                scan_chunks_by_device.items(),
                key=lambda item: max(
                    (float(np.max(np.abs(chunk))) for chunk in item[1] if getattr(chunk, "size", 0)),
                    default=0.0,
                ),
            )
            job["device"] = device_name

    if not chunks:
        return {"error": " ".join(errors).strip() or "No Windows output audio was captured."}

    data = np.concatenate(chunks, axis=0)
    return {"device": device_name, "data": data, "stats": _audio_signal_stats(data, sample_rate)}


def _record_microphone_track(sc, job: dict, sample_rate: int, chunk_frames: int) -> dict:
    default_mic, microphones = _system_microphone_devices(sc)
    mic = _matching_audio_device(microphones, job.get("preferred_microphone", ""), default_mic)
    if mic is None:
        return {"error": "No microphone device was found."}

    device_name = getattr(mic, "name", "microphone")
    job["microphone_device"] = device_name
    chunks = []
    try:
        while not job["stop_event"].is_set():
            chunk = mic.record(samplerate=sample_rate, numframes=chunk_frames)
            chunks.append(chunk)
            job["microphone_peak"] = float(np.max(np.abs(chunk))) if chunk.size else 0.0
    except Exception as exc:
        return {"device": device_name, "error": _friendly_audio_error(exc)}

    if not chunks:
        return {"device": device_name, "error": "No microphone audio was captured."}

    data = np.concatenate(chunks, axis=0)
    return {"device": device_name, "data": data, "stats": _audio_signal_stats(data, sample_rate)}


def _system_audio_worker(job: dict, sample_rate: int = 48000):
    com_initialized = False
    try:
        sc, soundcard_already_loaded = _import_soundcard_for_audio_thread()
        if soundcard_already_loaded:
            com_initialized = _initialize_com_for_audio_thread()

        chunk_frames = sample_rate // 2
        include_microphone = bool(job.get("include_microphone"))
        results = {}

        if include_microphone:
            threads = [
                threading.Thread(
                    target=lambda: results.update(loopback=_record_loopback_track(sc, job, sample_rate, chunk_frames)),
                    daemon=True,
                ),
                threading.Thread(
                    target=lambda: results.update(microphone=_record_microphone_track(sc, job, sample_rate, chunk_frames)),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            results["loopback"] = _record_loopback_track(sc, job, sample_rate, chunk_frames)

        loopback_result = results.get("loopback", {})
        microphone_result = results.get("microphone", {})
        tracks = []
        track_stats = {}
        warnings = []

        if loopback_result.get("data") is not None:
            tracks.append((loopback_result["data"], 1.0))
            track_stats["loopback"] = loopback_result.get("stats", {})
            job["device"] = loopback_result.get("device", job.get("device", "Windows output"))
            if _audio_is_silent(loopback_result.get("stats", {})):
                warnings.append("Windows output track was silent.")
        elif loopback_result.get("error"):
            warnings.append(f"Windows output track: {loopback_result['error']}")

        if include_microphone:
            if microphone_result.get("data") is not None:
                tracks.append((microphone_result["data"], 2.0))
                track_stats["microphone"] = microphone_result.get("stats", {})
                job["microphone_device"] = microphone_result.get("device", "")
                if _audio_is_silent(microphone_result.get("stats", {})):
                    warnings.append("Microphone track was silent.")
            elif microphone_result.get("error"):
                warnings.append(f"Microphone track: {microphone_result['error']}")

        if not tracks:
            raise RuntimeError("No loopback or microphone audio was captured.")

        data = _mix_audio_tracks(tracks)
        stats = _audio_signal_stats(data, sample_rate)
        job["signal_stats"] = stats
        job["track_stats"] = track_stats
        if warnings:
            job["warning"] = " ".join(warnings)
        if _audio_is_silent(stats):
            job["error"] = (
                "No usable audio signal was detected. Make sure Teams audio is playing through the selected "
                "Windows output device and your microphone is not muted."
            )
            job["status"] = "silent"
            return

        job["bytes"] = _wav_bytes_from_samples(data, sample_rate)
        job["name"] = "teams_loopback_plus_mic_recording.wav" if include_microphone else "system_audio_recording.wav"
        job["status"] = "complete"
    except Exception as exc:
        job["error"] = _friendly_audio_error(exc)
        job["status"] = "error"
    finally:
        if com_initialized:
            ctypes.windll.ole32.CoUninitialize()
        job["finished_event"].set()


def start_system_audio_recording(
    preferred_device: str = "",
    include_microphone: bool = False,
    preferred_microphone: str = "",
) -> dict:
    job = {
        "status": "recording",
        "device": "",
        "preferred_device": preferred_device,
        "include_microphone": include_microphone,
        "preferred_microphone": preferred_microphone,
        "started_at": time.time(),
        "stop_event": threading.Event(),
        "finished_event": threading.Event(),
    }
    thread = threading.Thread(target=_system_audio_worker, args=(job,), daemon=True)
    job["thread"] = thread
    thread.start()
    return job


def stop_system_audio_recording(job: dict):
    if not job:
        raise RuntimeError("No active system audio recording was found.")
    job["stop_event"].set()
    job["finished_event"].wait(timeout=15)
    if job.get("status") == "recording":
        raise RuntimeError("System audio recorder did not stop in time. Please try again.")
    return job


st.set_page_config(page_title="MeetingOps AI MVP", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.16), transparent 34rem),
            linear-gradient(135deg, #0b1020 0%, #111827 48%, #10151f 100%);
        color: #e5e7eb;
    }
    .card { background: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #334155; }
    .hero {
        font-size: 44px;
        font-weight: 800;
        letter-spacing: 0;
        background: linear-gradient(90deg,#38bdf8,#34d399,#f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .login-shell {
        max-width: 1120px;
        margin: 5.5rem auto 0;
        padding: 0 1rem;
    }
    .login-eyebrow {
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .login-title {
        color: #f8fafc;
        font-size: clamp(2.3rem, 4vw, 4.4rem);
        font-weight: 900;
        letter-spacing: 0;
        line-height: 0.98;
        margin: 0 0 1rem;
        max-width: 660px;
    }
    .login-copy {
        color: #cbd5e1;
        font-size: 1.06rem;
        line-height: 1.65;
        max-width: 600px;
        margin-bottom: 1.4rem;
    }
    .login-proof-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        max-width: 620px;
        margin-top: 1.8rem;
    }
    .login-proof {
        min-height: 92px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 0.9rem;
        background: rgba(15, 23, 42, 0.56);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .login-proof b {
        color: #f8fafc;
        display: block;
        font-size: 0.96rem;
        margin-bottom: 0.25rem;
    }
    .login-proof span {
        color: #94a3b8;
        display: block;
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .login-card-heading {
        margin-bottom: 1.15rem;
    }
    .login-card-heading h2 {
        color: #f8fafc;
        font-size: 1.55rem;
        margin: 0 0 0.35rem;
        letter-spacing: 0;
    }
    .login-card-heading p {
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }
    .login-hint {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-top: 0.9rem;
        text-align: center;
    }
    .login-hint code {
        color: #e2e8f0;
        background: rgba(15, 23, 42, 0.84);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 6px;
        padding: 0.12rem 0.34rem;
    }
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 1.6rem;
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(18px);
    }
    div[data-testid="stForm"] label {
        color: #cbd5e1;
        font-weight: 700;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.32);
        background: rgba(2, 6, 23, 0.54);
        color: #f8fafc;
        min-height: 2.9rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #2dd4bf;
        box-shadow: 0 0 0 0.16rem rgba(45, 212, 191, 0.18);
    }
    div[data-testid="stForm"] button[kind="primary"] {
        border-radius: 8px;
        min-height: 3rem;
        background: linear-gradient(90deg, #0ea5e9, #14b8a6);
        border: 0;
        color: #f8fafc;
        font-weight: 800;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover {
        background: linear-gradient(90deg, #0284c7, #0f766e);
        border: 0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8;
    }
    .home-hero {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 2rem;
        background:
            linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(20, 184, 166, 0.08) 48%, rgba(245, 158, 11, 0.10)),
            rgba(15, 23, 42, 0.68);
        box-shadow: 0 24px 68px rgba(0, 0, 0, 0.24);
        margin-bottom: 1.5rem;
    }
    .home-kicker {
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .home-title {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 4.5rem);
        font-weight: 900;
        line-height: 1.02;
        letter-spacing: 0;
        max-width: 780px;
        margin: 0 0 1rem;
    }
    .home-title span {
        background: linear-gradient(90deg, #38bdf8, #34d399, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .home-copy {
        color: #cbd5e1;
        font-size: 1.06rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;
    }
    .home-metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.5rem;
    }
    .home-metric {
        background: rgba(2, 6, 23, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 1rem;
        min-height: 104px;
    }
    .home-metric strong {
        color: #f8fafc;
        display: block;
        font-size: 1.45rem;
        margin-bottom: 0.24rem;
    }
    .home-metric span {
        color: #94a3b8;
        display: block;
        font-size: 0.9rem;
        line-height: 1.35;
    }
    .home-card {
        min-height: 178px;
        background: rgba(30, 41, 59, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 1.45rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .home-card-label {
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .home-card h3 {
        color: #f8fafc;
        font-size: 1.45rem;
        margin: 0 0 0.75rem;
        letter-spacing: 0;
    }
    .home-card p {
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
    }
    .home-section-title {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 850;
        margin: 1.45rem 0 0.85rem;
        letter-spacing: 0;
    }
    .pipeline-card {
        min-height: 132px;
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 1rem;
        position: relative;
        overflow: hidden;
    }
    .pipeline-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #34d399);
    }
    .pipeline-step {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 900;
        margin-bottom: 0.55rem;
    }
    .pipeline-card b {
        color: #f8fafc;
        display: block;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }
    .pipeline-card span {
        color: #94a3b8;
        display: block;
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .live-hero {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 1.75rem;
        background:
            linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(14, 165, 233, 0.10) 45%, rgba(20, 184, 166, 0.10)),
            rgba(15, 23, 42, 0.72);
        box-shadow: 0 24px 68px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.15rem;
    }
    .live-kicker {
        color: #fca5a5;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .live-title {
        color: #f8fafc;
        font-size: clamp(2rem, 3.4vw, 3.7rem);
        font-weight: 900;
        line-height: 1.02;
        letter-spacing: 0;
        margin: 0 0 0.85rem;
    }
    .live-title span {
        background: linear-gradient(90deg, #fb7185, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .live-copy {
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 820px;
        margin: 0;
    }
    .live-status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.35rem;
    }
    .live-status {
        min-height: 96px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        background: rgba(2, 6, 23, 0.42);
        padding: 0.9rem;
    }
    .live-status b {
        color: #f8fafc;
        display: block;
        font-size: 0.98rem;
        margin-bottom: 0.28rem;
    }
    .live-status span {
        color: #94a3b8;
        display: block;
        font-size: 0.86rem;
        line-height: 1.35;
    }
    .live-section-title {
        color: #f8fafc;
        font-size: 1.32rem;
        font-weight: 850;
        margin: 1.25rem 0 0.55rem;
        letter-spacing: 0;
    }
    .live-section-copy {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: -0.2rem 0 0.85rem;
    }
    .live-calendar-card {
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 8px;
        background: rgba(14, 165, 233, 0.08);
        padding: 1rem;
        margin: 0.4rem 0 0.85rem;
    }
    .live-calendar-card b {
        color: #f8fafc;
        display: block;
        font-size: 1rem;
        margin-bottom: 0.28rem;
    }
    .live-calendar-card span {
        color: #bae6fd;
        display: block;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .live-mode-card {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.58);
        padding: 1rem;
        margin-bottom: 0.85rem;
    }
    .live-mode-card b {
        color: #f8fafc;
        display: block;
        margin-bottom: 0.25rem;
    }
    .live-mode-card span {
        color: #94a3b8;
        display: block;
        font-size: 0.88rem;
        line-height: 1.4;
    }
    .brief-hero {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 1.75rem;
        background:
            linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(56, 189, 248, 0.10) 48%, rgba(245, 158, 11, 0.10)),
            rgba(15, 23, 42, 0.72);
        box-shadow: 0 24px 68px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.15rem;
    }
    .brief-kicker {
        color: #86efac;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .brief-title {
        color: #f8fafc;
        font-size: clamp(2rem, 3.4vw, 3.6rem);
        font-weight: 900;
        line-height: 1.02;
        letter-spacing: 0;
        margin: 0 0 0.85rem;
    }
    .brief-title span {
        background: linear-gradient(90deg, #34d399, #38bdf8, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brief-copy {
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 800px;
        margin: 0;
    }
    .brief-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.35rem;
    }
    .brief-tile {
        min-height: 94px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        background: rgba(2, 6, 23, 0.42);
        padding: 0.9rem;
    }
    .brief-tile b {
        color: #f8fafc;
        display: block;
        font-size: 0.98rem;
        margin-bottom: 0.28rem;
    }
    .brief-tile span {
        color: #94a3b8;
        display: block;
        font-size: 0.86rem;
        line-height: 1.35;
    }
    .brief-section-title {
        color: #f8fafc;
        font-size: 1.32rem;
        font-weight: 850;
        margin: 1.25rem 0 0.55rem;
        letter-spacing: 0;
    }
    .brief-section-copy {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: -0.2rem 0 0.85rem;
    }
    .brief-meeting-card {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.72);
        padding: 1.2rem;
        margin: 0.7rem 0 1rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .brief-meeting-card h3 {
        color: #f8fafc;
        font-size: 1.35rem;
        letter-spacing: 0;
        margin: 0 0 0.75rem;
    }
    .brief-meta-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }
    .brief-meta {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.52);
        padding: 0.85rem;
        min-height: 84px;
    }
    .brief-meta span {
        color: #94a3b8;
        display: block;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
    }
    .brief-meta b {
        color: #f8fafc;
        display: block;
        font-size: 0.94rem;
        line-height: 1.35;
        word-break: break-word;
    }
    .actions-hero {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 1.75rem;
        background:
            linear-gradient(135deg, rgba(56, 189, 248, 0.13), rgba(129, 140, 248, 0.10) 45%, rgba(52, 211, 153, 0.10)),
            rgba(15, 23, 42, 0.72);
        box-shadow: 0 24px 68px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.15rem;
    }
    .actions-kicker {
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .actions-title {
        color: #f8fafc;
        font-size: clamp(2rem, 3.4vw, 3.6rem);
        font-weight: 900;
        line-height: 1.02;
        letter-spacing: 0;
        margin: 0 0 0.85rem;
    }
    .actions-title span {
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .actions-copy {
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 800px;
        margin: 0;
    }
    .actions-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.35rem;
    }
    .actions-stat {
        min-height: 94px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        background: rgba(2, 6, 23, 0.42);
        padding: 0.9rem;
    }
    .actions-stat strong {
        color: #f8fafc;
        display: block;
        font-size: 1.45rem;
        margin-bottom: 0.24rem;
    }
    .actions-stat span {
        color: #94a3b8;
        display: block;
        font-size: 0.86rem;
        line-height: 1.35;
    }
    .actions-section-title {
        color: #f8fafc;
        font-size: 1.32rem;
        font-weight: 850;
        margin: 1.25rem 0 0.55rem;
        letter-spacing: 0;
    }
    .actions-section-copy {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: -0.2rem 0 0.85rem;
    }
    .action-card {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        background: rgba(30, 41, 59, 0.72);
        padding: 1.15rem;
        margin: 0.8rem 0 0.4rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .action-card h3 {
        color: #f8fafc;
        font-size: 1.12rem;
        letter-spacing: 0;
        line-height: 1.35;
        margin: 0 0 0.85rem;
    }
    .action-meta-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
    }
    .action-meta {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.52);
        padding: 0.75rem;
        min-height: 76px;
    }
    .action-meta span {
        color: #94a3b8;
        display: block;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .action-meta b {
        color: #f8fafc;
        display: block;
        font-size: 0.9rem;
        line-height: 1.35;
        word-break: break-word;
    }
    .empty-actions {
        border: 1px dashed rgba(148, 163, 184, 0.30);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.44);
        padding: 1.4rem;
        color: #cbd5e1;
    }
    .admin-hero {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 8px;
        padding: 1.75rem;
        background:
            linear-gradient(135deg, rgba(168, 85, 247, 0.12), rgba(14, 165, 233, 0.10) 48%, rgba(245, 158, 11, 0.10)),
            rgba(15, 23, 42, 0.72);
        box-shadow: 0 24px 68px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.15rem;
    }
    .admin-kicker {
        color: #c4b5fd;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .admin-title {
        color: #f8fafc;
        font-size: clamp(2rem, 3.4vw, 3.6rem);
        font-weight: 900;
        line-height: 1.02;
        letter-spacing: 0;
        margin: 0 0 0.85rem;
    }
    .admin-title span {
        background: linear-gradient(90deg, #c084fc, #38bdf8, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .admin-copy {
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 820px;
        margin: 0;
    }
    .admin-section-title {
        color: #f8fafc;
        font-size: 1.32rem;
        font-weight: 850;
        margin: 1.25rem 0 0.55rem;
        letter-spacing: 0;
    }
    .admin-section-copy {
        color: #94a3b8;
        font-size: 0.94rem;
        line-height: 1.5;
        margin: -0.2rem 0 0.85rem;
    }
    .small { color: #94a3b8; }
    @media (max-width: 780px) {
        .login-shell { margin-top: 2rem; }
        .login-title { font-size: 2.35rem; line-height: 1.06; }
        .login-proof-row { grid-template-columns: 1fr; }
        .home-hero { padding: 1.25rem; }
        .home-metric-grid { grid-template-columns: 1fr; }
        .home-title { font-size: 2.2rem; }
        .live-hero { padding: 1.25rem; }
        .live-status-grid { grid-template-columns: 1fr; }
        .brief-hero { padding: 1.25rem; }
        .brief-grid, .brief-meta-grid { grid-template-columns: 1fr; }
        .actions-hero { padding: 1.25rem; }
        .actions-stat-grid, .action-meta-grid { grid-template-columns: 1fr; }
        .admin-hero { padding: 1.25rem; }
        div[data-testid="stForm"] { padding: 1.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def logout():
    st.session_state.authenticated = False
    active_job = st.session_state.get("system_audio_job")
    if active_job and active_job.get("status") == "recording":
        active_job["stop_event"].set()
    for key in (
        "username",
        "graph_token",
        "graph_device_flow",
        "selected_calendar_meeting",
        "live_meeting_title",
        "live_meeting_source",
        "pending_page",
        "microphone_recording",
        "live_mic_recorder_output",
        "system_audio_recording",
        "system_audio_job",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def show_login_screen():
    st.markdown(
        """
        <style>
        section.main > div[data-testid="stVerticalBlockBorderWrapper"],
        .block-container {
            max-width: 1120px;
            padding-top: 5.5rem;
        }
        @media (max-width: 780px) {
            .block-container { padding-top: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    intro_col, form_col = st.columns([1.18, 0.82], gap="large")

    with intro_col:
        st.markdown(
            """
            <div class="login-eyebrow">Meeting intelligence workspace</div>
            <h1 class="login-title">Turn every meeting into clear next steps.</h1>
            <p class="login-copy">
                MeetingOps AI translates multilingual discussions, captures decisions, and keeps
                owners accountable before the follow-up work goes cold.
            </p>
            <div class="login-proof-row">
                <div class="login-proof"><b>Translate</b><span>Tamil, Hindi, and English into one clean transcript.</span></div>
                <div class="login-proof"><b>Summarize</b><span>Brief decisions and risks without digging through notes.</span></div>
                <div class="login-proof"><b>Track</b><span>Save owners, deadlines, and open action items.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.markdown(
            """
            <div class="login-card-heading">
                <h2>Welcome back</h2>
                <p>Sign in to open the MeetingOps command center.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin123")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        if username == DEFAULT_LOGIN_USERNAME and password == DEFAULT_LOGIN_PASSWORD:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    with form_col:
        st.markdown(
            '<div class="login-hint">Demo access: <code>admin</code> / <code>admin123</code></div>',
            unsafe_allow_html=True,
        )


def connect_microsoft_calendar_ui():
    if not is_microsoft_calendar_configured():
        st.warning("Set MS_GRAPH_CLIENT_ID in .env to enable Microsoft Calendar.")
        st.caption("Use a Microsoft Entra public client app with delegated Calendars.Read permission.")
        return

    token = st.session_state.get("graph_token")
    if token and not token_is_valid(token):
        try:
            st.session_state.graph_token = refresh_access_token(token)
            token = st.session_state.graph_token
        except MicrosoftCalendarError as exc:
            st.warning(str(exc))
            st.session_state.pop("graph_token", None)
            token = None

    if token_is_valid(token):
        st.success("Microsoft Calendar connected.")
        if st.button("Disconnect Microsoft Calendar"):
            st.session_state.pop("graph_token", None)
            st.session_state.pop("graph_device_flow", None)
            st.rerun()
        return

    device_flow = st.session_state.get("graph_device_flow")
    if not device_flow:
        if st.button("Connect Microsoft Calendar", type="primary"):
            try:
                st.session_state.graph_device_flow = start_device_login()
                st.rerun()
            except MicrosoftCalendarError as exc:
                st.error(str(exc))
        return

    st.info(device_flow.get("message", "Complete Microsoft sign-in in your browser."))
    st.code(device_flow.get("user_code", ""))
    verification_url = device_flow.get("verification_uri_complete") or device_flow.get("verification_uri")
    if verification_url:
        st.markdown(f"[Open Microsoft sign-in page]({verification_url})")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Complete Microsoft sign-in", type="primary"):
            try:
                result = complete_device_login(device_flow)
                if result["status"] == "complete":
                    st.session_state.graph_token = result["token"]
                    st.session_state.pop("graph_device_flow", None)
                    st.success("Microsoft Calendar connected.")
                    st.rerun()
                else:
                    st.warning(result["message"])
            except MicrosoftCalendarError as exc:
                st.error(str(exc))
    with c2:
        if st.button("Restart sign-in"):
            st.session_state.pop("graph_device_flow", None)
            st.rerun()


def use_meeting_in_live_page(meeting):
    st.session_state.selected_calendar_meeting = meeting
    st.session_state.live_meeting_title = meeting.get("title", "IT Operations Weekly Review")
    st.session_state.live_meeting_source = "MS Teams" if meeting.get("is_teams") else "Manual / In-room"
    st.session_state.pending_page = LIVE_PAGE
    st.rerun()


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    show_login_screen()
    st.stop()

if st.session_state.get("pending_page") in PAGES:
    st.session_state.page = st.session_state.pop("pending_page")

if "page" not in st.session_state or st.session_state.page not in PAGES:
    st.session_state.page = HOME_PAGE


with st.sidebar:
    st.title("🤖 MeetingOps AI")
    st.caption(f"Signed in as {st.session_state.get('username', 'user')}")
    demo_mode = st.toggle("DEMO_MODE", value=os.getenv("DEMO_MODE", "true").lower() == "true")
    page = st.radio("Navigate", PAGES, key="page")
    st.divider()
    if st.button("Logout", use_container_width=True):
        logout()


def show_action_items(items):
    if not items:
        st.info("No action items found.")
        return
    for i, item in enumerate(items, start=1):
        color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(item.get("priority"), "⚪")
        with st.expander(f"{color} {item.get('assignee')} - {item.get('description')}"):
            st.write("Deadline:", item.get("deadline"))
            st.write("Category:", item.get("category"))
            st.write("Status:", item.get("status", "OPEN"))


def parse_meeting_document(document: str, metadata: dict) -> dict:
    title = metadata.get("meeting_title", "")
    source = metadata.get("source", "")
    summary = ""
    transcript = ""

    if "\nTranscript: " in document:
        before_transcript, transcript = document.split("\nTranscript: ", 1)
    else:
        before_transcript = document

    if "\nSummary: " in before_transcript:
        before_summary, summary = before_transcript.split("\nSummary: ", 1)
    else:
        before_summary = before_transcript

    for line in before_summary.splitlines():
        if line.startswith("Title: "):
            title = line.removeprefix("Title: ").strip()
        elif line.startswith("Source: "):
            source = line.removeprefix("Source: ").strip()

    return {
        "title": title,
        "source": source,
        "summary": summary.strip(),
        "transcript": transcript.strip(),
    }


if page == HOME_PAGE:
    _, logout_col = st.columns([5, 1])
    with logout_col:
        if st.button("Exit", use_container_width=True):
            logout()

    st.markdown(
        """
        <div class="home-hero">
            <div class="home-kicker">MeetingOps AI</div>
            <h1 class="home-title">Your meeting ends. <span>The agent starts working.</span></h1>
            <p class="home-copy">
                Capture multilingual conversations, translate them into clear English, extract owners
                and deadlines, then bring the open work back before the next meeting begins.
            </p>
            <div class="home-metric-grid">
                <div class="home-metric"><strong>3</strong><span>Languages handled across Indian IT discussions.</span></div>
                <div class="home-metric"><strong>5</strong><span>Steps from calendar context to action-item tracking.</span></div>
                <div class="home-metric"><strong>1</strong><span>Workspace for summaries, owners, and follow-ups.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-label">The gap</div>
                <h3>Important details vanish after the call.</h3>
                <p>
                    Indian IT meetings often mix English, Tamil, and Hindi. Decisions are scattered,
                    action items get missed, and owners become unclear by the time follow-up starts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-card-label">The system</div>
                <h3>A meeting agent that keeps momentum visible.</h3>
                <p>
                    MeetingOps translates, summarizes, extracts action items, and saves pending work
                    so the next briefing starts with context instead of guesswork.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="home-section-title">MVP Pipeline</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    steps = [
        ("01", "Calendar", "Load the meeting context and attendees."),
        ("02", "Pre-brief", "Surface open work before the call."),
        ("03", "Live transcript", "Capture audio or demo discussion."),
        ("04", "Translate", "Normalize mixed-language speech."),
        ("05", "Action items", "Save owners, deadlines, and status."),
    ]
    for col, (number, title, description) in zip(cols, steps):
        col.markdown(
            f"""
            <div class="pipeline-card">
                <div class="pipeline-step">{number}</div>
                <b>{title}</b>
                <span>{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == LIVE_PAGE:
    st.markdown(
        """
        <div class="live-hero">
            <div class="live-kicker">Live meeting workspace</div>
            <h1 class="live-title">Capture the call. <span>Leave with action.</span></h1>
            <p class="live-copy">
                Run the demo path or record real audio from the meeting.
                MeetingOps turns the discussion into an English transcript, summary, and saved action items.
            </p>
            <div class="live-status-grid">
                <div class="live-status"><b>Input</b><span>Demo, microphone, upload, or Windows loopback audio.</span></div>
                <div class="live-status"><b>AI pass</b><span>Diarize, translate, summarize, and extract owners.</span></div>
                <div class="live-status"><b>Output</b><span>Saved meeting record with pending follow-ups.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_calendar_meeting = st.session_state.get("selected_calendar_meeting")
    if "live_meeting_title" not in st.session_state:
        st.session_state.live_meeting_title = (
            selected_calendar_meeting.get("title") if selected_calendar_meeting else "IT Operations Weekly Review"
        )
    if "live_meeting_source" not in st.session_state:
        st.session_state.live_meeting_source = (
            "MS Teams" if selected_calendar_meeting and selected_calendar_meeting.get("is_teams") else "Manual / In-room"
        )

    if selected_calendar_meeting:
        st.markdown(
            f"""
            <div class="live-calendar-card">
                <b>{selected_calendar_meeting.get('title')}</b>
                <span>Calendar meeting selected for {selected_calendar_meeting.get('start_time')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        attendees = selected_calendar_meeting.get("attendees", [])
        if attendees:
            st.caption("Attendees: " + ", ".join(attendees))
        join_url = selected_calendar_meeting.get("join_url")
        if join_url:
            st.markdown(f"[Open Teams meeting]({join_url})")
        if st.button("Clear selected calendar meeting"):
            st.session_state.pop("selected_calendar_meeting", None)
            st.rerun()

    st.markdown('<div class="live-section-title">Meeting Setup</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="live-section-copy">Name the session, choose the source, then pick the capture path that matches the material you have.</div>',
        unsafe_allow_html=True,
    )
    setup_col, mode_col = st.columns([1.1, 0.9], gap="large")

    with setup_col:
        meeting_title = st.text_input("Meeting title", key="live_meeting_title")
        meeting_source_options = ["Manual / In-room", "MS Teams", "Google Chat / Meet", "Slack Huddle"]
        if st.session_state.live_meeting_source not in meeting_source_options:
            st.session_state.live_meeting_source = "Manual / In-room"
        meeting_source = st.selectbox(
            "Meeting source",
            meeting_source_options,
            key="live_meeting_source",
        )

    with mode_col:
        st.markdown(
            """
            <div class="live-mode-card">
                <b>Choose processing mode</b>
                <span>Use demo for the hackathon path, or audio input for real meeting material.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mode_options = [DEMO_SIMULATION_MODE, MICROPHONE_AUDIO_MODE]
        if supports_system_audio_mode():
            mode_options.append(SYSTEM_AUDIO_MODE)
        elif st.session_state.get("live_capture_mode") == SYSTEM_AUDIO_MODE:
            st.session_state.live_capture_mode = DEMO_SIMULATION_MODE

        mode = st.radio("Mode", mode_options, horizontal=True, key="live_capture_mode")
        if not supports_system_audio_mode():
            st.caption(
                "Windows loopback/headset capture is available only in the local Windows app. "
                "The hosted cloud link supports demo, microphone, upload, and transcript fallback."
            )

    if mode == DEMO_SIMULATION_MODE:
        st.markdown('<div class="live-section-title">Demo Transcript Preview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="live-section-copy">A ready-made mixed-language meeting sample for a fast end-to-end run.</div>',
            unsafe_allow_html=True,
        )
        st.code(format_demo_transcript())
        run = st.button("Run Demo Meeting", type="primary")
        audio = None
        manual_transcript = ""
        use_manual_transcript = False
    elif mode == MICROPHONE_AUDIO_MODE:
        st.markdown('<div class="live-section-title">Capture Input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="live-section-copy">Record browser microphone audio or upload a saved recording.</div>',
            unsafe_allow_html=True,
        )
        st.info("For live room audio, allow microphone access in Edge/Chrome for localhost. For Teams/Meet audio, upload a recording if browser microphone capture is blocked.")
        audio = None
        manual_transcript = ""
        use_manual_transcript = False

        if mic_recorder is None:
            st.warning("Microphone recorder package is not installed. Upload audio to continue.")
        else:
            mic_audio = mic_recorder(
                start_prompt="Record meeting audio",
                stop_prompt="Stop recording",
                just_once=False,
                use_container_width=True,
                format="webm",
                key="live_mic_recorder",
            )
            if mic_audio and mic_audio.get("bytes"):
                audio_format = mic_audio.get("format", "webm")
                st.session_state.microphone_recording = {
                    "bytes": mic_audio["bytes"],
                    "name": f"microphone_recording.{audio_format}",
                }

        uploaded_audio = st.file_uploader(
            "Or upload recorded audio",
            type=["wav", "mp3", "m4a", "mp4", "webm", "ogg"],
        )
        if uploaded_audio is not None:
            audio = uploaded_audio
            st.success(f"Uploaded audio ready: {uploaded_audio.name}")
            st.audio(uploaded_audio.getvalue(), format=audio_mime_type(uploaded_audio.name))
        else:
            recorded_mic_audio = st.session_state.get("microphone_recording")
            if recorded_mic_audio:
                audio = RecordedAudio(recorded_mic_audio["bytes"], recorded_mic_audio["name"])
                st.success("Microphone audio ready.")
                st.audio(recorded_mic_audio["bytes"], format=audio_mime_type(recorded_mic_audio["name"], "audio/webm"))

        manual_transcript = st.text_area(
            "Emergency transcript fallback",
            placeholder="Paste meeting transcript here if audio capture is blocked.",
        )
        use_manual_transcript = st.checkbox(
            "Use pasted transcript instead of audio",
            disabled=not manual_transcript.strip(),
        )
        if use_manual_transcript:
            audio = None

        run = st.button("Process Audio", type="primary", disabled=audio is None and not use_manual_transcript)
    elif mode == SYSTEM_AUDIO_MODE:
        st.markdown('<div class="live-section-title">Capture Input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="live-section-copy">Experimentally record the Windows audio you hear from your headset or speakers.</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "Experimental: this can mix Windows output audio with your microphone. "
            "For Teams, select the Speaker output that carries your colleague voice, and keep microphone mixing enabled for your voice."
        )
        audio = None
        manual_transcript = ""
        use_manual_transcript = False

        active_job = st.session_state.get("system_audio_job")
        is_recording = bool(active_job and active_job.get("status") == "recording")
        device_summary = get_system_audio_device_summary()
        preferred_loopback_device = ""
        preferred_microphone = ""
        include_loopback_microphone = True

        if device_summary.get("error"):
            st.error(device_summary["error"])
        else:
            st.caption(f"Default Windows output: {device_summary.get('speaker', 'default speaker')}")
            st.caption(f"Default microphone: {device_summary.get('default_microphone', 'default microphone')}")
            loopback_options = device_summary.get("loopbacks", [])
            if loopback_options:
                selected_loopback = st.selectbox(
                    "Windows output device to record",
                    ["Auto-detect active Windows output"] + loopback_options,
                    disabled=is_recording,
                )
                preferred_loopback_device = "" if selected_loopback.startswith("Auto-detect") else selected_loopback
                st.caption(
                    "Tip: Auto scans detected output devices. For the safest Teams test, select the exact Speaker device used in Teams."
                )
            else:
                st.error("No Windows loopback output devices were found.")

            include_loopback_microphone = st.checkbox(
                "Also record my microphone for my side of the call",
                value=True,
                disabled=is_recording,
            )
            if include_loopback_microphone:
                microphone_options = device_summary.get("microphones", [])
                if microphone_options:
                    selected_microphone = st.selectbox(
                        "Microphone to mix with loopback audio",
                        ["Auto: default microphone"] + microphone_options,
                        disabled=is_recording,
                    )
                    preferred_microphone = "" if selected_microphone.startswith("Auto:") else selected_microphone
                else:
                    st.error("No microphone devices were found for your side of the call.")

        record_col, stop_col = st.columns(2)
        with record_col:
            if st.button(
                "Start Teams Audio Recording",
                type="primary",
                disabled=is_recording or bool(device_summary.get("error")),
            ):
                st.session_state.pop("system_audio_recording", None)
                st.session_state.system_audio_job = start_system_audio_recording(
                    preferred_device=preferred_loopback_device,
                    include_microphone=include_loopback_microphone,
                    preferred_microphone=preferred_microphone,
                )
                st.rerun()
        with stop_col:
            if st.button("Stop Teams Audio Recording", disabled=not is_recording):
                try:
                    stopped_job = stop_system_audio_recording(active_job)
                    if stopped_job.get("status") == "complete":
                        st.session_state.system_audio_recording = {
                            "bytes": stopped_job["bytes"],
                            "name": stopped_job["name"],
                            "device": stopped_job.get("device", "default speaker"),
                            "microphone_device": stopped_job.get("microphone_device", ""),
                            "include_microphone": stopped_job.get("include_microphone", False),
                            "signal_stats": stopped_job.get("signal_stats", {}),
                            "track_stats": stopped_job.get("track_stats", {}),
                            "warning": stopped_job.get("warning", ""),
                        }
                        st.session_state.pop("system_audio_job", None)
                    else:
                        st.error(stopped_job.get("error", "Teams audio recording failed."))
                        if stopped_job.get("signal_stats"):
                            st.caption(_format_audio_signal_stats(stopped_job["signal_stats"]))
                except Exception as exc:
                    st.error(str(exc))
                st.rerun()

        if is_recording:
            elapsed = int(time.time() - active_job.get("started_at", time.time()))
            device_label = active_job.get("device") or active_job.get("preferred_device") or "Windows output"
            loopback_peak = active_job.get("loopback_peak")
            mic_peak = active_job.get("microphone_peak")
            level_parts = []
            if loopback_peak is not None:
                level_parts.append(f"output peak {loopback_peak:.5f}")
            if mic_peak is not None:
                level_parts.append(f"mic peak {mic_peak:.5f}")
            level_text = " " + ", ".join(level_parts) + "." if level_parts else ""
            st.warning(
                f"Recording {device_label}... {elapsed} seconds.{level_text} "
                "Ask the other person to speak, speak once yourself, then stop and preview the recording."
            )
        elif active_job and active_job.get("status") in {"error", "silent"}:
            st.error(active_job.get("error", "Teams audio recording failed."))
            if active_job.get("signal_stats"):
                st.caption(_format_audio_signal_stats(active_job["signal_stats"]))
            st.session_state.pop("system_audio_job", None)

        recorded_system_audio = st.session_state.get("system_audio_recording")
        if recorded_system_audio:
            audio = RecordedAudio(recorded_system_audio["bytes"], recorded_system_audio["name"])
            success_text = f"Teams audio ready from {recorded_system_audio.get('device', 'Windows output')}"
            if recorded_system_audio.get("microphone_device"):
                success_text += f" + {recorded_system_audio['microphone_device']}"
            st.success(success_text + ".")
            st.caption("Mixed audio: " + _format_audio_signal_stats(recorded_system_audio.get("signal_stats")))
            track_stats = recorded_system_audio.get("track_stats", {})
            if track_stats.get("loopback"):
                st.caption("Output track: " + _format_audio_signal_stats(track_stats["loopback"]))
            if track_stats.get("microphone"):
                st.caption("Microphone track: " + _format_audio_signal_stats(track_stats["microphone"]))
            if recorded_system_audio.get("warning"):
                st.warning(recorded_system_audio["warning"])
            st.audio(recorded_system_audio["bytes"], format="audio/wav")
            st.info("Preview the audio above. If you hear both sides clearly, click Process Teams Audio.")

        run = st.button("Process Teams Audio", type="primary", disabled=audio is None or is_recording)
    if run:
        progress = st.progress(0)
        status = st.empty()
        live_audio_mode = not demo_mode and mode != DEMO_SIMULATION_MODE

        if live_audio_mode and audio is None and not use_manual_transcript:
            status.error("Record, upload, or paste meeting content before processing.")
            st.stop()

        if live_audio_mode and not os.getenv("OPENAI_API_KEY"):
            status.error("Real audio transcription needs OPENAI_API_KEY in .env. Add the key, restart the app, then process again.")
            st.stop()

        status.write("Step 1/5: Transcribing and diarizing...")
        try:
            if live_audio_mode and use_manual_transcript:
                segments = transcript_segments_from_text(manual_transcript)
            else:
                segments = transcribe_audio(audio, demo_mode=(demo_mode or mode == DEMO_SIMULATION_MODE))
        except Exception as exc:
            status.error(str(exc))
            st.stop()
        progress.progress(20)

        status.write("Step 2/5: Translating multilingual transcript...")
        try:
            lang_result = translate_segments(segments, demo_mode=demo_mode)
        except Exception as exc:
            fallback_text = best_recognizer_fallback(segments)
            if not fallback_text:
                status.error(str(exc))
                show_transcription_debug(segments)
                st.stop()
            status.warning("Strict English translation failed. Applying local English fallback to the best recognizer candidate.")
            st.caption(str(exc))
            show_transcription_debug(segments)
            try:
                lang_result = fallback_language_result(segments, fallback_text)
            except Exception:
                st.stop()
        if not lang_result.get("english_transcript", "").strip() or not format_english_transcript_segments(lang_result.get("segments", [])).strip():
            fallback_text = best_recognizer_fallback(segments)
            if not fallback_text:
                status.error("Clean English transcript is empty. Please record again or paste the transcript manually.")
                show_transcription_debug(segments)
                st.stop()
            status.warning("Clean English transcript was empty. Applying local English fallback to the best recognizer candidate.")
            show_transcription_debug(segments)
            try:
                lang_result = fallback_language_result(segments, fallback_text)
            except Exception:
                st.stop()
        progress.progress(45)

        status.write("Step 3/5: Extracting action items...")
        extraction = extract_action_items(lang_result["english_transcript"], demo_mode=demo_mode)
        progress.progress(65)

        status.write("Step 4/5: Generating summary...")
        summary = generate_meeting_summary(lang_result["english_transcript"], extraction, demo_mode=demo_mode)
        progress.progress(85)

        status.write("Step 5/5: Saving to ChromaDB...")
        meeting_id = save_meeting(meeting_title, summary, lang_result["english_transcript"], source=meeting_source)
        ticket_ids = save_action_items(meeting_id, extraction["action_items"])
        progress.progress(100)
        status.success("Meeting processed and saved.")
        st.markdown('<div class="live-section-title">Meeting Intelligence Output</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="live-section-copy">Review the normalized transcript, language details, summary, and saved action items from this run.</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs(["🎭 Transcript", "🌐 Languages", "📝 Summary", "📋 Action Items"])
        with tab1:
            st.subheader("English transcript")
            st.code(format_english_transcript_segments(lang_result["segments"]))
            with st.expander("Raw recognizer output"):
                st.code(format_diarized_transcript(segments))
            show_transcription_candidates(segments)
        with tab2:
            ls = lang_result["language_summary"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Primary", ls.get("primary_language"))
            m2.metric("Languages", len(ls.get("languages_detected", [])))
            m3.metric("Code switching", "Yes" if ls.get("code_switching_detected") else "No")
            for row in lang_result["bilingual_display"]:
                st.markdown(f"**{row['speaker']}** · {row['language']}")
                st.write("English:", row["english"])
                with st.expander("Raw source text"):
                    st.write(row["original"])
                st.divider()
        with tab3:
            st.write(summary)
            risks = extraction.get("key_risks", [])
            st.warning("Risks: " + (", ".join(risks) if risks else "None identified."))
        with tab4:
            st.caption(f"Saved ticket IDs: {', '.join(ticket_ids)}")
            show_action_items(extraction["action_items"])

elif page == BRIEFING_PAGE:
    st.markdown(
        """
        <div class="brief-hero">
            <div class="brief-kicker">Pre-meeting briefing</div>
            <h1 class="brief-title">Walk into the call <span>already caught up.</span></h1>
            <p class="brief-copy">
                Choose a calendar source, review the next meeting, pull pending context from saved
                action items, and send the session straight into Live Meeting.
            </p>
            <div class="brief-grid">
                <div class="brief-tile"><b>Calendar</b><span>Use demo data, a manual Teams link, or Microsoft 365.</span></div>
                <div class="brief-tile"><b>Context</b><span>Surface open actions and similar past tickets.</span></div>
                <div class="brief-tile"><b>Handoff</b><span>Move the selected meeting into the live capture flow.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="brief-section-title">Calendar Source</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brief-section-copy">Start with demo meetings, paste a Teams meeting manually, or connect Microsoft 365 calendar.</div>',
        unsafe_allow_html=True,
    )
    calendar_source = st.radio(
        "Calendar source",
        ["Demo calendar", "Manual Teams meeting", "Microsoft 365 calendar"],
        horizontal=True,
        key="calendar_source",
    )

    meetings = []
    if calendar_source == "Manual Teams meeting":
        st.markdown('<div class="brief-section-title">Manual Meeting Details</div>', unsafe_allow_html=True)
        manual_col_1, manual_col_2 = st.columns(2)
        with manual_col_1:
            manual_title = st.text_input("Teams meeting title", value="Teams Meeting")
            manual_start_time = st.text_input("Start time", value="")
        with manual_col_2:
            manual_attendees = st.text_input("Attendees", placeholder="Name 1, Name 2, Name 3")
            manual_join_url = st.text_input("Teams meeting link", placeholder="Paste the Teams meeting URL")
        meetings = [
            {
                "id": "manual-teams-meeting",
                "title": manual_title.strip() or "Teams Meeting",
                "start_time": manual_start_time.strip() or "Manual entry",
                "end_time": "",
                "attendees": [name.strip() for name in manual_attendees.split(",") if name.strip()],
                "organizer": "",
                "source": "MS Teams",
                "join_url": manual_join_url.strip(),
                "web_link": "",
                "is_online": True,
                "is_teams": True,
                "provider": "manual",
            }
        ]
    elif calendar_source == "Microsoft 365 calendar":
        st.markdown('<div class="brief-section-title">Microsoft Calendar</div>', unsafe_allow_html=True)
        connect_microsoft_calendar_ui()
        token = st.session_state.get("graph_token")
        if token_is_valid(token):
            include_non_teams = st.checkbox("Include non-Teams calendar events", value=False)
            try:
                meetings = get_upcoming_meetings(
                    source="microsoft",
                    access_token=token["access_token"],
                    include_non_teams=include_non_teams,
                )
            except MicrosoftCalendarError as exc:
                st.error(str(exc))
        else:
            st.info("Connect Microsoft Calendar to load upcoming Teams meetings.")
    else:
        meetings = get_upcoming_meetings()

    if not meetings:
        st.info("No upcoming meetings found for this calendar source.")
        st.stop()

    st.markdown('<div class="brief-section-title">Upcoming Meeting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brief-section-copy">Select the meeting you want to brief or send into the live workflow.</div>',
        unsafe_allow_html=True,
    )
    selected_index = st.selectbox(
        "Upcoming meeting",
        range(len(meetings)),
        format_func=lambda index: f"{meetings[index]['start_time']} - {meetings[index]['title']}",
    )
    meeting = meetings[selected_index]

    attendees = meeting.get("attendees", [])
    attendee_text = ", ".join(attendees) if attendees else "None listed"
    organizer_text = meeting.get("organizer") or "Not listed"
    st.markdown(
        f"""
        <div class="brief-meeting-card">
            <h3>{meeting.get('title', 'Untitled meeting')}</h3>
            <div class="brief-meta-grid">
                <div class="brief-meta"><span>Start</span><b>{meeting.get('start_time', '') or 'Not set'}</b></div>
                <div class="brief-meta"><span>End</span><b>{meeting.get('end_time', '') or 'Not set'}</b></div>
                <div class="brief-meta"><span>Source</span><b>{meeting.get('source', '') or 'Unknown'}</b></div>
                <div class="brief-meta"><span>Organizer</span><b>{organizer_text}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Attendees: " + attendee_text)
    if meeting.get("join_url"):
        st.markdown(f"[Open Teams meeting]({meeting['join_url']})")

    st.markdown('<div class="brief-section-title">Briefing Actions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Use this meeting in Live Meeting", type="primary"):
            use_meeting_in_live_page(meeting)
    with c2:
        if st.button("Generate 1-hour briefing"):
            brief = get_pre_meeting_brief(meeting["title"])
            st.success(brief["message"])
            st.markdown('<div class="brief-section-title">Pending Action Items</div>', unsafe_allow_html=True)
            show_action_items(brief["open_items"])
            st.markdown('<div class="brief-section-title">Similar Past Tickets</div>', unsafe_allow_html=True)
            show_action_items(brief["similar_items"])

elif page == ACTION_ITEMS_PAGE:
    items = get_open_action_items()
    total_items = len(items)
    high_priority_count = sum(1 for item in items if item.get("priority") == "HIGH")
    assignee_count = len({item.get("assignee") for item in items if item.get("assignee")})

    st.markdown(
        f"""
        <div class="actions-hero">
            <div class="actions-kicker">Action item queue</div>
            <h1 class="actions-title">Keep follow-ups <span>visible and owned.</span></h1>
            <p class="actions-copy">
                Review open work from processed meetings, filter by owner, and close completed
                commitments as the team moves them forward.
            </p>
            <div class="actions-stat-grid">
                <div class="actions-stat"><strong>{total_items}</strong><span>Open action items waiting for follow-up.</span></div>
                <div class="actions-stat"><strong>{high_priority_count}</strong><span>High-priority items that need attention.</span></div>
                <div class="actions-stat"><strong>{assignee_count}</strong><span>Owners currently carrying open work.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not items:
        st.markdown(
            """
            <div class="empty-actions">
                No open action items yet. Run the demo meeting or process a real meeting to fill this queue.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="actions-section-title">Work Queue</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="actions-section-copy">Filter by assignee, scan priority and deadlines, then close items when they are complete.</div>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(items)
        assignees = ["All"] + sorted(df["assignee"].dropna().unique().tolist())
        assignee = st.selectbox("Filter by assignee", assignees)
        if assignee != "All":
            items = [x for x in items if x.get("assignee") == assignee]

        if not items:
            st.info("No open action items match this filter.")

        for item in items:
            priority = item.get("priority") or "Unspecified"
            assignee_name = item.get("assignee") or "Unassigned"
            deadline = item.get("deadline") or "No deadline"
            category = item.get("category") or "General"
            status_text = item.get("status", "OPEN")
            description = item.get("description") or "No description"
            st.markdown(
                f"""
                <div class="action-card">
                    <h3>{description}</h3>
                    <div class="action-meta-grid">
                        <div class="action-meta"><span>Owner</span><b>{assignee_name}</b></div>
                        <div class="action-meta"><span>Priority</span><b>{priority}</b></div>
                        <div class="action-meta"><span>Deadline</span><b>{deadline}</b></div>
                        <div class="action-meta"><span>Status</span><b>{status_text}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            close_col, detail_col = st.columns([0.22, 0.78])
            with close_col:
                if st.button("Close", key=item["item_id"]):
                    close_action_item(item["item_id"])
                    st.rerun()
            with detail_col:
                st.caption(f"Category: {category}")

elif page == DATA_MANAGER_PAGE:
    status = storage_status()
    meetings = list_meetings()
    all_action_items = list_action_items(include_closed=True)

    st.markdown(
        f"""
        <div class="admin-hero">
            <div class="admin-kicker">Backend data manager</div>
            <h1 class="admin-title">Edit stored meeting data <span>without touching JSON.</span></h1>
            <p class="admin-copy">
                Review and update saved meeting transcripts, summaries, and action-item metadata.
                Current storage mode: <b>{html.escape(status['mode'])}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    storage_col_1, storage_col_2, storage_col_3 = st.columns(3)
    storage_col_1.metric("Meetings", len(meetings))
    storage_col_2.metric("Action items", len(all_action_items))
    storage_col_3.metric("Storage", "JSON" if status["mode"] == "fallback_json" else "ChromaDB")
    st.caption(f"Fallback JSON: {status['fallback_path']}")
    st.caption(f"Chroma path: {status['chroma_path']}")

    meeting_tab, action_tab = st.tabs(["Meetings", "Action Items"])

    with meeting_tab:
        st.markdown('<div class="admin-section-title">Meeting Records</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="admin-section-copy">Update the stored title, source, summary, and transcript for a processed meeting.</div>',
            unsafe_allow_html=True,
        )

        if not meetings:
            st.info("No stored meetings found yet.")
        else:
            meeting_search = st.text_input(
                "Search meetings",
                placeholder="Search by title, date, source, or meeting id",
            ).strip().lower()
            visible_meetings = [
                meeting
                for meeting in meetings
                if not meeting_search
                or meeting_search in meeting.get("meeting_id", "").lower()
                or meeting_search in meeting.get("meeting_title", "").lower()
                or meeting_search in meeting.get("source", "").lower()
                or meeting_search in meeting.get("created_at", "").lower()
            ]
            if not visible_meetings:
                st.info("No meetings match that search.")
                visible_meetings = meetings

            meeting_rows = [
                {
                    "created_at": meeting.get("created_at", ""),
                    "title": meeting.get("meeting_title", ""),
                    "source": meeting.get("source", ""),
                    "meeting_id": meeting.get("meeting_id", ""),
                }
                for meeting in visible_meetings[:25]
            ]
            selected_from_table = None
            meeting_table_key = f"meeting_records_table_{st.session_state.get('meeting_records_table_reset', 0)}"
            table_event = st.dataframe(
                meeting_rows,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key=meeting_table_key,
            )
            selected_rows = table_event.selection.rows if table_event and table_event.selection else []
            selected_rows = [
                row_index
                for row_index in selected_rows
                if isinstance(row_index, int) and 0 <= row_index < len(meeting_rows)
            ]
            selected_meeting_ids = [
                meeting_rows[row_index]["meeting_id"]
                for row_index in selected_rows
            ]
            if selected_meeting_ids:
                st.session_state.data_manager_selected_meeting_ids = selected_meeting_ids
            else:
                st.session_state.pop("data_manager_selected_meeting_ids", None)
                st.session_state.pop("data_manager_selected_meeting_id", None)
            if selected_rows:
                selected_from_table = meeting_rows[selected_rows[0]]["meeting_id"]
            if len(visible_meetings) > 25:
                st.caption(f"Showing 25 of {len(visible_meetings)} matching meetings. Refine the search to narrow the list.")

            if selected_meeting_ids:
                st.caption(f"{len(selected_meeting_ids)} meeting record(s) selected.")
            bulk_delete_col, bulk_confirm_col = st.columns([0.3, 0.7])
            with bulk_confirm_col:
                selected_signature = abs(hash(tuple(selected_meeting_ids)))
                bulk_delete_confirm = st.checkbox(
                    "I understand this will delete the selected meeting records",
                    disabled=not selected_meeting_ids,
                    key=f"bulk_delete_meetings_confirm_{selected_signature}",
                )
            with bulk_delete_col:
                bulk_delete_meetings = st.button(
                    f"Delete selected ({len(selected_meeting_ids)})",
                    disabled=not selected_meeting_ids,
                    key="bulk_delete_meetings",
                )

            if bulk_delete_meetings:
                if not bulk_delete_confirm:
                    st.warning("Tick the bulk delete confirmation checkbox first.")
                else:
                    deleted_count = 0
                    failed_ids = []
                    for meeting_id in selected_meeting_ids:
                        if delete_meeting(meeting_id):
                            deleted_count += 1
                        else:
                            failed_ids.append(meeting_id)
                    st.session_state["meeting_records_table_reset"] = (
                        st.session_state.get("meeting_records_table_reset", 0) + 1
                    )
                    if st.session_state.get("data_manager_selected_meeting_id") in selected_meeting_ids:
                        st.session_state.pop("data_manager_selected_meeting_id", None)
                    st.session_state.pop("data_manager_selected_meeting_ids", None)
                    if failed_ids:
                        st.error(
                            f"Deleted {deleted_count} meeting record(s), but could not delete: {', '.join(failed_ids)}"
                        )
                    else:
                        st.success(f"Deleted {deleted_count} meeting record(s).")
                    st.rerun()

            meeting_ids = [meeting["meeting_id"] for meeting in visible_meetings]
            if not selected_from_table:
                st.info("Select a meeting row from the table to edit or delete a single record.")
            else:
                selected_meeting_id = selected_from_table
                st.session_state.data_manager_selected_meeting_id = selected_meeting_id
                meeting_record = next(row for row in meetings if row["meeting_id"] == selected_meeting_id)
                parsed_meeting = parse_meeting_document(
                    meeting_record.get("document", ""),
                    meeting_record.get("metadata", {}),
                )

                with st.form(f"meeting_editor_{selected_meeting_id}"):
                    st.caption(f"Editing selected meeting: {selected_meeting_id}")
                    edited_title = st.text_input("Meeting title", value=parsed_meeting["title"])
                    edited_source = st.text_input("Source", value=parsed_meeting["source"])
                    edited_summary = st.text_area("Summary", value=parsed_meeting["summary"], height=160)
                    edited_transcript = st.text_area("Transcript", value=parsed_meeting["transcript"], height=260)
                    save_meeting_changes = st.form_submit_button("Save meeting changes", type="primary")
                    delete_meeting_col, delete_meeting_confirm_col = st.columns([0.28, 0.72])
                    with delete_meeting_col:
                        remove_meeting = st.form_submit_button("Delete meeting")
                    with delete_meeting_confirm_col:
                        delete_meeting_confirm = st.checkbox("I understand this will delete this meeting record")

                if save_meeting_changes:
                    if update_meeting(
                        selected_meeting_id,
                        edited_title,
                        edited_summary,
                        edited_transcript,
                        source=edited_source,
                    ):
                        st.success("Meeting record updated.")
                        st.rerun()
                    else:
                        st.error("Could not update meeting record.")

                if remove_meeting:
                    if not delete_meeting_confirm:
                        st.warning("Tick the delete confirmation checkbox first.")
                    elif delete_meeting(selected_meeting_id):
                        st.session_state["meeting_records_table_reset"] = (
                            st.session_state.get("meeting_records_table_reset", 0) + 1
                        )
                        st.session_state.pop("data_manager_selected_meeting_id", None)
                        st.success("Meeting record deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete meeting record.")

    with action_tab:
        st.markdown('<div class="admin-section-title">Action Item Records</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="admin-section-copy">Edit owners, deadlines, priority, category, and status for saved action items.</div>',
            unsafe_allow_html=True,
        )

        selected_meeting_ids_for_items = [
            meeting_id
            for meeting_id in st.session_state.get("data_manager_selected_meeting_ids", [])
            if meeting_id
        ]
        if not selected_meeting_ids_for_items and st.session_state.get("data_manager_selected_meeting_id"):
            selected_meeting_ids_for_items = [st.session_state.data_manager_selected_meeting_id]
        selected_meeting_ids_for_items = [
            meeting_id
            for meeting_id in selected_meeting_ids_for_items
            if any(meeting.get("meeting_id") == meeting_id for meeting in meetings)
        ]
        selected_meeting_titles = [
            meeting.get("meeting_title") or meeting.get("meeting_id", "")
            for meeting in meetings
            if meeting.get("meeting_id") in selected_meeting_ids_for_items
        ]
        action_status_filter = st.radio(
            "Action item status",
            ["All", "Active only", "Closed only"],
            horizontal=True,
            key="data_manager_action_status_filter",
        )
        filter_to_selected_meeting = st.checkbox(
            "Only show action items for selected meeting(s)",
            value=bool(selected_meeting_ids_for_items),
            disabled=not bool(selected_meeting_ids_for_items),
        )
        action_items = list_action_items(include_closed=True)
        if action_status_filter == "Active only":
            action_items = [
                item
                for item in action_items
                if not is_closed_action_status(item.get("status", "OPEN"))
            ]
        elif action_status_filter == "Closed only":
            action_items = [
                item
                for item in action_items
                if is_closed_action_status(item.get("status", "OPEN"))
            ]
        if filter_to_selected_meeting and selected_meeting_ids_for_items:
            selected_meeting_id_set = set(selected_meeting_ids_for_items)
            action_items = [
                item
                for item in action_items
                if item.get("meeting_id") in selected_meeting_id_set
            ]
            if len(selected_meeting_ids_for_items) == 1:
                selected_title = selected_meeting_titles[0] if selected_meeting_titles else selected_meeting_ids_for_items[0]
                st.caption(f"Showing action items linked to: {selected_title} ({selected_meeting_ids_for_items[0]})")
            else:
                st.caption(f"Showing action items linked to {len(selected_meeting_ids_for_items)} selected meetings.")
        elif selected_meeting_ids_for_items:
            if len(selected_meeting_ids_for_items) == 1:
                selected_title = selected_meeting_titles[0] if selected_meeting_titles else selected_meeting_ids_for_items[0]
                st.caption(
                    f"Selected meeting: {selected_title} ({selected_meeting_ids_for_items[0]}). "
                    "Toggle the filter above to show only its action items."
                )
            else:
                st.caption(
                    f"{len(selected_meeting_ids_for_items)} meetings selected. "
                    "Toggle the filter above to show action items from those meetings."
                )

        if not action_items:
            if filter_to_selected_meeting and selected_meeting_ids_for_items:
                st.info("No action items are linked to the selected meeting(s).")
            elif action_status_filter == "Closed only":
                st.info("No closed action items found.")
            elif action_status_filter == "Active only":
                st.info("No active action items found.")
            else:
                st.info("No action items found.")
        else:
            action_rows = [
                {
                    "status": item.get("status", "OPEN"),
                    "assignee": item.get("assignee", ""),
                    "priority": item.get("priority", ""),
                    "deadline": item.get("deadline", ""),
                    "description": item.get("description", ""),
                    "meeting_id": item.get("meeting_id", ""),
                    "item_id": item.get("item_id", ""),
                }
                for item in action_items[:50]
            ]
            selected_item_from_table = None
            action_table_key = f"action_items_table_{st.session_state.get('action_items_table_reset', 0)}"
            action_table_event = st.dataframe(
                action_rows,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key=action_table_key,
            )
            selected_action_rows = action_table_event.selection.rows if action_table_event and action_table_event.selection else []
            selected_action_rows = [
                row_index
                for row_index in selected_action_rows
                if isinstance(row_index, int) and 0 <= row_index < len(action_rows)
            ]
            selected_item_ids = [
                action_rows[row_index]["item_id"]
                for row_index in selected_action_rows
            ]
            if selected_item_ids:
                selected_item_from_table = selected_item_ids[0]
                st.caption(f"{len(selected_item_ids)} action item record(s) selected.")
            if len(action_items) > 50:
                st.caption(f"Showing 50 of {len(action_items)} action items. Use the selected-meeting filter to narrow the list.")

            item_delete_col, item_confirm_col = st.columns([0.3, 0.7])
            with item_confirm_col:
                selected_item_signature = abs(hash(tuple(selected_item_ids)))
                bulk_delete_items_confirm = st.checkbox(
                    "I understand this will delete the selected action item records",
                    disabled=not selected_item_ids,
                    key=f"bulk_delete_action_items_confirm_{selected_item_signature}",
                )
            with item_delete_col:
                bulk_delete_items = st.button(
                    f"Delete selected ({len(selected_item_ids)})",
                    disabled=not selected_item_ids,
                    key="bulk_delete_action_items",
                )

            if bulk_delete_items:
                if not bulk_delete_items_confirm:
                    st.warning("Tick the action-item delete confirmation checkbox first.")
                else:
                    deleted_count = 0
                    failed_ids = []
                    for item_id in selected_item_ids:
                        if delete_action_item(item_id):
                            deleted_count += 1
                        else:
                            failed_ids.append(item_id)
                    st.session_state["action_items_table_reset"] = (
                        st.session_state.get("action_items_table_reset", 0) + 1
                    )
                    if st.session_state.get("data_manager_selected_item_id") in selected_item_ids:
                        st.session_state.pop("data_manager_selected_item_id", None)
                    if failed_ids:
                        st.error(
                            f"Deleted {deleted_count} action item record(s), but could not delete: {', '.join(failed_ids)}"
                        )
                    else:
                        st.success(f"Deleted {deleted_count} action item record(s).")
                    st.rerun()

            if not selected_item_from_table:
                st.info("Select an action item row from the table to edit or delete a single record.")
                st.session_state.pop("data_manager_selected_item_id", None)
            else:
                selected_item_id = selected_item_from_table
                st.session_state.data_manager_selected_item_id = selected_item_id
                item_record = next(row for row in action_items if row["item_id"] == selected_item_id)

                priority_options = ["HIGH", "MEDIUM", "LOW"]
                current_priority = item_record.get("priority", "MEDIUM")
                current_status = item_record.get("status", "OPEN") or "OPEN"
                status_options = ["OPEN", "Not Started", "In Progress", "CLOSED"]
                if current_status not in status_options:
                    status_options.insert(0, current_status)

                with st.form(f"action_editor_{selected_item_id}"):
                    st.caption(f"Editing selected action item: {selected_item_id}")
                    edited_assignee = st.text_input("Assignee", value=item_record.get("assignee", ""))
                    edited_description = st.text_area("Description", value=item_record.get("description", ""), height=130)
                    action_col_1, action_col_2 = st.columns(2)
                    with action_col_1:
                        edited_deadline = st.text_input("Deadline", value=item_record.get("deadline", ""))
                        edited_priority = st.selectbox(
                            "Priority",
                            priority_options,
                            index=priority_options.index(current_priority) if current_priority in priority_options else 1,
                        )
                    with action_col_2:
                        edited_category = st.text_input("Category", value=item_record.get("category", "General"))
                        edited_status = st.selectbox(
                            "Status",
                            status_options,
                            index=status_options.index(current_status) if current_status in status_options else 0,
                        )
                    save_item_changes = st.form_submit_button("Save action item changes", type="primary")
                    delete_item_col, delete_item_confirm_col = st.columns([0.28, 0.72])
                    with delete_item_col:
                        remove_item = st.form_submit_button("Delete action item")
                    with delete_item_confirm_col:
                        delete_item_confirm = st.checkbox("I understand this will delete this action item")

                if save_item_changes:
                    updates = {
                        "assignee": edited_assignee,
                        "description": edited_description,
                        "deadline": edited_deadline,
                        "priority": edited_priority,
                        "category": edited_category,
                        "status": edited_status,
                        "meeting_id": item_record.get("meeting_id", ""),
                        "created_at": item_record.get("created_at", ""),
                    }
                    if update_action_item(selected_item_id, updates):
                        st.success("Action item updated.")
                        st.rerun()
                    else:
                        st.error("Could not update action item.")

                if remove_item:
                    if not delete_item_confirm:
                        st.warning("Tick the delete confirmation checkbox first.")
                    elif delete_action_item(selected_item_id):
                        st.session_state["action_items_table_reset"] = (
                            st.session_state.get("action_items_table_reset", 0) + 1
                        )
                        st.session_state.pop("data_manager_selected_item_id", None)
                        st.success("Action item deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete action item.")
