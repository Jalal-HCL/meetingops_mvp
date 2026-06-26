import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List

import httpx
from openai import APITimeoutError, OpenAI

from agents.demo_data import DEMO_DIARIZED_TRANSCRIPT

TRANSCRIPTION_PROMPT = (
    "This is an Indian IT operations meeting. Speakers may code-switch between "
    "English, Tamil, and Hindi in the same sentence. Transcribe exactly what is "
    "spoken; do not translate, summarize, collapse, or omit repeated phrases. "
    "Actively listen for spoken Tamil/Tanglish and spoken Hindi/Hinglish, even "
    "when they appear after an English phrase. "
    "Use Latin letters only for the transcript. Do not use Tamil, Devanagari, "
    "Urdu, Arabic, Korean, Chinese, or Japanese script. Write Tamil and Hindi "
    "speech in simple Latin transliteration, then keep English words as English. "
    "Do not skip Tamil or Hindi phrases between English phrases. "
    "For example, write 'hello eppadi irukeenga kya kar rahe aap all good', not "
    "the same sounds in Tamil script. Do not convert a Tamil operational question "
    "into the previous English question. For example, if the speaker says "
    "'what is Oracle database? Oracle database sariyaga run aagudha?', transcribe "
    "both parts in order, not 'what is Oracle database' twice. "
    "Preserve names, dates, owners, deadlines, and technical terms such as "
    "Oracle, database, server, backup job, connection pool, incident, SLA, "
    "Teams, Jira, WhatsApp, channel, and logs."
)
TAMIL_FOCUSED_PROMPT = (
    "This audio may contain English, Tamil, and Hindi code-switching. Focus on "
    "capturing any Tamil words or Tamil sentences exactly. Do not omit Tamil "
    "phrases just because an English or Hindi phrase has the same meaning. "
    "Understand spoken Tamil/Tanglish as Tamil content, not as noise and not as "
    "a repeat of nearby English. "
    "Use Latin letters only. Do not use Tamil, Devanagari, Urdu, or Arabic script. "
    "Also keep English words around Tamil phrases so the order is understandable. "
    "Tamil phrases like 'sariyaga run aagudha', 'sariya work aagudha', "
    "'nalla odudha', and 'velai seigiratha' mean running or working fine; do not "
    "rewrite them as the earlier English question. "
    "Return the full transcript in spoken order."
)
HINDI_FOCUSED_PROMPT = (
    "This audio may contain English, Tamil, and Hindi code-switching. Focus on "
    "capturing any Hindi words or Hindi sentences exactly. Do not omit Hindi "
    "phrases just because an English or Tamil phrase has the same meaning. "
    "Understand spoken Hindi/Hinglish as Hindi content, not as noise and not as "
    "a repeat of nearby English. "
    "Use Latin letters only. Do not use Tamil, Devanagari, Urdu, or Arabic script. "
    "Also keep English words around Hindi phrases so the order is understandable. "
    "Return the full transcript in spoken order."
)
DIRECT_AUDIO_REVIEW_PROMPT = (
    "Transcribe this short Indian multilingual audio exactly. The speaker may say "
    "different questions in English, Tamil, and Hindi. Do not collapse a Tamil "
    "question into the previous English question. Use Latin letters only. Keep English as English and write "
    "Tamil/Hindi in simple Latin transliteration. If the speaker asks a question, "
    "write the question exactly; do not answer or explain it. Example: "
    "'what is Oracle database? Oracle database sariyaga run aagudha?' must stay as "
    "those two questions. Return only the transcript."
)
DIRECT_AUDIO_ENGLISH_PROMPT = (
    "Listen to the entire audio from start to end. The speaker may mix English, "
    "Tamil, and Hindi. Translate every spoken phrase into English in the same "
    "order. Understand spoken Tamil/Tanglish and translate its meaning into "
    "English. Understand spoken Hindi/Hinglish and translate its meaning into "
    "English. Keep spoken English as English. Do not replace Tamil or Hindi with "
    "a nearby English phrase unless it is genuinely the same phrase repeated. "
    "Do not omit short English or Hindi phrases at the end. If the speaker "
    "says 'what are you doing' or a Hindi word after a Tamil sentence, include it. "
    "If the speaker asks a question such as 'what is Oracle database', keep it as "
    "the question 'What is Oracle database?'; do not answer the question. "
    "If the speaker then asks in Tamil 'Oracle database sariyaga run aagudha', "
    "translate that second question as 'Is Oracle Database running fine?', not as "
    "another 'What is Oracle Database?'. If the speaker asks in Hindi/Hinglish "
    "'Oracle database sahi se chal raha hai kya', translate it as 'Is Oracle "
    "Database running properly?'. "
    "Return compact JSON only with this shape: "
    '{"english_text":"complete English transcript","languages_detected":["English","Tamil","Hindi"]}.'
)
WHISPER_TRANSLATION_PROMPT = (
    "Indian IT meeting audio. The speaker may mix English, Tamil, Hindi, Tanglish, "
    "and Hinglish. Understand spoken Tamil and translate it to English. Understand "
    "spoken Hindi and translate it to English. Keep spoken English as English. "
    "Translate the entire audio to English. Include every phrase in order, including "
    "short middle phrases and final phrases. Example: Tamil 'sariyaga run aagudha' "
    "means 'is it running fine'; Hindi 'sahi se chal raha hai kya' means 'is it "
    "running properly'. Do not answer questions."
)
MERGE_TRANSCRIPTS_PROMPT = (
    "Merge speech-to-text drafts from the same Indian multilingual audio. "
    "Expected languages are English, Tamil, and Hindi. Keep every distinct phrase "
    "spoken; do not translate. Preserve technical terms such as Oracle, database, "
    "server, backup job, connection pool, incident, SLA, Teams, Jira, and logs. "
    "If one draft contains Tamil or Hindi that another draft missed, include it. "
    "If Hindi/Hindustani appears in Urdu or Arabic script, rewrite it in "
    "Devanagari or simple Latin Hindi. If the speaker asks a question, keep the "
    "question as spoken; do not answer or explain it. If one candidate repeats "
    "'What is Oracle database?' but another contains Tamil like 'sariyaga run "
    "aagudha', keep the Tamil meaning as a separate running-fine question. Return only the merged transcript."
)
DIARIZATION_PROMPT = (
    "Speaker labels came from a diarization pass and transcript text came from "
    "a higher-accuracy prompted transcription pass. Split the accurate transcript "
    "into speaker turns using the diarization draft as timing/order guidance. "
    "Do not add words, translate, or remove code-switched Tamil/Hindi/English. "
    "Return only JSON with a top-level segments array. Each segment must have "
    "speaker, role, and text fields."
)

ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
TAMIL_SCRIPT_RE = re.compile(r"[\u0b80-\u0bff]")
DEVANAGARI_SCRIPT_RE = re.compile(r"[\u0900-\u097f]")
CJK_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _ssl_verify() -> bool:
    return os.getenv("OPENAI_SSL_VERIFY", "true").lower() != "false"


def _transcription_text(transcription) -> str:
    if isinstance(transcription, str):
        return transcription
    text = getattr(transcription, "text", None)
    if text:
        return str(text)
    if isinstance(transcription, dict) and transcription.get("text"):
        return str(transcription["text"])
    return str(transcription)


def _transcription_segments(transcription) -> List[Dict]:
    if isinstance(transcription, dict):
        raw_segments = transcription.get("segments") or []
    else:
        raw_segments = getattr(transcription, "segments", []) or []

    segments = []
    for index, segment in enumerate(raw_segments, start=1):
        if isinstance(segment, dict):
            speaker = segment.get("speaker") or f"Speaker {index}"
            text = segment.get("text") or ""
        else:
            speaker = getattr(segment, "speaker", None) or f"Speaker {index}"
            text = getattr(segment, "text", "") or ""
        text = str(text).strip()
        if text:
            segments.append({"speaker": str(speaker), "role": "Participant", "text": text})
    return segments


def _env_true(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _transcribe_file(client: OpenAI, audio_path: str, prompt: str) -> str:
    return _transcribe_file_with_language(client, audio_path, prompt, None)


def _transcribe_file_with_language(client: OpenAI, audio_path: str, prompt: str, language: str | None) -> str:
    kwargs = {
        "model": os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
        "response_format": os.getenv("OPENAI_TRANSCRIBE_RESPONSE_FORMAT", "text"),
        "prompt": prompt,
    }
    if language:
        kwargs["language"] = language
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(file=audio_file, **kwargs)
    return _transcription_text(transcription).strip()


def _transcribe_file_with_diarization(client: OpenAI, audio_path: str) -> List[Dict]:
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=os.getenv("OPENAI_DIARIZE_MODEL", "gpt-4o-transcribe-diarize"),
            response_format="diarized_json",
            chunking_strategy="auto",
        )
    return _transcription_segments(transcription)


def _direct_audio_review(client: OpenAI, audio_path: str, suffix: str) -> str:
    audio_format = suffix.lower().lstrip(".")
    if audio_format not in {"wav", "mp3"}:
        return ""

    try:
        audio_data = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_AUDIO_REVIEW_MODEL", "gpt-4o-audio-preview"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DIRECT_AUDIO_REVIEW_PROMPT},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": audio_format}},
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _direct_audio_english_translation(client: OpenAI, audio_path: str, suffix: str) -> Dict:
    audio_format = suffix.lower().lstrip(".")
    if audio_format not in {"wav", "mp3"}:
        return {}

    try:
        audio_data = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_AUDIO_REVIEW_MODEL", "gpt-4o-audio-preview"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DIRECT_AUDIO_ENGLISH_PROMPT},
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": audio_format}},
                    ],
                }
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        try:
            data = json.loads(content)
            return {
                "text": str(data.get("english_text", "")).strip(),
                "languages": [str(language) for language in data.get("languages_detected", [])],
            }
        except Exception:
            return {"text": content, "languages": []}
    except Exception:
        return {}


def _audio_translation_to_english(client: OpenAI, audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            translation = client.audio.translations.create(
                model=os.getenv("OPENAI_AUDIO_TRANSLATION_MODEL", "whisper-1"),
                file=audio_file,
                prompt=WHISPER_TRANSLATION_PROMPT,
            )
        return _transcription_text(translation).strip()
    except Exception:
        return ""


def _unique_candidates(candidates: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for candidate in candidates:
        text = candidate.get("text", "").strip()
        key = re.sub(r"\s+", " ", text.lower())
        if text and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _merge_transcript_candidates(client: OpenAI, candidates: List[Dict]) -> str:
    candidates = _unique_candidates(candidates)
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]["text"]

    tamil_candidates = [c for c in candidates if TAMIL_SCRIPT_RE.search(c["text"])]
    if tamil_candidates:
        richest = max(tamil_candidates, key=lambda c: len(c["text"]))
        has_hindi_elsewhere = any(DEVANAGARI_SCRIPT_RE.search(c["text"]) for c in candidates if c is not richest)
        primary_words = set(candidates[0]["text"].lower().split())
        richest_words = set(richest["text"].lower().split())
        if not has_hindi_elsewhere and len(primary_words & richest_words) >= max(2, len(primary_words) // 2):
            return richest["text"]

    drafts = "\n\n".join(f"{candidate['source']}:\n{candidate['text']}" for candidate in candidates)
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSCRIPT_CLEANUP_MODEL", os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o")),
            messages=[
                {"role": "system", "content": MERGE_TRANSCRIPTS_PROMPT},
                {"role": "user", "content": drafts},
            ],
        )
        merged = (response.choices[0].message.content or "").strip()
        return merged or candidates[0]["text"]
    except Exception:
        return candidates[0]["text"]


def _align_diarized_segments(client: OpenAI, accurate_text: str, diarized_segments: List[Dict]) -> List[Dict]:
    if not accurate_text or len(diarized_segments) < 2:
        return []

    draft = "\n".join(
        f"[{segment['speaker']}]: {segment['text']}" for segment in diarized_segments if segment.get("text")
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSCRIPT_CLEANUP_MODEL", os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o")),
            messages=[
                {"role": "system", "content": DIARIZATION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Accurate transcript:\n"
                        f"{accurate_text}\n\n"
                        "Diarization draft:\n"
                        f"{draft}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "{}")
        cleaned = []
        for index, segment in enumerate(data.get("segments", []), start=1):
            text = str(segment.get("text", "")).strip()
            if text:
                cleaned.append(
                    {
                        "speaker": str(segment.get("speaker") or f"Speaker {index}"),
                        "role": str(segment.get("role") or "Participant"),
                        "text": text,
                    }
                )
        return cleaned
    except Exception:
        return []


def _normalize_expected_scripts(client: OpenAI, text: str) -> str:
    if not ARABIC_SCRIPT_RE.search(text):
        return text

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSCRIPT_CLEANUP_MODEL", os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o")),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Clean an Indian meeting transcript after speech recognition. "
                        "Expected languages are English, Tamil, and Hindi only. If any "
                        "Hindi/Hindustani words are rendered in Urdu or Arabic script, "
                        "rewrite those words in Devanagari Hindi or simple Latin Hindi. "
                        "Preserve English, Tamil, names, dates, and technical terms. "
                        "Do not translate the transcript to English."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        normalized = (response.choices[0].message.content or "").strip()
        return normalized or text
    except Exception:
        return text


def _validate_expected_languages(text: str) -> None:
    if CJK_SCRIPT_RE.search(text):
        raise RuntimeError(
            "The transcription appears to be in an unexpected East Asian script. "
            "Expected English, Hindi, or Tamil. Please record at least 8-10 seconds "
            "of clear speech, keep the microphone close, or upload a clearer audio file."
        )

    spoken_tokens = re.findall(r"[\w\u0900-\u097f\u0b80-\u0bff]+", text, flags=re.UNICODE)
    if len(spoken_tokens) < 3:
        raise RuntimeError(
            "The recording was too short or unclear for reliable transcription. "
            "Please record a longer Hindi/English/Tamil sample and process again."
        )


def transcribe_audio(uploaded_audio, demo_mode: bool = False) -> List[Dict]:
    if demo_mode:
        return DEMO_DIARIZED_TRANSCRIPT
    if uploaded_audio is None:
        raise ValueError("No audio was recorded or uploaded.")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env or turn DEMO_MODE on.")

    tmp_path = None
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=httpx.Client(timeout=120.0, trust_env=False, verify=_ssl_verify()),
            max_retries=2,
        )
        suffix = Path(getattr(uploaded_audio, "name", "")).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_audio.getvalue())
            tmp_path = tmp.name

        prompt = os.getenv("OPENAI_TRANSCRIBE_PROMPT", TRANSCRIPTION_PROMPT)
        primary_text = _transcribe_file(client, tmp_path, prompt)
        candidates = [{"source": "gpt-4o-transcribe", "text": primary_text}]
        display_candidates = list(candidates)
        audio_english_hint = ""
        audio_language_hint = []
        audio_translation_hint = ""
        if _env_true("OPENAI_AUDIO_ENGLISH_PASS", True):
            audio_english_result = _direct_audio_english_translation(client, tmp_path, suffix)
            audio_english_hint = audio_english_result.get("text", "")
            audio_language_hint = audio_english_result.get("languages", [])
            if audio_english_hint:
                display_candidates.append({"source": "direct-audio-english", "text": audio_english_hint})

        if _env_true("OPENAI_AUDIO_TRANSLATION_PASS", True):
            audio_translation_hint = _audio_translation_to_english(client, tmp_path)
            if audio_translation_hint:
                display_candidates.append({"source": "whisper-audio-translation", "text": audio_translation_hint})

        if _env_true("OPENAI_LANGUAGE_CANDIDATES", True):
            for candidate in (
                {"source": "tamil-recovery", "text": _transcribe_file_with_language(client, tmp_path, TAMIL_FOCUSED_PROMPT, "ta")},
                {"source": "hindi-recovery", "text": _transcribe_file_with_language(client, tmp_path, HINDI_FOCUSED_PROMPT, "hi")},
            ):
                candidates.append(candidate)
                display_candidates.append(candidate)
            reviewed = _direct_audio_review(client, tmp_path, suffix)
            if reviewed:
                review_candidate = {"source": "audio-review", "text": reviewed}
                candidates.append(review_candidate)
                display_candidates.append(review_candidate)

        text = _merge_transcript_candidates(client, candidates)
        if text and text.strip() != primary_text.strip():
            display_candidates.append({"source": "merged-transcript", "text": text})
        text = text.strip()
        if not text:
            text = audio_translation_hint or audio_english_hint
        if not text:
            raise RuntimeError("OpenAI returned an empty transcription and audio translation was also empty.")

        text = _normalize_expected_scripts(client, text)
        _validate_expected_languages(text)
        segments = []
        if _env_true("OPENAI_ENABLE_DIARIZATION", True):
            try:
                diarized_segments = _transcribe_file_with_diarization(client, tmp_path)
                segments = _align_diarized_segments(client, text, diarized_segments)
            except Exception:
                segments = []

        if not segments:
            segments = [{"speaker": "Unknown", "role": "Participant", "text": text}]

        english_hints = []
        for hint in (audio_english_hint, audio_translation_hint):
            if hint and hint not in english_hints:
                english_hints.append(hint)
        unique_display_candidates = _unique_candidates(display_candidates)
        for segment in segments:
            if english_hints:
                segment["english_hint"] = "\n".join(english_hints)
            if audio_language_hint:
                segment["language_hint"] = audio_language_hint
            segment["transcription_candidates"] = unique_display_candidates
        return segments
    except APITimeoutError as exc:
        raise RuntimeError(
            "Audio transcription timed out while calling OpenAI. "
            "Please try again with a shorter recording or upload the audio file instead of using the browser recorder."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Audio transcription failed: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def format_diarized_transcript(segments: List[Dict]) -> str:
    return "\n".join(f"[{s.get('speaker', 'Unknown')} - {s.get('role', '')}]: {s.get('text', '')}" for s in segments)
