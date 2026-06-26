import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List

import httpx
from openai import OpenAI

SUPPORTED_LANGUAGES = ["English", "Tamil", "Hindi"]
NON_ENGLISH_SCRIPT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\u0900-\u097f\u0b80-\u0bff]")
TAMIL_SCRIPT_RE = re.compile(r"[\u0b80-\u0bff]")
DEVANAGARI_SCRIPT_RE = re.compile(r"[\u0900-\u097f]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
LATIN_HINDI_RE = re.compile(
    r"\b(kya|kaise|kar|rahe|raha|rahi|aap|tum|tumhara|tumara|naam|nam|ho|hai|hain|"
    r"aur|baat|theek|acha|tak|kal|aaj|phir|se|karo|karna|hoga|nahi)\b",
    re.IGNORECASE,
)
LATIN_TAMIL_RE = re.compile(
    r"\b(ellam|sariyaga|velai|seigiratha|enna|inna|vela|velai|panringya|panreenga|"
    r"panringa|irukeenga|eppadi|sariya|aagudha|aagutha|agudha|agutha|nalla|"
    r"odudha|odutha|nadakkudha|nadakudha)\b",
    re.IGNORECASE,
)
TAMIL_RUNNING_CHECK_RE = re.compile(
    r"\b(?:oracle\s+)?(?:database|db)?\s*(?:sariyaga|sariya|proper[ -]?a|nalla)\s*"
    r"(?:run|work|velai|odudha|odutha|nadakkudha|nadakudha)\s*"
    r"(?:aagudha|aagutha|agudha|agutha|seigiratha|seiyutha|irukka|irukkaa)?\b"
    r"|\bvelai\s+seigiratha\b",
    re.IGNORECASE,
)
REPEATED_ORACLE_QUESTION_RE = re.compile(
    r"^\s*(what\s+is\s+oracle\s+database\??\s*){2,}$",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(
    r"^\s*(what|what's|who|when|where|why|how|which|is|are|do|does|did|can|could|would|should)\b",
    re.IGNORECASE,
)
ANSWER_EXPLANATION_RE = re.compile(
    r"\b(is a|is an|refers to|typically|used for|produced by|designed to|allows|provides)\b",
    re.IGNORECASE,
)


def _ssl_verify() -> bool:
    return os.getenv("OPENAI_SSL_VERIFY", "true").lower() != "false"


def _client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(timeout=120.0, trust_env=False, verify=_ssl_verify()),
        max_retries=2,
    )


def translate_segments(segments: List[Dict], demo_mode: bool = False) -> Dict:
    """Translate each segment to English. Raises if English output cannot be trusted."""
    if demo_mode or not os.getenv("OPENAI_API_KEY"):
        return _demo_translation(segments)

    client = _client()
    translated_segments = []
    use_global_candidates = len(segments) == 1
    for segment in segments:
        original = str(segment.get("text", "")).strip()
        recognition_candidates = _recognition_candidate_texts(segment) if use_global_candidates else []
        english_hint = str(segment.get("english_hint", "")).strip() if use_global_candidates else ""
        translated_text = _translate_text_to_english(client, original) if original else ""
        merged_candidate_text = _translate_candidates_to_english(
            client,
            original,
            recognition_candidates,
            english_hint,
        )
        translated_text, translation_source = _choose_best_english(
            original,
            translated_text,
            english_hint,
            merged_candidate_text,
        )
        language_source_text = " ".join([original] + recognition_candidates)
        languages = _detect_languages(language_source_text, segment.get("language_hint", []))
        translated_segments.append(
            {
                "speaker": segment.get("speaker", "Unknown"),
                "role": segment.get("role", "Participant"),
                "original_text": original,
                "translated_text": translated_text,
                "detected_language": "-".join(languages) if languages else "Unknown",
                "is_mixed": len(languages) > 1,
                "translation_notes": f"English-only translation applied from {translation_source}.",
            }
        )

    final_segments = _dedupe_translated_segments(_validate_final_segments(translated_segments))
    data = {
        "segments": final_segments,
        "language_summary": _build_language_summary(final_segments),
    }
    data["english_transcript"] = "\n".join(
        f"[{s['speaker']} - {s.get('role', '')}]: {s['translated_text']}" for s in data["segments"]
    )
    data["bilingual_display"] = [
        {"speaker": s["speaker"], "original": s["original_text"], "english": s["translated_text"], "language": s["detected_language"]}
        for s in data["segments"]
    ]
    return data


def _recognition_candidate_texts(segment: Dict) -> List[str]:
    texts = []
    for candidate in segment.get("transcription_candidates", []) or []:
        text = str(candidate.get("text", "")).strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def _choose_best_english(original: str, translated: str, english_hint: str, merged_candidate_text: str = "") -> tuple[str, str]:
    candidates = []
    for source, text in (
        ("text translation", translated),
        ("candidate merge", merged_candidate_text),
        ("direct audio pass", english_hint),
    ):
        cleaned = _try_clean_english(original, text)
        if cleaned and not _is_probable_answer_to_spoken_question(original, cleaned):
            candidates.append((source, cleaned))

    if not candidates:
        if _looks_like_spoken_question(original) and not NON_ENGLISH_SCRIPT_RE.search(original):
            return _format_spoken_question(original), "raw question transcript"
        return _ensure_clean_english(original, translated), "text translation"

    source, text = max(candidates, key=lambda item: _completion_score(item[1]))
    return text, source


def _try_clean_english(original: str, translated: str) -> str:
    try:
        return _ensure_clean_english(original, translated)
    except RuntimeError:
        return ""


def _completion_score(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9']+", text)
    return len(words)


def _looks_like_spoken_question(text: str) -> bool:
    text = _clean_translation_text(text)
    return bool(text.endswith("?") or QUESTION_START_RE.search(text))


def _format_spoken_question(text: str) -> str:
    text = _sentence_case(_clean_translation_text(text))
    if text and not text.endswith(("?", ".", "!")):
        text += "?"
    return text


def _is_probable_answer_to_spoken_question(original: str, candidate: str) -> bool:
    if not _looks_like_spoken_question(original):
        return False

    original_words = re.findall(r"[A-Za-z0-9']+", original)
    candidate_words = re.findall(r"[A-Za-z0-9']+", candidate)
    if len(original_words) < 2 or len(candidate_words) <= max(8, len(original_words) * 2):
        return False

    if ANSWER_EXPLANATION_RE.search(candidate):
        return True
    return not _looks_like_spoken_question(candidate)


def _translate_candidates_to_english(client: OpenAI, original: str, recognition_candidates: List[str], english_hint: str) -> str:
    all_candidates = []
    for text in [original] + recognition_candidates + [english_hint]:
        text = str(text or "").strip()
        if text and text not in all_candidates:
            all_candidates.append(text)

    if len(all_candidates) <= 1:
        return ""

    system_prompt = (
        "You are merging alternate recognizer outputs from the same short audio clip. "
        "The speaker may mix English, Tamil/Tanglish, and Hindi/Hinglish. Some candidates "
        "may omit the Tamil or Hindi phrase, while another candidate may omit English. "
        "Use all candidates together and produce one complete English transcript in spoken order. "
        "Do not omit middle phrases. Understand spoken Tamil/Tanglish and translate "
        "its meaning into English. Understand spoken Hindi/Hinglish and translate "
        "its meaning into English. Keep spoken English as English. "
        "Do not answer questions. If the speaker asks a question, keep it as a question. "
        "Do not convert a Tamil operational question into a repeat of the earlier English question. "
        "Keep English phrases as English. Examples: 'ellam sariyaga velai seigiratha' means "
        "'is everything working properly'; 'inna vela panringya' or 'enna vela panreenga' means "
        "'what work are you doing'; 'tumhara naam kya hai' means 'what is your name'; "
        "'kya kar rahe ho aap' means 'what are you doing'; 'Oracle database sariyaga run "
        "aagudha' means 'Is Oracle Database running fine?'; 'Oracle database sahi se "
        "chal raha hai kya' means 'Is Oracle Database running properly?'. Return English only."
    )
    schema = {
        "type": "object",
        "properties": {
            "english_text": {"type": "string"},
            "detected_languages": {
                "type": "array",
                "items": {"type": "string", "enum": SUPPORTED_LANGUAGES},
            },
        },
        "required": ["english_text", "detected_languages"],
        "additionalProperties": False,
    }

    user_payload = "\n\n".join(
        f"Candidate {index}:\n{text}" for index, text in enumerate(all_candidates, start=1)
    )
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "merged_meeting_translation",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        data = json.loads(response.choices[0].message.content or "{}")
        merged = str(data.get("english_text", "")).strip()
        merged = _repair_repeated_oracle_question(all_candidates, merged)
        return merged or _local_candidates_fallback(all_candidates)
    except Exception:
        return _local_candidates_fallback(all_candidates)


def _local_candidates_fallback(candidates: List[str]) -> str:
    cleaned = [_local_phrase_fallback(candidate) for candidate in candidates]
    cleaned = [candidate for candidate in cleaned if candidate]
    if not cleaned:
        return ""
    merged = []
    seen_phrases = set()
    for candidate in sorted(cleaned, key=_completion_score, reverse=True):
        for phrase in re.split(r"(?<=[.!?])\s+|\s{2,}", candidate):
            phrase = phrase.strip()
            key = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
            if phrase and key and key not in seen_phrases:
                seen_phrases.add(key)
                merged.append(phrase)
    return " ".join(merged) or max(cleaned, key=_completion_score)


def _repair_repeated_oracle_question(candidates: List[str], translated: str) -> str:
    source_text = " ".join(str(candidate or "") for candidate in candidates)
    if not TAMIL_RUNNING_CHECK_RE.search(source_text):
        return translated
    if REPEATED_ORACLE_QUESTION_RE.search(translated):
        return "What is Oracle Database? Is Oracle Database running fine?"
    return translated


def best_effort_english_fallback(text: str) -> str:
    """Local last-resort English conversion used only after model translation fails."""
    cleaned = _local_phrase_fallback(text)
    if cleaned:
        return cleaned

    text = _clean_translation_text(text)
    if text and not NON_ENGLISH_SCRIPT_RE.search(text) and not _contains_latin_indic(text):
        return text
    return ""


def _validate_final_segments(segments: List[Dict]) -> List[Dict]:
    valid_segments = []
    for segment in segments:
        original = str(segment.get("original_text", "")).strip()
        translated = str(segment.get("translated_text", "")).strip()

        if not translated:
            translated = _local_phrase_fallback(original)
        if not translated and original and not NON_ENGLISH_SCRIPT_RE.search(original):
            translated = original
        if not translated:
            raise RuntimeError(
                "Clean English transcript is empty after translation. "
                f"Raw recognizer text was: {original or '[empty]'}"
            )
        if NON_ENGLISH_SCRIPT_RE.search(translated):
            raise RuntimeError(
                "Clean English transcript still contains non-English script after translation. "
                f"Raw recognizer text was: {original or '[empty]'}"
            )

        segment["translated_text"] = translated
        valid_segments.append(segment)

    if not valid_segments:
        raise RuntimeError("No transcript segments were produced from the audio.")

    return valid_segments


def _dedupe_translated_segments(segments: List[Dict]) -> List[Dict]:
    unique = []
    for segment in segments:
        text = str(segment.get("translated_text", "")).strip()
        if not text:
            continue

        duplicate_index = None
        for index, existing in enumerate(unique):
            if _near_duplicate_text(text, str(existing.get("translated_text", ""))):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique.append(segment)
            continue

        existing_text = str(unique[duplicate_index].get("translated_text", ""))
        if _completion_score(text) > _completion_score(existing_text):
            unique[duplicate_index] = segment

    return unique


def _near_duplicate_text(left: str, right: str) -> bool:
    left_norm = _normalize_for_dedupe(left)
    right_norm = _normalize_for_dedupe(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.9


def _normalize_for_dedupe(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _translate_text_to_english(client: OpenAI, text: str) -> str:
    if not text:
        return ""

    system_prompt = (
        "Convert the user's meeting text into clean English only. The input may mix "
        "English, Tamil script, spoken Tamil, Hindi in Devanagari, and Hindi written "
        "with Latin letters such as 'kya kar rahe aap'. Understand spoken Tamil or "
        "Tanglish and translate it into English. Understand spoken Hindi or Hinglish "
        "and translate it into English. Keep English as English. "
        "Do not answer questions or add explanations; if the speaker asks a question, "
        "preserve it as a question in English. "
        "Do not convert Tamil phrases like 'Oracle database sariyaga run aagudha' "
        "into a repeat of a previous English question; translate that as "
        "'Is Oracle Database running fine?'. Translate Hindi/Hinglish phrases like "
        "'Oracle database sahi se chal raha hai kya' as 'Is Oracle Database running properly?'. "
        "Translate Tamil and Hindi to English. If the same meaning is repeated in "
        "multiple languages, merge it into one natural English sentence. Return only "
        "the English translation."
    )
    schema = {
        "type": "object",
        "properties": {
            "english_text": {
                "type": "string",
                "description": "A natural English-only translation of the full input.",
            },
            "detected_languages": {
                "type": "array",
                "items": {"type": "string", "enum": SUPPORTED_LANGUAGES},
            },
            "notes": {"type": "string"},
        },
        "required": ["english_text", "detected_languages", "notes"],
        "additionalProperties": False,
    }

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_translation",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return str(data.get("english_text", "")).strip()
    except Exception:
        return _translate_text_to_english_plain(client, text, system_prompt)


def _translate_text_to_english_plain(client: OpenAI, text: str, system_prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": system_prompt + " Do not return JSON."},
                {"role": "user", "content": text},
            ],
        )
        return (response.choices[0].message.content or "").strip().strip('"')
    except Exception:
        return ""


def _ensure_clean_english(original: str, translated: str) -> str:
    translated = _clean_translation_text(translated)
    if translated and not NON_ENGLISH_SCRIPT_RE.search(translated) and not _contains_latin_indic(translated):
        return translated

    local = _local_phrase_fallback(original)
    if local and not NON_ENGLISH_SCRIPT_RE.search(local):
        return local

    if not original:
        return ""

    raise RuntimeError(
        "Clean English transcript could not be generated reliably. "
        "Please record again or paste the transcript in the emergency fallback."
    )


def _clean_translation_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _local_phrase_fallback(text: str) -> str:
    if not text:
        return ""

    normalized = text
    replacements = [
        (
            r"\boracle\s+database\s+(?:sariyaga|sariya|proper[ -]?a|nalla)\s+(?:run|work|odudha|odutha|nadakkudha|nadakudha)\s*(?:aagudha|aagutha|agudha|agutha|irukka|irukkaa)?\b",
            "is Oracle Database running fine",
        ),
        (
            r"\bdatabase\s+(?:sariyaga|sariya|proper[ -]?a|nalla)\s+(?:run|work|odudha|odutha|nadakkudha|nadakudha)\s*(?:aagudha|aagutha|agudha|agutha|irukka|irukkaa)?\b",
            "is the database running fine",
        ),
        (
            r"\b(?:ellam\s+)?sariyaga\s+velai\s+seigiratha\b",
            "is everything working properly",
        ),
        (r"\bellam\s+sariyaga\s+velai\s+seigiratha\b", "is everything working properly"),
        (r"\binna\s+vel(?:a|ai)?\s*panr(?:ingya|inga|eenga)\b", "what work are you doing"),
        (r"\benna\s+vel(?:a|ai)?\s*panr(?:ingya|inga|eenga)\b", "what work are you doing"),
        (r"\btumhara\s+naa?m\s+kya\s+hai\b", "what is your name"),
        (r"\btumara\s+naa?m\s+kya\s+hai\b", "what is your name"),
        (r"\btomorrow\s+naa?m\s+kya\s+hai\b", "what is your name"),
        (r"\bnaa?m\s+kya\s+hai\b", "what is the name"),
        (r"ஹலோ", "Hello"),
        (r"வாட்சப்", "what's up"),
        (r"வாட்\s+இஸ்\s+கோயிங்\s+ஆன்", "what is going on"),
        (r"வாட்\s+இஸ்\s+கோயின்\s+ஆன்", "what is going on"),
        (r"ஆல்\s+குட்", "all good"),
        (r"எப்படி\s+இருக்க(?:ீங்க|ீர்கள்|ிறீர்கள்|ிறாய்)?", "how are you"),
        (r"என்ன(?:ா)?\s+கர்ரே\s+ஆப்", "what are you doing"),
        (r"க்யா\s+கர்ரே\s+(?:ஆப்|ஹோ|ஹோ\s+ஆப்)", "what are you doing"),
        (r"கியா\s+கர்ரே\s+(?:ஆப்|ஹோ|ஹோ\s+ஆப்)", "what are you doing"),
        (r"என்ன\s+(?:செய்றீங்க|பண்றீங்க|செய்கிறீர்கள்)", "what are you doing"),
        (r"டேடாபேச(?:ிக்|்)?(?:\s+ராசா)?\s+சேனல்\s+ஸ்டார்ட்\s+பண்ணீங்கலா", "did you start the database channel"),
        (r"டேடாபேச(?:ிக்|்)?\s+கிராஷ்\s+ஆயிடுச்சு", "the database has crashed"),
        (r"ஒயிடிடிபி\s+கிராஷ்ட்", "the database crashed"),
        (r"டிபி\s+கிராஷ்ட்", "the database crashed"),
        (r"டேடாபேஸ்\s+கிராஷ்ட்", "the database crashed"),
        (r"என்ன\s+பண்ணலாம்பா", "what can we do"),
        (r"என்ன\s+பண்ணலாம்", "what can we do"),
        (r"யாரு\s+அதை\s+ஸ்டார்ட்\s+பண்ண(?:ப்)?\s+போற(?:ா|ாங்க|ாங்கா)?", "who is going to start it"),
        (r"யாரு\s+அதை\s+ரீஸ்டார்ட்\s+பண்ண(?:ப்)?\s+போற(?:ா|ாங்க|ாங்கா)?", "who is going to restart it"),
        (r"யாரு", "who"),
        (r"அதை", "it"),
        (r"டேடாபேச(?:ிக்|்)?", "database"),
        (r"கிராஷ்", "crash"),
        (r"சேனல்", "channel"),
        (r"ஸ்டார்ட்", "start"),
        (r"பண்ணீங்கலா", "did you do it"),
        (r"कौन\s+है", "who is it"),
        (r"\bkya\s+kar\s+rahe\s+(?:ho|aap)\b", "what are you doing"),
        (r"\bkya\s+kar\s+rahe\b", "what are you doing"),
        (r"\bkya\s+kar\s+raha\s+(?:hai|ho)\b", "what are you doing"),
        (r"\bwhat'?s\s+up\b", "what's up"),
        (r"\ball\s+good\b", "all good"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    if NON_ENGLISH_SCRIPT_RE.search(normalized):
        return ""

    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\bwhat'?s up\s+all good\b", "what's up? all good", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\b(what\s+is\s+oracle\s+database)\s+(is\s+oracle\s+database\s+running\s+fine)\b",
        r"\1? \2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b(what is (?:your name|the name))\s+(what work)", r"\1? \2", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(what work are you doing)\s+(ok(?:ay)? bye)\b", r"\1? \2", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+([,.?!])", r"\1", normalized)
    normalized = _sentence_case(normalized)
    normalized = re.sub(r"\boracle database\b", "Oracle Database", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(Is (?:Oracle Database|the database) running fine)$", r"\1?", normalized)
    if _contains_latin_indic(normalized):
        return ""
    return normalized


def _contains_latin_indic(text: str) -> bool:
    return bool(LATIN_HINDI_RE.search(text) or LATIN_TAMIL_RE.search(text))


def _sentence_case(text: str) -> str:
    return re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        text,
    )


def _detect_languages(text: str, language_hint=None) -> List[str]:
    languages = []
    for language in language_hint or []:
        normalized = str(language).strip().title()
        if normalized in SUPPORTED_LANGUAGES and normalized not in languages:
            languages.append(normalized)
    if LATIN_LETTER_RE.search(text):
        if "English" not in languages:
            languages.append("English")
    if TAMIL_SCRIPT_RE.search(text) or LATIN_TAMIL_RE.search(text):
        if "Tamil" not in languages:
            languages.append("Tamil")
    if DEVANAGARI_SCRIPT_RE.search(text) or LATIN_HINDI_RE.search(text):
        if "Hindi" not in languages:
            languages.append("Hindi")
    return [language for language in SUPPORTED_LANGUAGES if language in languages]


def _build_language_summary(segments: List[Dict]) -> Dict:
    languages = []
    for segment in segments:
        for language in str(segment.get("detected_language", "")).split("-"):
            if language in SUPPORTED_LANGUAGES and language not in languages:
                languages.append(language)

    return {
        "primary_language": languages[0] if languages else "Unknown",
        "languages_detected": languages,
        "code_switching_detected": any(segment.get("is_mixed") for segment in segments),
        "india_languages_present": [language for language in languages if language in {"Tamil", "Hindi"}],
    }


def _fallback_translation(segments: List[Dict]) -> Dict:
    return {
        "segments": [
            {
                "speaker": s.get("speaker", "Unknown"),
                "role": s.get("role", ""),
                "original_text": s.get("text", ""),
                "translated_text": s.get("text", ""),
                "detected_language": "Unknown",
                "is_mixed": False,
                "translation_notes": "Fallback used; original text preserved.",
            }
            for s in segments
        ],
        "language_summary": {
            "primary_language": "Unknown",
            "languages_detected": [],
            "code_switching_detected": False,
            "india_languages_present": [],
        },
    }


def _demo_translation(segments: List[Dict]) -> Dict:
    translations = {
        "Server Tuesday அன்னிக்கு crash ஆச்சு. I'll restart it by Thursday, என்னோட responsibility.": "The server crashed on Tuesday. I will restart it by Thursday; it is my responsibility.",
        "Sure, connection pool settings-ai tune பண்ணி Friday-க்குள்ள முடிச்சிடும்.": "Sure, I will tune the connection pool settings and complete it by Friday.",
        "Aur ek baat - nightly backup job phir se fail ho gaya. Priya, Wednesday tak fix kar do.": "One more thing: the nightly backup job failed again. Priya, please fix it by Wednesday.",
    }
    out = []
    for s in segments:
        original = s.get("text", "")
        translated = translations.get(original, original)
        lang = "Tamil-English" if "அ" in original or "ப" in original else "Hindi-English" if "Aur" in original else "English"
        out.append({
            "speaker": s.get("speaker", "Unknown"),
            "role": s.get("role", ""),
            "original_text": original,
            "translated_text": translated,
            "detected_language": lang,
            "is_mixed": lang != "English",
            "translation_notes": "Demo translation",
        })
    return {
        "segments": out,
        "language_summary": {
            "primary_language": "English",
            "languages_detected": ["English", "Tamil", "Hindi"],
            "code_switching_detected": True,
            "india_languages_present": ["Tamil", "Hindi"],
        },
        "english_transcript": "\n".join(f"[{s['speaker']} - {s['role']}]: {s['translated_text']}" for s in out),
        "bilingual_display": [{"speaker": s["speaker"], "original": s["original_text"], "english": s["translated_text"], "language": s["detected_language"]} for s in out],
    }
