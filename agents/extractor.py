import json
import os
import re
from typing import Dict, List

import httpx
from openai import OpenAI


def _ssl_verify() -> bool:
    return os.getenv("OPENAI_SSL_VERIFY", "true").lower() != "false"


def _client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(timeout=120.0, trust_env=False, verify=_ssl_verify()),
        max_retries=2,
    )


def _env_true(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def extract_action_items(english_transcript: str, demo_mode: bool = False) -> Dict:
    return _local_extraction(english_transcript)


def generate_meeting_summary(transcript: str, action_items: Dict, demo_mode: bool = False) -> str:
    return _local_summary(transcript, action_items)


def _validated_model_extraction(transcript: str, model_result: Dict, local_result: Dict) -> Dict:
    if not isinstance(model_result, dict):
        return local_result

    transcript_text = _clean_transcript(transcript)
    valid_items = []
    for item in model_result.get("action_items", []) or []:
        if _action_item_is_grounded(transcript_text, item):
            valid_items.append(item)

    risks = [
        risk for risk in model_result.get("key_risks", []) or []
        if _text_is_grounded(transcript_text, str(risk))
    ]

    model_summary = str(model_result.get("meeting_summary", "")).strip()
    return {
        "meeting_title": model_result.get("meeting_title") or local_result.get("meeting_title", "Meeting Transcript"),
        "action_items": valid_items,
        "key_risks": risks,
        "meeting_sentiment": model_result.get("meeting_sentiment") or local_result.get("meeting_sentiment", "Neutral"),
        "meeting_summary": model_summary if _summary_is_grounded(transcript_text, model_summary) else local_result.get("meeting_summary", ""),
    }


def _action_item_is_grounded(transcript: str, item: Dict) -> bool:
    if not isinstance(item, dict):
        return False
    assignee = str(item.get("assignee", "")).strip()
    description = str(item.get("description", "")).strip()
    if assignee and assignee.lower() not in {"unassigned", "unknown"} and assignee.lower() not in transcript.lower():
        return False
    return _text_is_grounded(transcript, description)


def _summary_is_grounded(transcript: str, summary: str) -> bool:
    transcript_words = _content_words(transcript)
    summary_words = _content_words(summary)
    if not transcript_words or not summary_words:
        return False
    if len(summary_words) > max(60, len(transcript_words) * 5):
        return False
    overlap = set(transcript_words) & set(summary_words)
    return len(overlap) >= min(2, len(set(transcript_words)))


def _text_is_grounded(transcript: str, text: str) -> bool:
    text_words = set(_content_words(text))
    if not text_words:
        return False
    transcript_words = set(_content_words(transcript))
    return bool(text_words & transcript_words)


def _content_words(text: str) -> List[str]:
    stopwords = {
        "about", "after", "again", "also", "and", "are", "been", "but", "can",
        "did", "for", "from", "has", "have", "into", "is", "it", "its", "of",
        "on", "or", "our", "that", "the", "their", "there", "this", "to",
        "was", "were", "what", "when", "where", "who", "why", "will", "with",
    }
    return [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9']+", str(text or ""))
        if len(word) > 2 and word.lower() not in stopwords
    ]


def _local_extraction(transcript: str) -> Dict:
    cleaned = _clean_transcript(transcript)
    sentences = _split_sentences(cleaned)
    action_items = _local_action_items(sentences)
    risks = _local_risks(sentences)
    return {
        "meeting_title": _local_title(cleaned),
        "action_items": action_items,
        "key_risks": risks,
        "meeting_sentiment": _local_sentiment(sentences, risks),
        "meeting_summary": _local_summary(cleaned, {"action_items": action_items, "key_risks": risks}),
    }


def _clean_transcript(transcript: str) -> str:
    lines = []
    for line in str(transcript or "").splitlines():
        line = re.sub(r"^\[[^\]]+\]:\s*", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|;\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _local_title(text: str) -> str:
    lowered = text.lower()
    if "oracle" in lowered or "database" in lowered:
        return "Database Discussion"
    if "backup" in lowered:
        return "Backup Operations Discussion"
    if "server" in lowered:
        return "Server Operations Discussion"
    return "Meeting Transcript"


def _local_action_items(sentences: List[str]) -> List[Dict]:
    items = []
    patterns = [
        re.compile(r"\b(?P<assignee>[A-Z][a-zA-Z]+)\s+will\s+(?P<task>.+?)(?:\s+by\s+(?P<deadline>[^.?!]+))?[.?!]?$", re.IGNORECASE),
        re.compile(r"\b(?P<assignee>[A-Z][a-zA-Z]+),?\s+(?:please\s+)?(?P<task>(?:fix|check|restart|tune|update|review|send|create|complete|investigate|follow up).+?)(?:\s+by\s+(?P<deadline>[^.?!]+))?[.?!]?$", re.IGNORECASE),
        re.compile(r"\b(?P<assignee>[A-Z][a-zA-Z]+)\s+(?:should|must|needs to|has to)\s+(?P<task>.+?)(?:\s+by\s+(?P<deadline>[^.?!]+))?[.?!]?$", re.IGNORECASE),
    ]
    for sentence in sentences:
        if _is_question_only(sentence):
            continue
        for pattern in patterns:
            match = pattern.search(sentence)
            if not match:
                continue
            task = _clean_task(match.group("task"))
            if not task:
                continue
            item = {
                "assignee": _title_name(match.group("assignee")),
                "description": task,
                "deadline": _clean_deadline(match.groupdict().get("deadline")),
                "priority": _priority_for(sentence),
                "category": _category_for(sentence),
                "status": "OPEN",
            }
            if item not in items:
                items.append(item)
            break
    return items


def _is_question_only(sentence: str) -> bool:
    return sentence.strip().endswith("?") and not re.search(r"\b(will|please|by|fix|restart|check|tune)\b", sentence, re.IGNORECASE)


def _clean_task(task: str) -> str:
    task = re.sub(r"\s+", " ", str(task or "")).strip(" .?!")
    task = re.sub(r"^(please\s+|to\s+)", "", task, flags=re.IGNORECASE)
    return task


def _clean_deadline(deadline: str | None) -> str:
    deadline = re.sub(r"\s+", " ", str(deadline or "")).strip(" .?!")
    return deadline or "Not specified"


def _title_name(name: str) -> str:
    return str(name or "Unassigned").strip().title()


def _priority_for(text: str) -> str:
    if re.search(r"\b(critical|urgent|crash|down|failed|failure|blocker|incident|asap)\b", text, re.IGNORECASE):
        return "HIGH"
    if re.search(r"\b(issue|timeout|risk|problem|delay)\b", text, re.IGNORECASE):
        return "MEDIUM"
    return "LOW"


def _category_for(text: str) -> str:
    if re.search(r"\b(oracle|database|db|sql|connection pool)\b", text, re.IGNORECASE):
        return "Database"
    if re.search(r"\b(backup|job|logs|pipeline|deploy)\b", text, re.IGNORECASE):
        return "DevOps"
    if re.search(r"\b(server|network|cpu|memory|disk|infrastructure)\b", text, re.IGNORECASE):
        return "Infrastructure"
    return "General"


def _local_risks(sentences: List[str]) -> List[str]:
    risks = []
    for sentence in sentences:
        if _is_question_only(sentence):
            continue
        if re.search(r"\b(crash|down|failed|failure|timeout|blocked|risk|issue|problem|error|incident|hanging|hung)\b", sentence, re.IGNORECASE):
            risk = sentence.strip(" .")
            if risk and risk not in risks:
                risks.append(risk)
    return risks[:5]


def _local_sentiment(sentences: List[str], risks: List[str]) -> str:
    statement_sentences = [sentence for sentence in sentences if not _is_question_only(sentence)]
    text = " ".join(statement_sentences)
    if re.search(r"\b(critical|urgent|down|crash|blocked|incident)\b", text, re.IGNORECASE):
        return "Critical"
    if risks:
        return "Concerning"
    if re.search(r"\b(good|fine|working|resolved|completed|done)\b", text, re.IGNORECASE):
        return "Positive"
    return "Neutral"


def _local_summary(transcript: str, action_items: Dict) -> str:
    cleaned = _clean_transcript(transcript)
    sentences = _split_sentences(cleaned)
    if not sentences:
        return "No transcript content was available to summarize."

    discussion = " ".join(sentences[:5])
    items = action_items.get("action_items", []) if isinstance(action_items, dict) else []
    risks = action_items.get("key_risks", []) if isinstance(action_items, dict) else []

    if items:
        action_line = "Action items: " + "; ".join(
            f"{item.get('assignee', 'Unassigned')} - {item.get('description', '')}"
            + (f" by {item.get('deadline')}" if item.get("deadline") and item.get("deadline") != "Not specified" else "")
            for item in items
        )
    else:
        action_line = "No explicit action items or owners were assigned in this transcript."

    risk_line = "Risks discussed: " + "; ".join(risks) if risks else "No explicit risks were discussed."
    return f"Discussion: {discussion}\n\n{action_line}\n\n{risk_line}"


def _demo_action_items() -> Dict:
    return {
        "meeting_title": "IT Operations Weekly Review",
        "action_items": [
            {"assignee": "Ramesh", "description": "Restart the crashed server", "deadline": "Thursday", "priority": "HIGH", "category": "Infrastructure", "status": "OPEN"},
            {"assignee": "Jalal", "description": "Tune Oracle connection pool settings", "deadline": "Friday", "priority": "HIGH", "category": "Database", "status": "OPEN"},
            {"assignee": "Priya", "description": "Fix the failed nightly backup job", "deadline": "Wednesday", "priority": "HIGH", "category": "DevOps", "status": "OPEN"},
        ],
        "key_risks": ["Server instability", "Oracle connection timeout", "Recurring backup job failure"],
        "meeting_sentiment": "Concerning",
        "meeting_summary": (
            "The team reviewed current IT operations issues including a server crash, Oracle connection pool timeouts, and repeated nightly backup failures. "
            "These issues are operationally important because they can affect application availability and recovery reliability.\n\n"
            "Clear ownership was assigned during the meeting. Ramesh will restart the crashed server, Jalal will tune the Oracle connection pool settings, and Priya will fix the nightly backup job failure.\n\n"
            "The most critical items are the server restart and backup job fix because they directly impact uptime and recovery. The Oracle connection pool issue should also be tracked closely because repeated timeouts may affect user-facing systems."
        ),
    }
