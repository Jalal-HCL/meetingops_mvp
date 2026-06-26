import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import httpx
from chromadb.config import Settings
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "chroma_store"
FALLBACK_PATH = BASE_DIR / "chroma_fallback.json"
CHROMA_DISABLED_PATH = BASE_DIR / ".chroma_disabled"
_CHROMA_AVAILABLE: Optional[bool] = None


def _ssl_verify() -> bool:
    return os.getenv("OPENAI_SSL_VERIFY", "true").lower() != "false"


def _client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(timeout=120.0, trust_env=False, verify=_ssl_verify()),
        max_retries=2,
    )


def _embed(text: str) -> List[float]:
    if not os.getenv("OPENAI_API_KEY"):
        # deterministic fake embedding for demo/no-key mode
        h = hashlib.sha256(text.encode()).digest()
        return [(b / 255.0) for b in h] * 48
    response = _client().embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def _collection(name: str):
    client = _persistent_client()
    return client.get_or_create_collection(name=name)


def _chroma_available() -> bool:
    global _CHROMA_AVAILABLE
    if _CHROMA_AVAILABLE is not None:
        return _CHROMA_AVAILABLE
    if CHROMA_DISABLED_PATH.exists():
        _CHROMA_AVAILABLE = False
        return False

    try:
        _collection("mo_healthcheck")
        _CHROMA_AVAILABLE = True
        return True
    except Exception as exc:
        _CHROMA_AVAILABLE = False
        CHROMA_DISABLED_PATH.write_text(
            f"ChromaDB disabled after startup failure: {exc}\n",
            encoding="utf-8",
        )
        return False


def _persistent_client():
    db_path = _active_db_path()
    db_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(anonymized_telemetry=False)
    try:
        return chromadb.PersistentClient(path=str(db_path), settings=settings)
    except Exception as exc:
        if "disk I/O error" not in str(exc):
            raise

        recovered_path = DB_PATH.with_name(f"{DB_PATH.name}_recovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        try:
            backup_path = DB_PATH.with_name(f"{DB_PATH.name}_broken_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.move(str(DB_PATH), str(backup_path))
            db_path = DB_PATH
        except PermissionError:
            db_path = recovered_path

        _write_active_db_path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(db_path), settings=settings)


def _active_db_path() -> Path:
    marker = BASE_DIR / ".chroma_db_path"
    if marker.exists():
        saved_path = Path(marker.read_text().strip())
        if saved_path:
            return saved_path
    return DB_PATH


def _write_active_db_path(path: Path) -> None:
    marker = BASE_DIR / ".chroma_db_path"
    marker.write_text(str(path), encoding="utf-8")


def _meeting_doc(meeting_title: str, summary: str, transcript: str, source: str = "Manual") -> str:
    return f"Title: {meeting_title}\nSource: {source}\nSummary: {summary}\nTranscript: {transcript}"


def _action_doc(item: Dict) -> str:
    return f"{item.get('assignee')} must {item.get('description')} by {item.get('deadline')} priority {item.get('priority')} category {item.get('category')} status {item.get('status', 'OPEN')}"


def storage_status() -> Dict:
    return {
        "mode": "chroma" if _chroma_available() else "fallback_json",
        "chroma_path": str(_active_db_path()),
        "fallback_path": str(FALLBACK_PATH),
        "chroma_disabled": CHROMA_DISABLED_PATH.exists(),
    }


def save_meeting(meeting_title: str, summary: str, transcript: str, source: str = "Manual") -> str:
    meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    doc = _meeting_doc(meeting_title, summary, transcript, source)
    embedding = _embed(doc)
    metadata = {"meeting_title": meeting_title, "source": source, "created_at": datetime.now().isoformat()}
    if not _chroma_available():
        _fallback_add("mo_meetings", meeting_id, doc, metadata, embedding)
        return meeting_id

    col = _collection("mo_meetings")
    col.add(
        ids=[meeting_id],
        documents=[doc],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    return meeting_id


def list_meetings() -> List[Dict]:
    if not _chroma_available():
        rows = []
        for meeting_id, row in _fallback_data().get("mo_meetings", {}).items():
            meta = dict(row.get("metadata", {}))
            rows.append(
                {
                    "meeting_id": meeting_id,
                    "document": row.get("document", ""),
                    "meeting_title": meta.get("meeting_title", ""),
                    "source": meta.get("source", ""),
                    "created_at": meta.get("created_at", ""),
                    "metadata": meta,
                }
            )
        return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)

    col = _collection("mo_meetings")
    data = col.get()
    rows = []
    for idx, meeting_id in enumerate(data.get("ids", [])):
        meta = data.get("metadatas", [])[idx] or {}
        rows.append(
            {
                "meeting_id": meeting_id,
                "document": data.get("documents", [])[idx] or "",
                "meeting_title": meta.get("meeting_title", ""),
                "source": meta.get("source", ""),
                "created_at": meta.get("created_at", ""),
                "metadata": dict(meta),
            }
        )
    return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)


def update_meeting(meeting_id: str, meeting_title: str, summary: str, transcript: str, source: str = "Manual") -> bool:
    doc = _meeting_doc(meeting_title, summary, transcript, source)
    metadata = {
        "meeting_title": meeting_title,
        "source": source,
        "updated_at": datetime.now().isoformat(),
    }

    if not _chroma_available():
        data = _fallback_data()
        row = data.get("mo_meetings", {}).get(meeting_id)
        if not row:
            return False
        metadata["created_at"] = row.get("metadata", {}).get("created_at", datetime.now().isoformat())
        row["document"] = doc
        row["metadata"] = metadata
        row["embedding"] = _embed(doc)
        _write_fallback_data(data)
        return True

    col = _collection("mo_meetings")
    data = col.get(ids=[meeting_id])
    if not data.get("ids"):
        return False
    old_meta = data.get("metadatas", [{}])[0] or {}
    metadata["created_at"] = old_meta.get("created_at", datetime.now().isoformat())
    col.update(ids=[meeting_id], documents=[doc], metadatas=[metadata], embeddings=[_embed(doc)])
    return True


def delete_meeting(meeting_id: str) -> bool:
    if not _chroma_available():
        data = _fallback_data()
        if meeting_id not in data.get("mo_meetings", {}):
            return False
        del data["mo_meetings"][meeting_id]
        _write_fallback_data(data)
        return True

    col = _collection("mo_meetings")
    data = col.get(ids=[meeting_id])
    if not data.get("ids"):
        return False
    col.delete(ids=[meeting_id])
    return True


def save_action_items(meeting_id: str, action_items: List[Dict]) -> List[str]:
    chroma_ok = _chroma_available()
    col = _collection("mo_action_items") if chroma_ok else None
    ids = []
    for i, item in enumerate(action_items, start=1):
        item_id = f"item_{meeting_id}_{i}"
        metadata = {
            "meeting_id": meeting_id,
            "assignee": item.get("assignee", ""),
            "description": item.get("description", ""),
            "deadline": item.get("deadline", ""),
            "priority": item.get("priority", "MEDIUM"),
            "category": item.get("category", "General"),
            "status": item.get("status", "OPEN"),
            "created_at": datetime.now().isoformat(),
        }
        text = _action_doc(metadata)
        embedding = _embed(text)
        if col is None:
            _fallback_add("mo_action_items", item_id, text, metadata, embedding)
            ids.append(item_id)
            continue

        col.add(
            ids=[item_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        ids.append(item_id)
    return ids


def list_action_items(include_closed: bool = True) -> List[Dict]:
    if not _chroma_available():
        rows = []
        for item_id, row in _fallback_data().get("mo_action_items", {}).items():
            meta = dict(row.get("metadata", {}))
            if include_closed or meta.get("status") == "OPEN":
                meta["item_id"] = item_id
                meta["document"] = row.get("document", "")
                rows.append(meta)
        return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)

    col = _collection("mo_action_items")
    data = col.get() if include_closed else col.get(where={"status": "OPEN"})
    rows = []
    for idx, item_id in enumerate(data.get("ids", [])):
        meta = dict(data.get("metadatas", [])[idx] or {})
        meta["item_id"] = item_id
        meta["document"] = data.get("documents", [])[idx] or ""
        rows.append(meta)
    return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)


def update_action_item(item_id: str, updates: Dict) -> bool:
    if not _chroma_available():
        data = _fallback_data()
        row = data.get("mo_action_items", {}).get(item_id)
        if not row:
            return False
        meta = dict(row.get("metadata", {}))
        meta.update(updates)
        meta["updated_at"] = datetime.now().isoformat()
        row["metadata"] = meta
        row["document"] = _action_doc(meta)
        row["embedding"] = _embed(row["document"])
        _write_fallback_data(data)
        return True

    col = _collection("mo_action_items")
    data = col.get(ids=[item_id])
    if not data.get("ids"):
        return False
    meta = dict(data.get("metadatas", [{}])[0] or {})
    meta.update(updates)
    meta["updated_at"] = datetime.now().isoformat()
    doc = _action_doc(meta)
    col.update(ids=[item_id], documents=[doc], metadatas=[meta], embeddings=[_embed(doc)])
    return True


def delete_action_item(item_id: str) -> bool:
    if not _chroma_available():
        data = _fallback_data()
        if item_id not in data.get("mo_action_items", {}):
            return False
        del data["mo_action_items"][item_id]
        _write_fallback_data(data)
        return True

    col = _collection("mo_action_items")
    data = col.get(ids=[item_id])
    if not data.get("ids"):
        return False
    col.delete(ids=[item_id])
    return True


def get_open_action_items() -> List[Dict]:
    if not _chroma_available():
        return _fallback_open_items()

    col = _collection("mo_action_items")
    data = col.get(where={"status": "OPEN"})
    items = []
    for idx, item_id in enumerate(data.get("ids", [])):
        meta = data["metadatas"][idx]
        meta["item_id"] = item_id
        items.append(meta)
    return items


def close_action_item(item_id: str) -> bool:
    if not _chroma_available():
        return _fallback_close_item(item_id)

    col = _collection("mo_action_items")
    data = col.get(ids=[item_id])
    if not data.get("ids"):
        return False
    meta = data["metadatas"][0]
    doc = data["documents"][0].replace("status OPEN", "status CLOSED")
    meta["status"] = "CLOSED"
    meta["closed_at"] = datetime.now().isoformat()
    col.update(ids=[item_id], documents=[doc], metadatas=[meta], embeddings=[_embed(doc)])
    return True


def search_similar_tickets(query: str, top_k: int = 5) -> List[Dict]:
    if not _chroma_available():
        return _fallback_search(query, top_k)

    col = _collection("mo_action_items")
    result = col.query(query_embeddings=[_embed(query)], n_results=top_k)
    rows = []
    for i, item_id in enumerate(result.get("ids", [[]])[0]):
        distance = result.get("distances", [[0]])[0][i]
        meta = result.get("metadatas", [[]])[0][i]
        rows.append({"item_id": item_id, "similarity_score": round(max(0, 1 - distance), 2), **meta})
    return rows


def get_pre_meeting_brief(meeting_title: str) -> Dict:
    open_items = get_open_action_items()
    similar = search_similar_tickets(meeting_title, top_k=5) if meeting_title else []
    return {
        "meeting_title": meeting_title,
        "open_items": open_items[:10],
        "similar_items": similar,
        "message": f"Prepare for {meeting_title}. Review pending action items before the call.",
    }


def _fallback_data() -> Dict:
    if not FALLBACK_PATH.exists():
        return {"mo_meetings": {}, "mo_action_items": {}}
    return json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))


def _write_fallback_data(data: Dict) -> None:
    FALLBACK_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fallback_add(collection: str, item_id: str, document: str, metadata: Dict, embedding: List[float]) -> None:
    data = _fallback_data()
    data.setdefault(collection, {})[item_id] = {
        "document": document,
        "metadata": metadata,
        "embedding": embedding,
    }
    _write_fallback_data(data)


def _fallback_open_items() -> List[Dict]:
    items = []
    for item_id, row in _fallback_data().get("mo_action_items", {}).items():
        meta = dict(row.get("metadata", {}))
        if meta.get("status") == "OPEN":
            meta["item_id"] = item_id
            items.append(meta)
    return items


def _fallback_close_item(item_id: str) -> bool:
    data = _fallback_data()
    row = data.get("mo_action_items", {}).get(item_id)
    if not row:
        return False
    row["metadata"]["status"] = "CLOSED"
    row["metadata"]["closed_at"] = datetime.now().isoformat()
    row["document"] = row["document"].replace("status OPEN", "status CLOSED")
    _write_fallback_data(data)
    return True


def _fallback_search(query: str, top_k: int) -> List[Dict]:
    query_embedding = _embed(query)
    rows = []
    for item_id, row in _fallback_data().get("mo_action_items", {}).items():
        score = _cosine_similarity(query_embedding, row.get("embedding", []))
        rows.append({"item_id": item_id, "similarity_score": round(score, 2), **row.get("metadata", {})})
    return sorted(rows, key=lambda item: item["similarity_score"], reverse=True)[:top_k]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = sum(left[i] * left[i] for i in range(size)) ** 0.5
    right_norm = sum(right[i] * right[i] for i in range(size)) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
