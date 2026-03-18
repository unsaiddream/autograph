"""
File-based session storage — each session is a JSON file in sessions/.
No SQLite, no locking issues, works fine with uvicorn --reload.
"""
import json
import time
import uuid
from pathlib import Path

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
SESSION_TTL = 86400  # 24 hours


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def create_session(file_path: str, file_name: str, file_type: str) -> str:
    session_id = str(uuid.uuid4())
    now = time.time()
    data = {
        "id": session_id,
        "created_at": now,
        "updated_at": now,
        "file_path": file_path,
        "file_name": file_name,
        "file_type": file_type,
        "page_count": 0,
        "pages_b64": "[]",
        "zones": "[]",
        "signature_b64": None,
        "stamp_b64": None,
        "fullname": None,
        "sign_date": None,
    }
    _session_path(session_id).write_text(json.dumps(data))
    return session_id


def get_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def update_session(session_id: str, **kwargs):
    session = get_session(session_id)
    if session is None:
        return
    kwargs["updated_at"] = time.time()
    session.update(kwargs)
    _session_path(session_id).write_text(json.dumps(session))


def cleanup_old_sessions():
    cutoff = time.time() - SESSION_TTL
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            if data.get("created_at", 0) < cutoff:
                file_path = data.get("file_path")
                if file_path:
                    Path(file_path).unlink(missing_ok=True)
                path.unlink(missing_ok=True)
        except Exception:
            pass
