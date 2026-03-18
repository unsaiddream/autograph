import sqlite_utils
import json
import time
import uuid
from pathlib import Path

DB_PATH = Path("autograph.db")
SESSION_TTL = 86400  # 24 hours


def get_db():
    db = sqlite_utils.Database(DB_PATH)
    _init_tables(db)
    return db


def _init_tables(db: sqlite_utils.Database):
    if "sessions" not in db.table_names():
        db["sessions"].create({
            "id": str,
            "created_at": float,
            "updated_at": float,
            "file_path": str,
            "file_name": str,
            "file_type": str,
            "page_count": int,
            "pages_b64": str,      # JSON list of base64 page images
            "zones": str,          # JSON list of zone data per page
            "signature_b64": str,  # base64 PNG
            "stamp_b64": str,      # base64 PNG
            "fullname": str,
            "sign_date": str,
        }, pk="id")

    if "saved_signatures" not in db.table_names():
        db["saved_signatures"].create({
            "id": str,
            "created_at": float,
            "name": str,
            "signature_b64": str,
        }, pk="id")


def create_session(file_path: str, file_name: str, file_type: str) -> str:
    db = get_db()
    session_id = str(uuid.uuid4())
    now = time.time()
    db["sessions"].insert({
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
    })
    return session_id


def get_session(session_id: str) -> dict | None:
    db = get_db()
    try:
        row = db["sessions"].get(session_id)
        return dict(row)
    except Exception:
        return None


def update_session(session_id: str, **kwargs):
    db = get_db()
    kwargs["updated_at"] = time.time()
    db["sessions"].update(session_id, kwargs)


def cleanup_old_sessions():
    db = get_db()
    cutoff = time.time() - SESSION_TTL
    old_sessions = list(db["sessions"].rows_where("created_at < ?", [cutoff]))
    for session in old_sessions:
        file_path = session.get("file_path")
        if file_path and Path(file_path).exists():
            Path(file_path).unlink(missing_ok=True)
    db["sessions"].delete_where("created_at < ?", [cutoff])
