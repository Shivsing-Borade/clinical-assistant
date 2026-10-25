import sqlite3
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from app.config import settings

class ConversationStore:
    """
    Local SQLite storage for conversation threads and messages.
    """
    def __init__(self, db_path: str = "data/chat_history.sqlite3"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            # Threads table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    route TEXT,
                    evidence_status TEXT,
                    clinician_review_required BOOLEAN DEFAULT 0,
                    caution TEXT,
                    metrics_json TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)")
            conn.commit()

    def create_thread(self, title: str) -> str:
        thread_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, title, now, now)
            )
            conn.commit()
        return thread_id

    def list_threads(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM threads ORDER BY updated_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY timestamp ASC",
                (thread_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def add_message(self, thread_id: str, role: str, text: str, **kwargs) -> str:
        message_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        route = kwargs.get("route")
        evidence_status = kwargs.get("evidence_status")
        clinician_review_required = 1 if kwargs.get("clinician_review_required") else 0
        caution = kwargs.get("caution")
        metrics_json = kwargs.get("metrics_json") # Expected as JSON string

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO messages
                (message_id, thread_id, role, text, route, evidence_status, clinician_review_required, caution, metrics_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (message_id, thread_id, role, text, route, evidence_status, clinician_review_required, caution, metrics_json, now)
            )
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id)
            )
            conn.commit()
        return message_id

    def rename_thread(self, thread_id: str, new_title: str):
        with self._get_connection() as conn:
            conn.execute("UPDATE threads SET title = ? WHERE thread_id = ?", (new_title, thread_id))
            conn.commit()

    def delete_thread(self, thread_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
            conn.commit()

    def clear_messages(self, thread_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            conn.commit()
