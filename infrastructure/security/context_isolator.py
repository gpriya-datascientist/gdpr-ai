"""
Layer: INFRASTRUCTURE — Security
Purpose: Session context isolator — ensures no data bleeds between sessions.
         Each session gets an isolated memory space, wiped on GDPR erase.
"""
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sessions: dict[str, dict] = {}
SESSION_TTL_MINUTES = 60


class ContextIsolator:
    """
    Isolates session context so no data bleeds between users or sessions.
    Implements sliding window conversation history (max 4 turns).
    Auto-expires sessions after TTL.
    """

    def __init__(self, max_turns: int = 4, ttl_minutes: int = SESSION_TTL_MINUTES) -> None:
        self._max_turns = max_turns
        self._ttl = timedelta(minutes=ttl_minutes)

    def get_history(self, session_id: str) -> list[dict]:
        """Get sanitized conversation history for a session."""
        with _lock:
            self._expire_old_sessions()
            session = _sessions.get(session_id, {})
            return session.get("history", [])

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Add a sanitized conversation turn to session history."""
        with _lock:
            if session_id not in _sessions:
                _sessions[session_id] = {"history": [], "created_at": datetime.utcnow()}
            session = _sessions[session_id]
            session["history"].append({"role": "user", "content": user_msg})
            session["history"].append({"role": "assistant", "content": assistant_msg})
            session["last_active"] = datetime.utcnow()
            # Sliding window — keep only last N turns
            if len(session["history"]) > self._max_turns * 2:
                session["history"] = session["history"][-(self._max_turns * 2):]
            logger.debug("Session %s: %d turns in history", session_id, len(session["history"]) // 2)

    def wipe_session(self, session_id: str) -> int:
        """GDPR Art.17 — wipe all session context. Returns turns deleted."""
        with _lock:
            session = _sessions.pop(session_id, {})
            turns = len(session.get("history", [])) // 2
            logger.warning("Context wiped for session %s: %d turns deleted", session_id, turns)
            return turns

    def _expire_old_sessions(self) -> None:
        """Remove sessions that exceeded TTL."""
        now = datetime.utcnow()
        expired = [
            sid for sid, data in _sessions.items()
            if now - data.get("last_active", data.get("created_at", now)) > self._ttl
        ]
        for sid in expired:
            del _sessions[sid]
            logger.info("Session %s expired and wiped", sid)

    def active_session_count(self) -> int:
        with _lock:
            return len(_sessions)
