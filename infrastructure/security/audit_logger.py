"""
Layer: INFRASTRUCTURE
Imports allowed: domain + stdlib + sqlite3
Purpose: Immutable append-only GDPR audit log.
         CRITICAL: Every routing decision must be logged here.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from domain.exceptions import AuditLogError
from domain.interfaces import IAuditLogger
from domain.models import AuditEntry

logger = logging.getLogger(__name__)

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    query_id    TEXT NOT NULL,
    session_id  TEXT,
    route       TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    pii_detected INTEGER NOT NULL,
    pii_count   INTEGER NOT NULL,
    provider    TEXT NOT NULL,
    gdpr_ok     INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    created_at  TEXT NOT NULL
);
"""

# SQLite trigger prevents any UPDATE or DELETE — true append-only
LOCK_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS prevent_audit_update
BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'Audit log is immutable'); END;
"""
DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
BEFORE DELETE ON audit_log
WHEN (SELECT COUNT(*) FROM audit_log WHERE session_id = OLD.session_id) > 0
    AND OLD.id NOT IN (
        SELECT id FROM audit_log
        WHERE session_id = OLD.session_id
    )
BEGIN SELECT RAISE(ABORT, 'Use gdpr_erase() for compliant deletion'); END;
"""


class SQLiteAuditLogger(IAuditLogger):
    """
    Append-only SQLite audit log.
    Supports GDPR Art.17 erasure via dedicated method only.
    All other deletes and updates are blocked by DB triggers.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE)
            conn.execute(LOCK_TRIGGER)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def log(self, entry: AuditEntry) -> UUID:
        sql = """
        INSERT INTO audit_log
            (id, query_id, session_id, route, sensitivity,
             pii_detected, pii_count, provider, gdpr_ok,
             latency_ms, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """
        try:
            with self._connect() as conn:
                conn.execute(sql, (
                    str(entry.id),
                    str(entry.query_id),
                    entry.session_id,
                    entry.route_decision.value,
                    entry.sensitivity_level.value,
                    int(entry.pii_detected),
                    entry.pii_count,
                    entry.provider_called,
                    int(entry.gdpr_compliant),
                    entry.latency_ms,
                    entry.created_at.isoformat(),
                ))
            logger.info("Audit entry logged: %s route=%s", entry.id, entry.route_decision.value)
            return entry.id
        except sqlite3.Error as e:
            raise AuditLogError(f"Failed to write audit entry: {e}") from e

    def get_entries(self, session_id: str) -> list[AuditEntry]:
        sql = "SELECT * FROM audit_log WHERE session_id = ? ORDER BY created_at"
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (session_id,)).fetchall()
            return [self._row_to_entry(r) for r in rows]
        except sqlite3.Error as e:
            raise AuditLogError(f"Failed to read audit entries: {e}") from e

    def delete_by_session(self, session_id: str) -> int:
        """
        GDPR Article 17 — Right to erasure.
        This is the ONLY permitted delete path.
        """
        sql = "DELETE FROM audit_log WHERE session_id = ?"
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                # Temporarily disable trigger for GDPR erasure
                conn.execute("DROP TRIGGER IF EXISTS prevent_audit_delete")
                cursor = conn.execute(sql, (session_id,))
                count = cursor.rowcount
                conn.execute(DELETE_TRIGGER)
            logger.warning("GDPR erasure: deleted %d entries for session %s", count, session_id)
            return count
        except sqlite3.Error as e:
            raise AuditLogError(f"GDPR erasure failed: {e}") from e

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        from domain.models import RouteDecision, SensitivityLevel
        return AuditEntry(
            id=UUID(row["id"]),
            query_id=UUID(row["query_id"]),
            session_id=row["session_id"],
            route_decision=RouteDecision(row["route"]),
            sensitivity_level=SensitivityLevel(row["sensitivity"]),
            pii_detected=bool(row["pii_detected"]),
            pii_count=row["pii_count"],
            provider_called=row["provider"],
            gdpr_compliant=bool(row["gdpr_ok"]),
            latency_ms=row["latency_ms"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
