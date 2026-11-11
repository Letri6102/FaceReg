"""Lưu trữ SQLite: thí sinh, vector đặc trưng (template), phiên check-in, kết quả và sự kiện giám sát."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    code          TEXT NOT NULL UNIQUE,          -- số báo danh / mã thí sinh
    full_name     TEXT NOT NULL DEFAULT '',
    photo_path    TEXT NOT NULL,                 -- ảnh 4x6 gốc
    face_path     TEXT NOT NULL,                 -- ảnh khuôn mặt đã cắt
    quality_json  TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS templates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id  INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    set_id        INTEGER NOT NULL DEFAULT 0,    -- một "bộ mẫu" = các vector của nhiều mô hình từ cùng một ảnh
    source        TEXT NOT NULL,                 -- 'enroll' (ảnh 4x6) | 'live' (webcam đã xác thực)
    model         TEXT NOT NULL DEFAULT '',      -- tên mô hình nhận diện
    embedding     BLOB NOT NULL,                 -- float32, đã chuẩn hóa L2
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_candidate ON templates(candidate_id);

CREATE TABLE IF NOT EXISTS checkin_sessions (
    id            TEXT PRIMARY KEY,
    candidate_id  INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    exam_code     TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL,                 -- 'active' | 'finished' | 'expired'
    state_json    TEXT NOT NULL,
    client_ip     TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    expires_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS checkins (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL UNIQUE,
    candidate_id   INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    exam_code      TEXT NOT NULL DEFAULT '',
    auto_status    TEXT NOT NULL,                -- quyết định của hệ thống: ACCEPT | REVIEW | REJECT
    final_status   TEXT NOT NULL,                -- sau khi giám thị duyệt: ACCEPT | REVIEW | REJECT
    score          REAL NOT NULL,
    spoof_score    REAL,
    scores_json    TEXT NOT NULL DEFAULT '{}',   -- điểm từng mô hình
    reasons_json   TEXT NOT NULL DEFAULT '[]',
    captured_path  TEXT NOT NULL DEFAULT '',
    monitor_token  TEXT,
    reviewer_note  TEXT NOT NULL DEFAULT '',
    reviewed_at    REAL,
    client_ip      TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkins_status ON checkins(final_status);
CREATE INDEX IF NOT EXISTS idx_checkins_candidate ON checkins(candidate_id);

CREATE TABLE IF NOT EXISTS monitor_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    checkin_id   INTEGER NOT NULL REFERENCES checkins(id) ON DELETE CASCADE,
    status       TEXT NOT NULL,                  -- OK | NO_FACE | MULTI_FACE | MISMATCH
    score        REAL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_monitor_checkin ON monitor_events(checkin_id);
"""


def emb_to_blob(emb: np.ndarray) -> bytes:
    return np.asarray(emb, dtype=np.float32).tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Nâng cấp CSDL tạo bởi phiên bản cũ (chưa có cột model/set_id/scores_json)."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(templates)")}
        if "model" not in cols:
            conn.execute("ALTER TABLE templates ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        if "set_id" not in cols:
            conn.execute("ALTER TABLE templates ADD COLUMN set_id INTEGER NOT NULL DEFAULT 0")
            conn.execute("UPDATE templates SET set_id = id")
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(checkins)")}
        if "scores_json" not in cols:
            conn.execute("ALTER TABLE checkins ADD COLUMN scores_json TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------- Thí sinh ----------------
    def get_candidate(self, code: str) -> dict | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM candidates WHERE code = ?", (code,)).fetchone()
        return dict(row) if row else None

    def get_candidate_by_id(self, cid: int) -> dict | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None

    def upsert_candidate(self, code: str, full_name: str, photo_path: str, face_path: str,
                         quality: dict, embeddings: dict[str, np.ndarray]) -> int:
        """Tạo mới hoặc thay thế thí sinh. Khi thay ảnh, xóa toàn bộ template cũ (kể cả 'live')."""
        now = time.time()
        with self.connect() as c:
            row = c.execute("SELECT id FROM candidates WHERE code = ?", (code,)).fetchone()
            if row:
                cid = row["id"]
                c.execute(
                    "UPDATE candidates SET full_name=?, photo_path=?, face_path=?, quality_json=?, updated_at=? WHERE id=?",
                    (full_name, photo_path, face_path, json.dumps(quality), now, cid),
                )
                c.execute("DELETE FROM templates WHERE candidate_id = ?", (cid,))
            else:
                cur = c.execute(
                    "INSERT INTO candidates (code, full_name, photo_path, face_path, quality_json, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (code, full_name, photo_path, face_path, json.dumps(quality), now, now),
                )
                cid = cur.lastrowid
            self._insert_set(c, int(cid), self._next_set_id(c, int(cid)), "enroll", embeddings, now)
        return int(cid)

    def list_candidates(self, q: str = "", limit: int = 200, offset: int = 0) -> list[dict]:
        sql = (
            "SELECT c.id, c.code, c.full_name, c.created_at, c.updated_at,"
            " (SELECT COUNT(DISTINCT t.set_id) FROM templates t WHERE t.candidate_id=c.id AND t.source='live') AS live_templates,"
            " (SELECT COUNT(*) FROM checkins k WHERE k.candidate_id=c.id) AS checkins"
            " FROM candidates c"
        )
        args: list[Any] = []
        if q:
            sql += " WHERE c.code LIKE ? OR c.full_name LIKE ?"
            args += [f"%{q}%", f"%{q}%"]
        sql += " ORDER BY c.updated_at DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def delete_candidate(self, code: str) -> dict | None:
        cand = self.get_candidate(code)
        if not cand:
            return None
        with self.connect() as c:
            c.execute("DELETE FROM candidates WHERE id = ?", (cand["id"],))
        return cand

    # ---------------- Template ----------------
    @staticmethod
    def _next_set_id(c: sqlite3.Connection, candidate_id: int) -> int:
        row = c.execute("SELECT COALESCE(MAX(set_id), 0) AS m FROM templates WHERE candidate_id = ?",
                        (candidate_id,)).fetchone()
        return int(row["m"]) + 1

    @staticmethod
    def _insert_set(c: sqlite3.Connection, candidate_id: int, set_id: int, source: str,
                    embeddings: dict[str, np.ndarray], now: float) -> None:
        for model, emb in embeddings.items():
            c.execute(
                "INSERT INTO templates (candidate_id, set_id, source, model, embedding, created_at) VALUES (?,?,?,?,?,?)",
                (candidate_id, set_id, source, model, emb_to_blob(emb), now),
            )

    def get_template_sets(self, candidate_id: int) -> list[dict]:
        """[{set_id, source, embs: {model: vector}}], bộ 'enroll' đứng đầu."""
        with self.connect() as c:
            rows = c.execute(
                "SELECT set_id, source, model, embedding FROM templates WHERE candidate_id = ? ORDER BY set_id, id",
                (candidate_id,),
            ).fetchall()
        sets: dict[int, dict] = {}
        for r in rows:
            st = sets.setdefault(r["set_id"], {"set_id": r["set_id"], "source": r["source"], "embs": {}})
            if r["model"]:
                st["embs"][r["model"]] = blob_to_emb(r["embedding"])
        return sorted(sets.values(), key=lambda x: (x["source"] != "enroll", x["set_id"]))

    def add_embeddings_to_set(self, candidate_id: int, set_id: int, source: str,
                              embeddings: dict[str, np.ndarray]) -> None:
        """Bổ sung vector của mô hình mới cho một bộ mẫu sẵn có (ghi đè nếu đã có)."""
        with self.connect() as c:
            for model in embeddings:
                c.execute("DELETE FROM templates WHERE candidate_id=? AND set_id=? AND model=?",
                          (candidate_id, set_id, model))
            self._insert_set(c, candidate_id, set_id, source, embeddings, time.time())

    def add_live_template(self, candidate_id: int, embeddings: dict[str, np.ndarray], max_live: int) -> None:
        """Thêm bộ mẫu từ webcam; giữ tối đa `max_live` bộ mới nhất (FIFO)."""
        now = time.time()
        with self.connect() as c:
            self._insert_set(c, candidate_id, self._next_set_id(c, candidate_id), "live", embeddings, now)
            rows = c.execute(
                "SELECT DISTINCT set_id FROM templates WHERE candidate_id=? AND source='live' ORDER BY set_id DESC",
                (candidate_id,),
            ).fetchall()
            for r in rows[max_live:]:
                c.execute("DELETE FROM templates WHERE candidate_id=? AND set_id=?", (candidate_id, r["set_id"]))

    def all_candidate_ids(self) -> list[int]:
        with self.connect() as c:
            return [r["id"] for r in c.execute("SELECT id FROM candidates ORDER BY id").fetchall()]

    # ---------------- Phiên check-in ----------------
    def create_session(self, sid: str, candidate_id: int, exam_code: str, state: dict,
                       ttl: int, client_ip: str) -> None:
        now = time.time()
        with self.connect() as c:
            c.execute(
                "INSERT INTO checkin_sessions (id, candidate_id, exam_code, status, state_json, client_ip, created_at, expires_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (sid, candidate_id, exam_code, "active", json.dumps(state), client_ip, now, now + ttl),
            )

    def get_session(self, sid: str) -> dict | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM checkin_sessions WHERE id = ?", (sid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["state"] = json.loads(d.pop("state_json"))
        return d

    def update_session(self, sid: str, state: dict, status: str | None = None) -> None:
        with self.connect() as c:
            if status:
                c.execute("UPDATE checkin_sessions SET state_json=?, status=? WHERE id=?",
                          (json.dumps(state), status, sid))
            else:
                c.execute("UPDATE checkin_sessions SET state_json=? WHERE id=?", (json.dumps(state), sid))

    def purge_old_sessions(self, older_than_seconds: int = 86400) -> int:
        cutoff = time.time() - older_than_seconds
        with self.connect() as c:
            cur = c.execute("DELETE FROM checkin_sessions WHERE expires_at < ?", (cutoff,))
            return cur.rowcount

    # ---------------- Kết quả check-in ----------------
    def create_checkin(self, **kw) -> int:
        kw.setdefault("created_at", time.time())
        kw["reasons_json"] = json.dumps(kw.pop("reasons", []), ensure_ascii=False)
        kw["scores_json"] = json.dumps(kw.pop("scores", {}))
        cols = ", ".join(kw.keys())
        qs = ", ".join("?" for _ in kw)
        with self.connect() as c:
            cur = c.execute(f"INSERT INTO checkins ({cols}) VALUES ({qs})", list(kw.values()))
            return int(cur.lastrowid)

    def _checkin_row(self, row) -> dict:
        d = dict(row)
        d["reasons"] = json.loads(d.pop("reasons_json") or "[]")
        d["scores"] = json.loads(d.pop("scores_json", None) or "{}")
        return d

    def get_checkin(self, checkin_id: int) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT k.*, c.code AS candidate_code, c.full_name FROM checkins k"
                " JOIN candidates c ON c.id = k.candidate_id WHERE k.id = ?", (checkin_id,)
            ).fetchone()
        return self._checkin_row(row) if row else None

    def get_checkin_by_session(self, sid: str) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT k.*, c.code AS candidate_code, c.full_name FROM checkins k"
                " JOIN candidates c ON c.id = k.candidate_id WHERE k.session_id = ?", (sid,)
            ).fetchone()
        return self._checkin_row(row) if row else None

    def get_checkin_by_token(self, token: str) -> dict | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM checkins WHERE monitor_token = ?", (token,)).fetchone()
        return self._checkin_row(row) if row else None

    def list_checkins(self, status: str = "", exam_code: str = "", candidate_code: str = "",
                      limit: int = 100) -> list[dict]:
        sql = (
            "SELECT k.*, c.code AS candidate_code, c.full_name,"
            " (SELECT COUNT(*) FROM monitor_events m WHERE m.checkin_id=k.id AND m.status!='OK') AS monitor_alerts"
            " FROM checkins k JOIN candidates c ON c.id = k.candidate_id WHERE 1=1"
        )
        args: list[Any] = []
        if status:
            sql += " AND k.final_status = ?"
            args.append(status)
        if exam_code:
            sql += " AND k.exam_code = ?"
            args.append(exam_code)
        if candidate_code:
            sql += " AND c.code = ?"
            args.append(candidate_code)
        sql += " ORDER BY k.id DESC LIMIT ?"
        args.append(limit)
        with self.connect() as c:
            return [self._checkin_row(r) for r in c.execute(sql, args).fetchall()]

    def review_checkin(self, checkin_id: int, final_status: str, note: str, monitor_token: str | None) -> None:
        with self.connect() as c:
            c.execute(
                "UPDATE checkins SET final_status=?, reviewer_note=?, reviewed_at=?,"
                " monitor_token=COALESCE(monitor_token, ?) WHERE id=?",
                (final_status, note, time.time(), monitor_token, checkin_id),
            )

    # ---------------- Giám sát ----------------
    def add_monitor_event(self, checkin_id: int, status: str, score: float | None) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT INTO monitor_events (checkin_id, status, score, created_at) VALUES (?,?,?,?)",
                (checkin_id, status, score, time.time()),
            )

    def last_monitor_event_time(self, checkin_id: int) -> float | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT MAX(created_at) AS t FROM monitor_events WHERE checkin_id = ?", (checkin_id,)
            ).fetchone()
        return row["t"] if row and row["t"] is not None else None

    def list_monitor_events(self, checkin_id: int, limit: int = 500) -> list[dict]:
        with self.connect() as c:
            rows = c.execute(
                "SELECT status, score, created_at FROM monitor_events WHERE checkin_id=? ORDER BY id DESC LIMIT ?",
                (checkin_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
