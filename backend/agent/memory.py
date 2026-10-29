import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import DATA_DIR, GOOGLE_API_KEY, LLM_MODELS_FALLBACK

DB_PATH = os.path.join(DATA_DIR, "chat_history.db")

class ConversationMemory:
    """Multi-turn conversation memory with sliding window & running summary buffer."""

    def __init__(self, window_size: int = 4, db_path: Optional[str] = None):
        self.window_size = window_size
        self.db_path = db_path or DB_PATH
        self._init_db()
        self.llm = None
        for model in LLM_MODELS_FALLBACK:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_output_tokens=300
                )
                break
            except Exception:
                pass

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                summary TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.commit()
        conn.close()

    def get_or_create_session(self, session_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, title, summary FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            sess_title = title or f"Research Session {datetime.datetime.now().strftime('%b %d, %H:%M')}"
            cursor.execute("INSERT INTO sessions (session_id, title, summary) VALUES (?, ?, ?)", (session_id, sess_title, ""))
            conn.commit()
            row = (session_id, sess_title, "")
        conn.close()
        return {"session_id": row[0], "title": row[1], "summary": row[2]}

    def list_sessions(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.session_id, s.title, s.created_at, COUNT(m.id) as msg_count
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        return [
            {"session_id": r[0], "title": r[1], "created_at": r[2], "message_count": r[3]}
            for r in rows
        ]

    def add_message(self, session_id: str, role: str, content: str):
        self.get_or_create_session(session_id)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
        conn.commit()

        # Check if we should update running summary for older messages
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        count = cursor.fetchone()[0]
        conn.close()

        if count > (self.window_size * 2) + 2:
            self._update_session_summary(session_id)

    def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Fetch sliding window of recent turns and older summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT summary FROM sessions WHERE session_id = ?", (session_id,))
        s_row = cursor.fetchone()
        summary = s_row[0] if s_row else ""

        cursor.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """, (session_id, self.window_size * 2))
        recent_rows = cursor.fetchall()
        conn.close()

        recent_messages = [{"role": r[0], "content": r[1]} for r in reversed(recent_rows)]
        return {
            "summary": summary,
            "recent_messages": recent_messages
        }

    def _update_session_summary(self, session_id: str):
        """Compress older messages into a rolling session summary."""
        if not self.llm:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
        all_rows = cursor.fetchall()
        conn.close()

        if len(all_rows) <= self.window_size * 2:
            return

        older_rows = all_rows[:-(self.window_size * 2)]
        dialogue_text = "\n".join([f"{r[0].upper()}: {r[1]}" for r in older_rows])

        prompt = f"""
Summarize the key facts, user goals, and established answers from this past conversation history concisely (2-3 sentences max):

{dialogue_text}

Summary:
"""
        try:
            res = self.llm.invoke(prompt)
            new_summary = res.content if hasattr(res, 'content') else str(res)
            if isinstance(new_summary, list):
                new_summary = " ".join([s.get("text", str(s)) for s in new_summary])

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE sessions SET summary = ? WHERE session_id = ?", (str(new_summary).strip(), session_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating memory summary: {e}")

    def clear_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
