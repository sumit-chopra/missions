"""SQLite-backed store for the last N conversation turns.

A *turn* is one exchange: the user's message and the assistant's reply. The
store keeps only the most recent ``max_turns`` turns; older rows are pruned on
insert.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

ENV_DB_PATH = "GLASS_COCKPIT_HISTORY_DB"
DEFAULT_DB_PATH = Path(".missions/glass_cockpit.db")
MAX_TURNS = 10


@dataclass(frozen=True)
class Turn:
    """One user/assistant exchange."""

    user_message: str
    assistant_message: str


class ConversationStore:
    """A SQLite-backed store of the most recent conversation turns.

    ``path`` defaults to the ``GLASS_COCKPIT_HISTORY_DB`` environment variable,
    then to :data:`DEFAULT_DB_PATH`.
    Parent directories are created as needed.
    """

    def __init__(self) -> None:
        path = os.environ.get(ENV_DB_PATH) or str(DEFAULT_DB_PATH)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """Record one exchange, then prune all but the most recent ``max_turns``."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO turns (user_message, assistant_message) VALUES (?, ?)",
                (user_message, assistant_message),
            )
            self._conn.execute(
                "DELETE FROM turns WHERE id NOT IN (SELECT id FROM turns ORDER BY id DESC LIMIT ?)",
                (MAX_TURNS,),
            )

    def recent_turns(self) -> list[Turn]:
        """Return the ``max_turns`` most recent turns, oldest first."""
        rows = self._conn.execute(
            "SELECT user_message, assistant_message FROM turns ORDER BY id DESC LIMIT ?",
            (MAX_TURNS,),
        ).fetchall()
        return [Turn(user, assistant) for user, assistant in reversed(rows)]

    def close(self) -> None:
        self._conn.close()
