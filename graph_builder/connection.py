"""
Neo4j driver connection management.

Provides a context-manager class that reads credentials from ``.env``,
creates a driver, verifies connectivity, and ensures cleanup.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Managed Neo4j driver connection.

    Usage
    -----
    >>> with Neo4jConnection() as conn:
    ...     with conn.session() as session:
    ...         session.run("RETURN 1")

    Or without context manager (call ``close()`` manually):

    >>> conn = Neo4jConnection()
    >>> conn.connect()
    >>> # ... use conn.session() ...
    >>> conn.close()
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self._uri      = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self._user     = user     or os.getenv("NEO4J_USER",     "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._database = database or os.getenv("NEO4J_DATABASE")
        self._driver   = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Create the driver and verify connectivity."""
        if self._driver is not None:
            return

        logger.info(f"Connecting to Neo4j at {self._uri} …")

        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )

        # Verify the connection works
        self._driver.verify_connectivity()
        logger.info("Neo4j connection verified ✓")

    def close(self) -> None:
        """Close the driver and release resources."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "Neo4jConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ── Session factory ───────────────────────────────────────────────────

    def session(self, **kwargs):
        """Return a new Neo4j session.

        Parameters
        ----------
        **kwargs:
            Passed through to ``driver.session()``.

        Returns
        -------
        neo4j.Session
        """
        if self._driver is None:
            raise RuntimeError(
                "Not connected. Call connect() or use as a context manager."
            )
        return self._driver.session(
            database=self._database,
            **kwargs,
        )

    @property
    def driver(self):
        """Access the underlying Neo4j driver (or None if not connected)."""
        return self._driver
