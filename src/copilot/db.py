import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    # The default of 40 silently returns at most 40 rows, however high the LIMIT.
    conn.execute("SET hnsw.ef_search = 200")
    return conn


def migrate(conn: psycopg.Connection) -> None:
    conn.execute((Path(__file__).parent / "schema.sql").read_text())
