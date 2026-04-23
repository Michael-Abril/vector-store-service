import asyncio
import os
import time
import uuid
from typing import Optional
import requests
import psycopg2
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://varity:varity@localhost:5432/app")
MODEL = "tinyllama"
EMBED_DIM = 2048

_state = {"ready": False, "error": None}


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def _wait_for_postgres(max_wait=120):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            conn = get_conn()
            conn.close()
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("Postgres not ready after 120s")


def _wait_for_ollama(max_wait=300):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("Ollama not ready after 300s")


def _pull_model():
    resp = requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"name": MODEL, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()


def _init_db():
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding vector({EMBED_DIM})
                    )
                """)
    finally:
        conn.close()


def _background_init():
    try:
        _wait_for_postgres()
        _wait_for_ollama()
        _pull_model()
        _init_db()
        _state["ready"] = True
    except Exception as exc:
        _state["error"] = str(exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Start init in a thread so the HTTP server comes up immediately.
    # Varity health check can then pass before sidecars are fully ready.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _background_init)
    yield


app = FastAPI(lifespan=lifespan)


def _require_ready():
    if not _state["ready"]:
        detail = _state["error"] or "still initializing"
        raise HTTPException(status_code=503, detail=detail)


class IngestRequest(BaseModel):
    text: str
    id: Optional[str] = None


@app.get("/health")
def health():
    if _state["error"]:
        return {"status": "error", "detail": _state["error"]}
    if not _state["ready"]:
        return {"status": "initializing"}
    return {"status": "ready"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    _require_ready()
    doc_id = req.id or str(uuid.uuid4())
    embedding: list[float] = []
    for attempt in range(3):
        try:
            embedding = get_embedding(req.text)
            break
        except Exception:
            if attempt == 2:
                raise HTTPException(status_code=503, detail="embedding failed")
            _pull_model()
            time.sleep(5)
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (id, text, embedding) VALUES (%s, %s, %s::vector) "
                    "ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, embedding=EXCLUDED.embedding",
                    (doc_id, req.text, str(embedding)),
                )
    finally:
        conn.close()
    return {"id": doc_id, "status": "stored"}


@app.get("/search")
def search(q: str, k: int = 5):
    _require_ready()
    embedding: list[float] = []
    for attempt in range(3):
        try:
            embedding = get_embedding(q)
            break
        except Exception:
            if attempt == 2:
                raise HTTPException(status_code=503, detail="embedding failed")
            _pull_model()
            time.sleep(5)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, 1 - (embedding <=> %s::vector) AS score "
                "FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
                (str(embedding), str(embedding), k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]


@app.get("/documents")
def list_documents():
    _require_ready()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, left(text, 100) FROM documents ORDER BY id")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [{"id": r[0], "preview": r[1]} for r in rows]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    _require_ready()
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="document not found")
    finally:
        conn.close()
    return {"id": doc_id, "status": "deleted"}
