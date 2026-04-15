from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from trainer_service.paths import SAVED_MODELS_DIR


SAVED_MODELS_TABLE = "saved_models"


def _get_saved_models_storage() -> str:
    configured = os.getenv("SAVED_MODELS_STORAGE", "").strip().lower()
    if configured in {"file", "postgres"}:
        return configured
    return "postgres" if _get_database_url() else "file"


def _get_database_url() -> str | None:
    return os.getenv("SAVED_MODELS_DATABASE_URL") or os.getenv("DATABASE_URL")


def _get_database_ssl_mode() -> str | None:
    ssl_mode = os.getenv("SAVED_MODELS_DATABASE_SSL", "").strip().lower()
    if ssl_mode:
        return ssl_mode
    shared_ssl_mode = os.getenv("AUDIT_DATABASE_SSL", "").strip().lower()
    if shared_ssl_mode == "require":
        return "require"
    return None


def _normalize_saved_model_name(name: str) -> str:
    safe_name = Path(name).name.strip()
    if not safe_name:
        raise ValueError("Missing model name.")
    if not safe_name.endswith(".json"):
        safe_name = f"{safe_name}.json"
    return safe_name


def _strip_json_extension(name: str) -> str:
    return name[:-5] if name.endswith(".json") else name


def _load_psycopg():
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "SAVED_MODELS_STORAGE=postgres requires psycopg. Install trainer-service requirements first."
        ) from exc
    return psycopg, Jsonb


def _connect():
    database_url = _get_database_url()
    if not database_url:
        raise RuntimeError("SAVED_MODELS_STORAGE=postgres requires SAVED_MODELS_DATABASE_URL or DATABASE_URL.")
    psycopg, _ = _load_psycopg()
    connect_kwargs: dict[str, Any] = {}
    ssl_mode = _get_database_ssl_mode()
    if ssl_mode:
        connect_kwargs["sslmode"] = ssl_mode
    return psycopg.connect(database_url, **connect_kwargs)


def _ensure_postgres_schema() -> None:
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SAVED_MODELS_TABLE} (
                    model_name TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {SAVED_MODELS_TABLE}_updated_at_idx
                ON {SAVED_MODELS_TABLE} (updated_at DESC)
                """
            )
        connection.commit()


def _list_saved_models_from_files() -> list[str]:
    if not SAVED_MODELS_DIR.exists():
        return []
    return sorted([path.stem for path in SAVED_MODELS_DIR.glob("*.json") if path.is_file()])


def _get_saved_model_from_files(name: str) -> dict[str, Any] | None:
    safe_name = _normalize_saved_model_name(name)
    path = SAVED_MODELS_DIR / safe_name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_saved_model_to_files(name: str, payload: dict[str, Any]) -> str:
    safe_name = _normalize_saved_model_name(name)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVED_MODELS_DIR / safe_name
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return safe_name


def _list_saved_models_from_postgres() -> list[str]:
    _ensure_postgres_schema()
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT model_name FROM {SAVED_MODELS_TABLE} ORDER BY model_name ASC")
            return [_strip_json_extension(str(row[0])) for row in cursor.fetchall()]


def _get_saved_model_from_postgres(name: str) -> dict[str, Any] | None:
    safe_name = _normalize_saved_model_name(name)
    _ensure_postgres_schema()
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {SAVED_MODELS_TABLE} WHERE model_name = %s",
                (safe_name,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            payload = row[0]
            if isinstance(payload, str):
                return json.loads(payload)
            return payload


def _save_saved_model_to_postgres(name: str, payload: dict[str, Any]) -> str:
    safe_name = _normalize_saved_model_name(name)
    _ensure_postgres_schema()
    _, jsonb = _load_psycopg()
    with _connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {SAVED_MODELS_TABLE} (model_name, payload)
                VALUES (%s, %s)
                ON CONFLICT (model_name)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (safe_name, jsonb(payload)),
            )
        connection.commit()
    return safe_name


def list_saved_model_names() -> list[str]:
    if _get_saved_models_storage() == "postgres":
        return _list_saved_models_from_postgres()
    return _list_saved_models_from_files()


def get_saved_model_payload(name: str) -> dict[str, Any] | None:
    if _get_saved_models_storage() == "postgres":
        return _get_saved_model_from_postgres(name)
    return _get_saved_model_from_files(name)


def save_saved_model_payload(name: str, payload: dict[str, Any]) -> str:
    if _get_saved_models_storage() == "postgres":
        return _save_saved_model_to_postgres(name, payload)
    return _save_saved_model_to_files(name, payload)
