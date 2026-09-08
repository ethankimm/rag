"""Command-order tests for local Make entry points."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("target", "application_command"),
    [
        ("dev", "uvicorn backend.api:app"),
        ("ingest", "python -m backend.ingestion"),
        ("eval", "python -m backend.evaluation"),
        ("rag-smoke", "python -m backend.rag"),
    ],
)
def test_application_target_starts_model_server_before_application(
    target: str,
    application_command: str,
) -> None:
    result = subprocess.run(
        ["make", "--dry-run", target],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    model_server_position = result.stdout.index(
        "python -m backend.utils.model_server_launcher up"
    )
    application_position = result.stdout.index(application_command)

    assert model_server_position < application_position


def test_runtime_gate_starts_persistent_server_before_real_model_test() -> None:
    result = subprocess.run(
        ["make", "--dry-run", "test-runtime"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    model_server_position = result.stdout.index(
        "python -m backend.utils.model_server_launcher up"
    )
    runtime_test_position = result.stdout.index(
        "pytest -m model_integration tests/test_model_server_integration.py"
    )

    assert model_server_position < runtime_test_position
    assert "RUN_MODEL_INTEGRATION=1" in result.stdout
    assert "model_server_launcher down" not in result.stdout
