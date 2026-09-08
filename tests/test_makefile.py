"""Command-order tests for local interactive Make entry points."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("target", "application_command"),
    [
        ("dev", "uvicorn backend.api:app"),
        ("rag-smoke", "python -m backend.rag"),
    ],
)
def test_interactive_target_starts_model_server_before_application(
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
