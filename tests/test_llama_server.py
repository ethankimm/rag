"""Unit tests for model-specific llama-server router configuration."""

from __future__ import annotations

import httpx

from backend.utils.llama_server import ManagedLlamaServer


def test_router_does_not_apply_rank_pooling_to_every_model(tmp_path, monkeypatch):
    """Embedding and reranking flags must come from per-model presets."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    preset = models_dir / "models.ini"
    preset.write_text("version = 1\n", encoding="utf-8")

    health_checks = 0

    class Response:
        status_code = 200

    def fake_get(*args, **kwargs):
        nonlocal health_checks
        del args, kwargs
        health_checks += 1
        if health_checks == 1:
            raise httpx.ConnectError("not running")
        return Response()

    launched_command = []

    class Process:
        @staticmethod
        def terminate():
            return None

        @staticmethod
        def wait(timeout):
            del timeout
            return None

    def fake_popen(command, **kwargs):
        del kwargs
        launched_command.extend(command)
        return Process()

    monkeypatch.setattr("backend.utils.llama_server.httpx.get", fake_get)
    monkeypatch.setattr("backend.utils.llama_server.subprocess.Popen", fake_popen)

    server = ManagedLlamaServer(models_dir=models_dir, models_preset=preset)
    server.start()

    assert "--embedding" not in launched_command
    assert "--reranking" not in launched_command
    assert launched_command[launched_command.index("--models-preset") + 1] == str(
        preset
    )
