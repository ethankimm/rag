"""Managed wrapper for launching and stopping a local llama-server process."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import TracebackType

import httpx


class ManagedLlamaServer:
    """Spawns and manages a local llama-server process in router mode."""

    def __init__(
        self,
        models_dir: Path | str,
        models_preset: Path | str | None = None,
        port: int = 8080,
        host: str = "127.0.0.1",
        gpu_layers: int = -1,
        models_max: int = 4,
    ):
        path = Path(models_dir).resolve()
        self.models_dir = path.parent if path.is_file() else path
        if models_preset is not None:
            self.models_preset = Path(models_preset).resolve()
        else:
            default_preset = self.models_dir / "llama-models.ini"
            self.models_preset = default_preset if default_preset.is_file() else None
        self.port = port
        self.host = host
        self.gpu_layers = gpu_layers
        self.models_max = models_max
        self.process: subprocess.Popen | None = None
        self.base_url = f"http://{host}:{port}"

    def __enter__(self) -> ManagedLlamaServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self, mode: str | None = None) -> None:
        """Start the server in router mode if it isn't already running.

        Args:
            mode: Optional mode name for backwards compatibility.
        """
        try:
            res = httpx.get(f"{self.base_url}/health", timeout=1.0)
            if res.status_code == 200:
                print(
                    f"llama-server running on port {self.port}. "
                    "Reusing existing instance."
                )
                return
        except httpx.HTTPError:
            pass  # No server running, safe to spawn

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found at: {self.models_dir}")
        if self.models_preset is not None and not self.models_preset.is_file():
            raise FileNotFoundError(f"Models preset not found at: {self.models_preset}")

        print(
            f"Launching llama-server router on port {self.port} "
            f"(models: {self.models_dir.name}, models_max: {self.models_max})..."
        )

        # Embedding and rank pooling are mutually incompatible. Their flags
        # must be model-specific in the router preset, never global here.
        cmd = [
            "llama-server",
            "--models-dir",
            str(self.models_dir),
            "--models-max",
            str(self.models_max),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            str(self.gpu_layers),
            "-c",
            "8192",
            "-b",
            "8192",
            "-ub",
            "8192",
        ]
        if self.models_preset is not None:
            cmd.extend(["--models-preset", str(self.models_preset)])

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print("Waiting for llama-server router to start...")
        start_time = time.time()
        while time.time() - start_time < 30.0:
            try:
                res = httpx.get(f"{self.base_url}/health", timeout=1.0)
                if res.status_code == 200:
                    print("llama-server router is ready!")
                    return
            except httpx.HTTPError:
                time.sleep(0.5)

        self.stop()
        raise RuntimeError(
            f"llama-server failed to start within 30 seconds on port {self.port}."
        )

    def stop(self) -> None:
        """Cleanly terminate the background process."""
        if self.process:
            print("Shutting down managed llama-server process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
