"""Persistent, repository-owned launcher for the local Model Server."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx

from backend.retrieval.model_client import model_ids_from_environment


@dataclass(frozen=True)
class ProcessIdentity:
    """Identity needed to distinguish an owned process from PID reuse."""

    pid: int
    executable: str
    started_at: str


@dataclass(frozen=True)
class PortOwner:
    """Port-owner details, including partial operating-system discovery."""

    pid: int
    executable: str | None
    identity: ProcessIdentity | None


@dataclass(frozen=True)
class LauncherConfig:
    """Paths and timing for one repository's Model Server."""

    base_url: str
    host: str
    port: int
    models_dir: Path
    models_preset: Path
    required_model_files: tuple[Path, ...]
    state_path: Path
    log_path: Path
    readiness_timeout: float = 30.0
    stop_timeout: float = 5.0
    poll_interval: float = 0.5

    @classmethod
    def from_environment(cls, repo_root: Path | None = None) -> LauncherConfig:
        root = repo_root or Path(__file__).resolve().parents[2]
        base_url = os.environ.get("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:8080")
        url = httpx.URL(base_url)
        if url.scheme != "http" or not url.host or url.port is None:
            raise ValueError(
                "LOCAL_MODEL_BASE_URL must be an absolute HTTP URL with a port"
            )
        if url.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "LOCAL_MODEL_BASE_URL must use a loopback host for local startup"
            )
        models_dir = root / "backend" / "models"
        runtime_dir = root / ".runtime"
        model_ids = model_ids_from_environment()
        return cls(
            base_url=base_url.rstrip("/"),
            host=url.host,
            port=url.port,
            models_dir=models_dir,
            models_preset=models_dir / "llama-models.ini",
            required_model_files=tuple(
                models_dir / f"{model_id}.gguf" for model_id in model_ids
            ),
            state_path=runtime_dir / "model-server.json",
            log_path=runtime_dir / "model-server.log",
        )


class System(Protocol):
    """Operating-system boundary used by command-level tests."""

    def health_is_ready(self, url: str, timeout: float) -> bool: ...

    def port_is_open(self, host: str, port: int, timeout: float) -> bool: ...

    def port_owner(self, port: int, timeout: float) -> PortOwner | None: ...

    def process_info(self, pid: int, timeout: float) -> ProcessIdentity | None: ...

    def spawn(self, command: list[str], log_path: Path) -> ProcessIdentity: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def signal(self, pid: int, action: signal.Signals) -> None: ...


class LocalSystem:
    """Real HTTP, process, socket, and clock operations."""

    def health_is_ready(self, url: str, timeout: float) -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=timeout).status_code == 200
        except httpx.HTTPError:
            return False

    def port_is_open(self, host: str, port: int, timeout: float) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def port_owner(self, port: int, timeout: float) -> PortOwner | None:
        try:
            result = subprocess.run(
                [
                    "lsof",
                    "-nP",
                    "-t",
                    f"-iTCP:{port}",
                    "-sTCP:LISTEN",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        first_pid = next(iter(result.stdout.splitlines()), "")
        try:
            pid = int(first_pid)
        except ValueError:
            return None
        identity = self.process_info(pid, timeout=timeout)
        return PortOwner(
            pid=pid,
            executable=identity.executable if identity else None,
            identity=identity,
        )

    def process_info(self, pid: int, timeout: float) -> ProcessIdentity | None:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart=", "-o", "comm="],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        fields = result.stdout.strip().split(maxsplit=5)
        if result.returncode != 0 or len(fields) != 6:
            return None
        return ProcessIdentity(
            pid=pid,
            executable=fields[5],
            started_at=" ".join(fields[:5]),
        )

    def spawn(self, command: list[str], log_path: Path) -> ProcessIdentity:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(10):
            if identity := self.process_info(process.pid, timeout=1.0):
                return identity
            time.sleep(0.01)
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "Could not stop unverifiable Model Server process"
                ) from error
        raise RuntimeError("Could not verify started Model Server process identity")

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def signal(self, pid: int, action: signal.Signals) -> None:
        os.kill(pid, action)


class ModelServerLauncher:
    """Start and stop only the Model Server recorded by this repository."""

    def __init__(self, config: LauncherConfig, *, system: System | None = None):
        self.config = config
        self.system = system or LocalSystem()

    def up(self) -> None:
        with self._command_lock():
            self._up_locked()

    def down(self) -> None:
        with self._command_lock():
            self._down_locked()

    def _up_locked(self) -> None:
        recorded = self._read_state()
        if recorded:
            current = self.system.process_info(recorded.pid, timeout=1.0)
            if current == recorded:
                self._validate_paths()
                print(f"Model Server: reusing owned process {recorded.pid}")
                self._announce_wait()
                self._wait_until_ready(recorded, started_here=False)
                return
            self._remove_state()
        else:
            self._remove_state()

        if self.system.port_is_open(self.config.host, self.config.port, timeout=0.25):
            self._raise_untracked_port_error()

        self._validate_paths()
        self._rotate_log()
        print(f"Model Server: starting on port {self.config.port}")
        process = self.system.spawn(self._command(), self.config.log_path)
        try:
            self._write_state(process)
        except OSError as error:
            self._stop_process(process)
            raise RuntimeError(
                "Could not record process ownership; started Model Server was stopped"
            ) from error
        self._announce_wait()
        self._wait_until_ready(process, started_here=True)

    def _down_locked(self) -> None:
        recorded = self._read_state()
        if not recorded:
            self._remove_state()
            print("Model Server: owned process is not running")
            return
        current = self.system.process_info(recorded.pid, timeout=1.0)
        if current != recorded:
            self._remove_state()
            print("Model Server: owned process is not running; removed stale state")
            return

        print(f"Model Server: stopping owned process {recorded.pid}")
        self._stop_process(recorded)
        print("Model Server: stopped")

    def _wait_until_ready(
        self, process: ProcessIdentity, *, started_here: bool
    ) -> None:
        def is_ready(remaining: float) -> bool:
            operation_timeout = min(0.25, remaining / 10)
            if (
                self.system.process_info(process.pid, timeout=operation_timeout)
                != process
            ):
                self._remove_state()
                message = (
                    f"Model Server process {process.pid} exited before becoming ready; "
                    f"see {self.config.log_path}"
                )
                print(message, file=sys.stderr)
                raise RuntimeError(message)
            self._require_owned_port_if_open(process, timeout=operation_timeout)
            if not self.system.health_is_ready(
                self.config.base_url, timeout=operation_timeout
            ):
                return False
            return self._require_owned_port_if_open(process, timeout=operation_timeout)

        try:
            ready = self._poll_until(is_ready, self.config.readiness_timeout)
        except Exception:
            if started_here:
                self._stop_process(process)
            raise
        if ready:
            print(f"Model Server: ready (PID {process.pid})")
            return

        message = (
            f"Model Server was not ready within "
            f"{self.config.readiness_timeout:g} seconds; see {self.config.log_path}"
        )
        if started_here:
            self._stop_process(process)
        print(message, file=sys.stderr)
        raise RuntimeError(message)

    def _stop_process(self, process: ProcessIdentity) -> None:
        if self.system.process_info(process.pid, timeout=1.0) != process:
            self._remove_state()
            return
        try:
            self.system.signal(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            self._remove_state()
            return
        if not self._poll_until(
            lambda remaining: (
                self.system.process_info(process.pid, timeout=min(1.0, remaining))
                != process
            ),
            self.config.stop_timeout,
        ):
            if self.system.process_info(process.pid, timeout=1.0) != process:
                self._remove_state()
                return
            self.system.signal(process.pid, signal.SIGKILL)
            if not self._poll_until(
                lambda remaining: (
                    self.system.process_info(process.pid, timeout=min(1.0, remaining))
                    != process
                ),
                self.config.stop_timeout,
            ):
                raise RuntimeError(
                    f"Owned Model Server process {process.pid} did not stop"
                )
        self._remove_state()

    def _poll_until(self, check: Callable[[float], bool], timeout: float) -> bool:
        deadline = self.system.monotonic() + timeout
        while (remaining := deadline - self.system.monotonic()) > 0:
            if check(remaining) and self.system.monotonic() <= deadline:
                return True
            remaining = deadline - self.system.monotonic()
            if remaining > 0:
                self.system.sleep(min(self.config.poll_interval, remaining))
        return False

    def _require_owned_port_if_open(
        self, process: ProcessIdentity, *, timeout: float
    ) -> bool:
        if not self.system.port_is_open(
            self.config.host, self.config.port, timeout=timeout
        ):
            return False
        owner = self.system.port_owner(self.config.port, timeout=timeout)
        if owner is None or owner.identity != process:
            self._raise_untracked_port_error(owner, timeout=timeout)
        return True

    def _raise_untracked_port_error(
        self, owner: PortOwner | None = None, *, timeout: float = 1.0
    ) -> None:
        health = (
            "healthy"
            if self.system.health_is_ready(self.config.base_url, timeout=timeout)
            else "unhealthy"
        )
        if owner is None:
            owner = self.system.port_owner(self.config.port, timeout=timeout)
        if owner and owner.executable:
            owner_details = f"PID {owner.pid}, executable {owner.executable}"
        elif owner:
            owner_details = f"PID {owner.pid}, executable unavailable"
        else:
            owner_details = "PID unavailable"
        message = (
            f"Port {self.config.port} is occupied by an untracked process "
            f"({owner_details}; health: {health}). Stop it manually or configure "
            "LOCAL_MODEL_BASE_URL. It was not terminated."
        )
        print(message, file=sys.stderr)
        raise RuntimeError(message)

    def _announce_wait(self) -> None:
        print(
            "Model Server: waiting up to "
            f"{self.config.readiness_timeout:g} seconds for health"
        )

    def _rotate_log(self) -> None:
        previous = self.config.log_path.with_name(
            f"{self.config.log_path.stem}.previous{self.config.log_path.suffix}"
        )
        previous.unlink(missing_ok=True)
        if self.config.log_path.exists():
            self.config.log_path.replace(previous)

    def _validate_paths(self) -> None:
        if not self.config.models_dir.is_dir():
            raise FileNotFoundError(
                f"Models directory not found at: {self.config.models_dir}"
            )
        if not self.config.models_preset.is_file():
            raise FileNotFoundError(
                f"Models preset not found at: {self.config.models_preset}"
            )
        missing_model_files = [
            path for path in self.config.required_model_files if not path.is_file()
        ]
        if missing_model_files:
            formatted_paths = ", ".join(str(path) for path in missing_model_files)
            raise FileNotFoundError(
                f"Required GGUF model files not found: {formatted_paths}"
            )

    def _command(self) -> list[str]:
        return [
            "llama-server",
            "--models-dir",
            str(self.config.models_dir),
            "--models-max",
            "4",
            "--models-preset",
            str(self.config.models_preset),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "-ngl",
            "-1",
            "-c",
            "8192",
            "-b",
            "8192",
            "-ub",
            "8192",
        ]

    def _read_state(self) -> ProcessIdentity | None:
        try:
            payload = json.loads(self.config.state_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("pid"), int)
                or isinstance(payload.get("pid"), bool)
                or not isinstance(payload.get("executable"), str)
                or not isinstance(payload.get("started_at"), str)
            ):
                return None
            return ProcessIdentity(
                pid=payload["pid"],
                executable=payload["executable"],
                started_at=payload["started_at"],
            )
        except (OSError, ValueError, TypeError):
            return None

    def _write_state(self, process: ProcessIdentity) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(process)), encoding="utf-8")
        temporary.replace(self.config.state_path)

    def _remove_state(self) -> None:
        self.config.state_path.unlink(missing_ok=True)

    @contextmanager
    def _command_lock(self) -> Iterator[None]:
        lock_path = self.config.state_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock:
            deadline = (
                time.monotonic()
                + self.config.readiness_timeout
                + self.config.stop_timeout
            )
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "Timed out waiting for another Model Server command"
                        ) from None
                    time.sleep(min(0.05, remaining))
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("up", "down"))
    args = parser.parse_args(argv)
    try:
        launcher = ModelServerLauncher(LauncherConfig.from_environment())
        getattr(launcher, args.command)()
    except (OSError, RuntimeError, ValueError) as error:
        if not str(error).startswith("Port "):
            print(f"Model Server: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
