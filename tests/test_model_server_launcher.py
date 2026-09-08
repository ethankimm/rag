"""Command-level tests for persistent Model Server ownership."""

from __future__ import annotations

import json
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from backend.utils.model_server_launcher import (
    LauncherConfig,
    ModelServerLauncher,
    PortOwner,
    ProcessIdentity,
)


@dataclass
class FakeSystem:
    processes: dict[int, ProcessIdentity] = field(default_factory=dict)
    port_pid: int | None = None
    port_open_without_owner: bool = False
    healthy: bool = False
    next_pid: int = 100
    now: float = 0.0
    exit_on_term: bool = True
    health_delay: float = 0.0
    spawned_commands: list[list[str]] = field(default_factory=list)
    signals: list[tuple[int, signal.Signals]] = field(default_factory=list)

    def health_is_ready(self, url: str, timeout: float) -> bool:
        del url, timeout
        self.now += self.health_delay
        return self.healthy

    def port_is_open(self, host: str, port: int, timeout: float) -> bool:
        del host, port, timeout
        return self.port_pid is not None or self.port_open_without_owner

    def port_owner(self, port: int, timeout: float) -> PortOwner | None:
        del port, timeout
        if self.port_pid is None:
            return None
        identity = self.processes.get(self.port_pid)
        return PortOwner(
            pid=self.port_pid,
            executable=identity.executable if identity else None,
            identity=identity,
        )

    def process_info(self, pid: int, timeout: float) -> ProcessIdentity | None:
        del timeout
        return self.processes.get(pid)

    def spawn(self, command: list[str], log_path: Path) -> ProcessIdentity:
        self.spawned_commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch()
        process = ProcessIdentity(
            pid=self.next_pid,
            executable="llama-server",
            started_at=f"start-{self.next_pid}",
        )
        self.processes[process.pid] = process
        self.port_pid = process.pid
        return process

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def signal(self, pid: int, action: signal.Signals) -> None:
        self.signals.append((pid, action))
        if action == signal.SIGKILL or self.exit_on_term:
            self.processes.pop(pid, None)
            if self.port_pid == pid:
                self.port_pid = None


@dataclass
class BlockingHealthSystem(FakeSystem):
    health_entered: threading.Event = field(default_factory=threading.Event)
    release_health: threading.Event = field(default_factory=threading.Event)
    health_calls: int = 0

    def health_is_ready(self, url: str, timeout: float) -> bool:
        del url, timeout
        self.health_calls += 1
        if self.health_calls == 1:
            self.health_entered.set()
            assert self.release_health.wait(timeout=1.0)
        return True


@pytest.fixture
def config(tmp_path: Path) -> LauncherConfig:
    models_dir = tmp_path / "backend" / "models"
    models_dir.mkdir(parents=True)
    preset = models_dir / "llama-models.ini"
    preset.write_text("version = 1\n", encoding="utf-8")
    return LauncherConfig(
        base_url="http://127.0.0.1:8080",
        host="127.0.0.1",
        port=8080,
        models_dir=models_dir,
        models_preset=preset,
        state_path=tmp_path / ".runtime" / "model-server.json",
        log_path=tmp_path / ".runtime" / "model-server.log",
        readiness_timeout=1.0,
        stop_timeout=1.0,
        poll_interval=0.5,
    )


def write_state(config: LauncherConfig, process: ProcessIdentity) -> None:
    config.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.state_path.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "executable": process.executable,
                "started_at": process.started_at,
            }
        ),
        encoding="utf-8",
    )


def test_launcher_configuration_rejects_non_loopback_bind(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://0.0.0.0:8080")

    with pytest.raises(ValueError, match="loopback"):
        LauncherConfig.from_environment(tmp_path)


def test_up_starts_model_server_and_reports_progress(config, capsys):
    system = FakeSystem(healthy=True)

    ModelServerLauncher(config, system=system).up()

    output = capsys.readouterr().out
    assert "starting" in output.lower()
    assert "waiting" in output.lower()
    assert "ready" in output.lower()
    assert config.state_path.is_file()
    assert config.log_path.is_file()
    command = system.spawned_commands[0]
    assert command[:2] == ["llama-server", "--models-dir"]
    assert command[command.index("--models-preset") + 1] == str(config.models_preset)
    assert command[command.index("--models-max") + 1] == "4"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8080"
    assert command[command.index("-ngl") + 1] == "-1"
    assert command[command.index("-c") + 1] == "8192"
    assert command[command.index("-b") + 1] == "8192"
    assert command[command.index("-ub") + 1] == "8192"
    assert "--embedding" not in command
    assert "--reranking" not in command


def test_new_launch_preserves_previous_log(config):
    config.log_path.parent.mkdir(parents=True)
    config.log_path.write_text("failed startup details", encoding="utf-8")
    system = FakeSystem(healthy=True)

    ModelServerLauncher(config, system=system).up()

    previous_log = config.log_path.with_name("model-server.previous.log")
    assert previous_log.read_text(encoding="utf-8") == "failed startup details"


def test_up_reuses_live_owned_process(config, capsys):
    process = ProcessIdentity(41, "llama-server", "same-start")
    write_state(config, process)
    system = FakeSystem(
        processes={process.pid: process}, port_pid=process.pid, healthy=True
    )

    ModelServerLauncher(config, system=system).up()

    assert not system.spawned_commands
    assert "reusing owned process 41" in capsys.readouterr().out.lower()


def test_concurrent_up_commands_create_one_owned_process(config):
    system = BlockingHealthSystem()
    errors: list[BaseException] = []
    second_finished = threading.Event()

    def start(finished: threading.Event | None = None) -> None:
        try:
            ModelServerLauncher(config, system=system).up()
        except BaseException as error:
            errors.append(error)
        finally:
            if finished:
                finished.set()

    first = threading.Thread(target=start)
    second = threading.Thread(target=start, args=(second_finished,))
    first.start()
    assert system.health_entered.wait(timeout=1.0)
    second.start()
    assert not second_finished.wait(timeout=0.05)
    system.release_health.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert errors == []
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(system.spawned_commands) == 1


@pytest.mark.parametrize("healthy", [True, False])
def test_up_rejects_foreign_port_owner_while_recorded_process_is_live(
    config, capsys, healthy
):
    recorded = ProcessIdentity(41, "llama-server", "same-start")
    foreign = ProcessIdentity(77, "foreign-server", "foreign-start")
    write_state(config, recorded)
    system = FakeSystem(
        processes={41: recorded, 77: foreign}, port_pid=77, healthy=healthy
    )

    with pytest.raises(RuntimeError, match="untracked process"):
        ModelServerLauncher(config, system=system).up()

    assert system.signals == []
    error = capsys.readouterr().err
    assert "PID 77" in error
    assert ("healthy" if healthy else "unhealthy") in error


def test_up_does_not_accept_foreign_health_after_ownership_check(config, monkeypatch):
    recorded = ProcessIdentity(41, "llama-server", "same-start")
    foreign = ProcessIdentity(77, "foreign-server", "foreign-start")
    write_state(config, recorded)
    system = FakeSystem(
        processes={41: recorded, 77: foreign}, port_pid=41, healthy=True
    )

    def foreign_takes_port(url: str, timeout: float) -> bool:
        del url, timeout
        system.port_pid = 77
        return True

    monkeypatch.setattr(system, "health_is_ready", foreign_takes_port)

    with pytest.raises(RuntimeError, match="untracked process"):
        ModelServerLauncher(config, system=system).up()

    assert system.signals == []


def test_up_replaces_stale_state_without_signaling_old_pid(config):
    recorded = ProcessIdentity(41, "llama-server", "old-start")
    unrelated = ProcessIdentity(41, "python", "new-start")
    write_state(config, recorded)
    system = FakeSystem(processes={41: unrelated}, healthy=True)

    ModelServerLauncher(config, system=system).up()

    assert system.spawned_commands
    assert system.signals == []


@pytest.mark.parametrize("healthy", [True, False])
def test_up_rejects_untracked_port_owner(config, capsys, healthy):
    foreign = ProcessIdentity(77, "/usr/local/bin/foreign-server", "foreign-start")
    system = FakeSystem(processes={77: foreign}, port_pid=77, healthy=healthy)

    with pytest.raises(RuntimeError, match="untracked process"):
        ModelServerLauncher(config, system=system).up()

    error = capsys.readouterr().err
    assert "8080" in error
    assert "77" in error
    assert "/usr/local/bin/foreign-server" in error
    assert ("healthy" if healthy else "unhealthy") in error
    assert system.signals == []


def test_up_reports_unavailable_pid_for_occupied_port(config, capsys):
    system = FakeSystem(port_open_without_owner=True)

    with pytest.raises(RuntimeError, match="PID unavailable"):
        ModelServerLauncher(config, system=system).up()

    assert "PID unavailable" in capsys.readouterr().err
    assert system.signals == []


def test_up_preserves_known_pid_when_executable_discovery_fails(config, capsys):
    system = FakeSystem(port_pid=77)

    with pytest.raises(RuntimeError, match="PID 77"):
        ModelServerLauncher(config, system=system).up()

    error = capsys.readouterr().err
    assert "PID 77" in error
    assert "executable unavailable" in error
    assert system.signals == []


def test_up_timeout_stops_owned_process_and_preserves_log(config):
    system = FakeSystem(healthy=False)

    with pytest.raises(RuntimeError, match="not ready within 1 second"):
        ModelServerLauncher(config, system=system).up()

    assert system.signals == [(100, signal.SIGTERM)]
    assert not config.state_path.exists()
    assert config.log_path.is_file()


def test_up_does_not_report_ready_after_readiness_deadline(config):
    system = FakeSystem(healthy=True, health_delay=1.1)

    with pytest.raises(RuntimeError, match="not ready within 1 second"):
        ModelServerLauncher(config, system=system).up()

    assert system.signals == [(100, signal.SIGTERM)]


def test_up_rolls_back_process_when_state_cannot_be_recorded(config, monkeypatch):
    system = FakeSystem(healthy=True)
    launcher = ModelServerLauncher(config, system=system)

    def fail_to_write_state(process: ProcessIdentity) -> None:
        del process
        raise OSError("disk full")

    monkeypatch.setattr(launcher, "_write_state", fail_to_write_state)

    with pytest.raises(RuntimeError, match="record process ownership"):
        launcher.up()

    assert system.signals == [(100, signal.SIGTERM)]


@pytest.mark.parametrize(
    ("recorded", "running"),
    [
        (None, None),
        (
            ProcessIdentity(41, "llama-server", "old-start"),
            ProcessIdentity(41, "python", "new-start"),
        ),
    ],
)
def test_down_is_safe_without_matching_owned_process(config, capsys, recorded, running):
    if recorded:
        write_state(config, recorded)
    system = FakeSystem(
        processes={running.pid: running} if running else {},
        port_pid=running.pid if running else None,
    )

    ModelServerLauncher(config, system=system).down()

    assert system.signals == []
    assert not config.state_path.exists()
    assert "not running" in capsys.readouterr().out.lower()


def test_down_terminates_only_matching_owned_process(config, capsys):
    owned = ProcessIdentity(41, "llama-server", "same-start")
    write_state(config, owned)
    system = FakeSystem(processes={41: owned}, port_pid=41)

    ModelServerLauncher(config, system=system).down()

    assert system.signals == [(41, signal.SIGTERM)]
    assert not config.state_path.exists()
    assert "stopped" in capsys.readouterr().out.lower()


def test_down_force_kills_owned_process_after_grace_period(config):
    owned = ProcessIdentity(41, "llama-server", "same-start")
    write_state(config, owned)
    system = FakeSystem(processes={41: owned}, port_pid=41, exit_on_term=False)

    ModelServerLauncher(config, system=system).down()

    assert system.signals == [
        (41, signal.SIGTERM),
        (41, signal.SIGKILL),
    ]
