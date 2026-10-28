"""Estado persistente e exclusao mutua do executor de treinamentos."""
from __future__ import annotations

import fcntl
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_STATES = {"validating", "running", "aggregating"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid: int) -> dict[str, Any] | None:
    """O instante de inicio e boot_id evitam confundir um PID reutilizado."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def runtime_directory(root: Path) -> Path:
    return root / "outputs/monitor"


class PipelineProgress:
    def __init__(self, directory: Path, seeds: tuple[int, ...]) -> None:
        self.directory = directory
        self.state = {"status": "validating", "phase": "validating", "seeds": list(seeds),
                      "process": process_identity(os.getpid()), "started_at": utc_now(),
                      "current_run": None, "finished_steps": 0, "total_steps": 0}
        self.lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "pipeline.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Ja existe um executor de treinamento ativo neste projeto.") from None
        self.previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._stopped)
        self.update()
        return self

    @staticmethod
    def _stopped(signum, frame) -> None:
        raise KeyboardInterrupt("Treinamento interrompido por SIGTERM.")

    def update(self, **changes) -> None:
        self.state.update(changes, updated_at=utc_now())
        temporary = self.directory / f"pipeline.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "pipeline.json")

    def __exit__(self, kind, error, traceback) -> None:
        try:
            if kind is None:
                self.update(status="completed", phase="completed", current_run=None, finished_at=utc_now())
            else:
                stopped = issubclass(kind, (KeyboardInterrupt, SystemExit))
                self.update(status="stopped" if stopped else "failed", error=str(error),
                            exit_code=getattr(error, "returncode", None), finished_at=utc_now())
        finally:
            signal.signal(signal.SIGTERM, self.previous_sigterm)
            self.lock.close()


def pipeline_status(directory: Path) -> dict[str, Any]:
    state = read_json(directory / "pipeline.json")
    if not state:
        return {"status": "idle", "phase": "idle", "current_run": None}
    identity = state.get("process")
    if state.get("status") in ACTIVE_STATES:
        if not identity or process_identity(identity["pid"]) != identity:
            state = {**state, "status": "interrupted",
                     "error": "O processo terminou sem registrar a conclusao. Consulte o log."}
    return state
