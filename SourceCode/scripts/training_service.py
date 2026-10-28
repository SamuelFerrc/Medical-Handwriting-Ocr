"""Executa treinamento e painel em servicos systemd do usuario (Linux)."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_ocr_research.experiments.monitor import access_token, monitor_address
from med_ocr_research.experiments.progress import pipeline_status, runtime_directory
from med_ocr_research.experiments.seeds import DEFAULT_CONFIG, DEFAULT_MATRIX


def unit_names(root: Path) -> tuple[str, str]:
    identifier = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10]
    prefix = f"med-ocr-{identifier}"
    return f"{prefix}-training.service", f"{prefix}-monitor.service"


def quote_unit(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def service_text(command: list[str], root: Path, log: Path, *, monitor: bool) -> str:
    if any("\n" in str(value) or "\r" in str(value) for value in [root, log, *command]):
        raise ValueError("Caminhos/argumentos do servico nao podem conter quebras de linha.")
    description = "Painel de acompanhamento OCR" if monitor else "Treinamento OCR por seeds"
    # ':' desativa expansao de variaveis de ambiente nos argumentos do ExecStart.
    return (
        f"[Unit]\nDescription={description}\n\n[Service]\nType=exec\n"
        f"WorkingDirectory={str(root).replace('%', '%%')}\n"
        f"ExecStart=:{' '.join(quote_unit(arg) for arg in command)}\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        f"StandardOutput=append:{str(log).replace('%', '%%')}\n"
        f"StandardError=append:{str(log).replace('%', '%%')}\n"
        "KillMode=control-group\nTimeoutStopSec=30\nUMask=0077\n"
        + ("Restart=on-failure\nRestartSec=5\n" if monitor else "Restart=no\n")
    )


def systemctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *arguments], check=check, text=True, capture_output=True)


def install_services(args) -> tuple[str, str]:
    directory = runtime_directory(PROJECT_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    access_token(directory)
    destination = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
    destination.mkdir(parents=True, exist_ok=True)
    training_unit, monitor_unit = unit_names(PROJECT_ROOT)
    shared = ["--config", str(args.config.absolute()), "--matrix", str(args.matrix.absolute())]
    train = [args.python, "-u", str(PROJECT_ROOT / "scripts/train_all_seeds.py"), *shared, "--python", args.python]
    host, port = monitor_address(args.config)
    host, port = args.host or host, port if args.port is None else args.port
    if not 1 <= port <= 65535:
        raise ValueError("A porta deve estar entre 1 e 65535.")
    monitor = [args.python, "-u", str(PROJECT_ROOT / "scripts/monitor_training.py"), *shared,
               "--host", host, "--port", str(port)]
    (destination / training_unit).write_text(service_text(train, PROJECT_ROOT, directory / "training.log", monitor=False), encoding="utf-8")
    (destination / monitor_unit).write_text(service_text(monitor, PROJECT_ROOT, directory / "monitor.log", monitor=True), encoding="utf-8")
    # Registra o endereco efetivo para o comando status, inclusive overrides da CLI.
    import json
    (directory / "address.json").write_text(json.dumps({"host": host, "port": port}), encoding="utf-8")
    systemctl("daemon-reload")
    return training_unit, monitor_unit


def print_access(config: Path) -> None:
    from med_ocr_research.experiments.progress import read_json
    directory = runtime_directory(PROJECT_ROOT)
    host, port = monitor_address(config)
    address = read_json(directory / "address.json")
    host, port = address.get("host", host), address.get("port", port)
    token_file = directory / "access_token"
    if token_file.is_file():
        token = token_file.read_text().strip()
        local_host = "127.0.0.1" if host == "0.0.0.0" else host
        print(f"Painel: http://{local_host}:{port}/#token={token}")
        if host == "0.0.0.0":
            print(f"Na rede local: substitua 127.0.0.1 pelo IP deste computador; mantenha a porta {port} e o token.")


def wait_for_monitor() -> None:
    from med_ocr_research.experiments.progress import read_json
    directory = runtime_directory(PROJECT_ROOT)
    address = read_json(directory / "address.json")
    host = "127.0.0.1" if address["host"] == "0.0.0.0" else address["host"]
    token = access_token(directory)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        connection = http.client.HTTPConnection(host, address["port"], timeout=5)
        try:
            connection.request("GET", "/api/status", headers={"Authorization": f"Bearer {token}"})
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.2)
    raise RuntimeError(f"O painel nao respondeu. Consulte {directory / 'monitor.log'} e o comando status.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "start", "monitor", "status", "stop", "stop-monitor"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--python", default=sys.executable, help="Python do ambiente de treinamento; exemplo: .venv/bin/python")
    parser.add_argument("--host", default=None, help="127.0.0.1 para acesso local; 0.0.0.0 para rede local.")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    executable = shutil.which(args.python)
    if not executable:
        parser.error(f"Interpretador nao encontrado: {args.python}")
    # Nao resolver symlinks: o caminho .venv/bin/python identifica o virtualenv.
    args.python = str(Path(executable).absolute())
    training_unit, monitor_unit = unit_names(PROJECT_ROOT)
    try:
        if args.action in {"install", "start", "monitor"}:
            if args.action == "start":
                dependencies = subprocess.run(
                    [args.python, "-c", "import torch, torchvision, pandas, transformers, timm, pytorch_lightning, yaml"],
                    text=True, capture_output=True,
                )
                if dependencies.returncode:
                    parser.exit(1, "O Python selecionado nao possui as dependencias de treinamento funcionais. "
                                "Instale requirements.txt e requirements-train.txt nesse ambiente ou use --python.\n"
                                + dependencies.stderr)
            training_unit, monitor_unit = install_services(args)
            if args.action in {"start", "monitor"}:
                systemctl("restart", monitor_unit)
                wait_for_monitor()
            if args.action == "start":
                systemctl("start", training_unit)
                print("Treinamento iniciado pelo systemd. Pode fechar o terminal e o VS Code.")
            elif args.action == "monitor":
                print("Painel iniciado em segundo plano. Nenhum treinamento foi iniciado.")
            else:
                print("Servicos instalados; nenhum processo foi iniciado.")
        elif args.action == "stop":
            systemctl("stop", training_unit)
            print("Treinamento e seus subprocessos interrompidos. O painel continua disponivel.")
        elif args.action == "stop-monitor":
            systemctl("stop", monitor_unit)
            print("Painel parado. O treinamento nao foi interrompido.")
        else:
            for unit in (training_unit, monitor_unit):
                result = systemctl("show", unit, "--property=LoadState,ActiveState,SubState,ExecMainStatus", check=False)
                print(unit + "\n" + result.stdout.strip())
            state = pipeline_status(runtime_directory(PROJECT_ROOT))
            print(f"Executor: {state['status']}; execucao atual: {state.get('current_run') or '-'}")
            if state.get("error"):
                print(state["error"])
        print_access(args.config)
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"Falha no systemd do usuario: {exc.stderr or exc}\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
