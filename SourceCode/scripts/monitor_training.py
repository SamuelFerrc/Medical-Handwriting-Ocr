from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from med_ocr_research.experiments.monitor import MonitorServer, access_token, monitor_address
from med_ocr_research.experiments.progress import runtime_directory
from med_ocr_research.experiments.seeds import DEFAULT_CONFIG, DEFAULT_MATRIX, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Painel de acompanhamento dos experimentos.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    host, port = monitor_address(args.config)
    settings = load_settings(args.config, args.matrix)
    token = access_token(runtime_directory(settings.root))
    with MonitorServer((args.host or host, port if args.port is None else args.port), settings, token) as server:
        print(f"Painel ouvindo em {server.server_address[0]}:{server.server_address[1]}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
