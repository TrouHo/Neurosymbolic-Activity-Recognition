"""Run with:  python -m emob_review  (from review_app\\backend)"""
import argparse

import uvicorn

from .app import create_app
from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="eMOB Data Review (local)")
    ap.add_argument("--config", help="path to config.json (default: review_app\\config.json)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"Refusing to bind to non-local host {cfg.host!r}; this app is local-only.")
    print(f"eMOB Data Review: http://{cfg.host}:{cfg.port}/   (outputs: {cfg.output_dir})")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
