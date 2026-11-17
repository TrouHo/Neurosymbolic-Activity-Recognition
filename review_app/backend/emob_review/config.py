"""Configuration loading and path-safety helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2]  # review_app/


@dataclass
class PairingRule:
    name: str
    csv_regex: str
    video_regex: str


@dataclass
class Config:
    data_roots: list[Path]
    scan_dirs: list[Path]
    output_dir: Path
    pairing_rules: list[PairingRule] = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8765

    def __post_init__(self) -> None:
        # Normalise (absolute, long-form) so path-containment checks compare like with like.
        self.data_roots = [Path(p).resolve() for p in self.data_roots]
        self.scan_dirs = [Path(p).resolve() for p in self.scan_dirs]
        self.output_dir = Path(self.output_dir).resolve()

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data_roots=[Path(p).resolve() for p in d["data_roots"]],
            scan_dirs=[Path(p).resolve() for p in d.get("scan_dirs", [])],
            output_dir=Path(d["output_dir"]).resolve(),
            pairing_rules=[PairingRule(**{k: r[k] for k in ("name", "csv_regex", "video_regex")})
                           for r in d.get("pairing_rules", [])],
            host=d.get("host", "127.0.0.1"),
            port=int(d.get("port", 8765)),
        )


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path or os.environ.get("EMOB_REVIEW_CONFIG") or APP_DIR / "config.json")
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}. Copy config.example.json to config.json and edit the paths.")
    return Config.from_dict(json.loads(p.read_text(encoding="utf-8")))


class PathNotAllowed(Exception):
    pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_read_path(cfg: Config, raw: str | os.PathLike) -> Path:
    """Resolve a user-supplied path and require it to lie inside a data root or the output dir."""
    p = Path(raw).resolve()
    roots = [*cfg.data_roots, cfg.output_dir]
    if not any(_within(p, r) for r in roots):
        raise PathNotAllowed(f"Path is outside the configured roots: {p}")
    return p


def resolve_write_path(cfg: Config, raw: str | os.PathLike) -> Path:
    """Writes are only ever allowed inside the output dir, never inside a data root."""
    p = Path(raw).resolve()
    if not _within(p, cfg.output_dir):
        raise PathNotAllowed(f"Writes are only allowed inside the output dir: {p}")
    if any(_within(p, r) for r in cfg.data_roots):
        raise PathNotAllowed(f"Refusing to write inside a source data root: {p}")
    return p
