from __future__ import annotations

import argparse
from pathlib import Path

from interp_lab.activations import collect_activations
from interp_lab.browser import build_browser, serve_reports
from interp_lab.config import ensure_dirs, load_config
from interp_lab.nla_toy import train_nla_toy
from interp_lab.patching import run_patching
from interp_lab.sae import analyze_features, train_sae


def parser(description: str) -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=description)
    command.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return command


def collect_activations_main() -> None:
    args = parser("Collect residual-stream activations.").parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    print(collect_activations(config))


def run_patching_main() -> None:
    args = parser("Run clean/corrupt activation patching.").parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    for path in run_patching(config).values():
        print(path)


def train_sae_main() -> None:
    args = parser("Train a sparse autoencoder.").parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    print(train_sae(config))


def analyze_features_main() -> None:
    args = parser("Analyze SAE feature cards.").parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    for path in analyze_features(config).values():
        print(path)


def train_nla_toy_main() -> None:
    args = parser("Train the tiny NLA-inspired text bottleneck.").parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    for path in train_nla_toy(config).values():
        print(path)


def serve_browser_main() -> None:
    command = parser("Serve generated reports as a local browser.")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--build-only", action="store_true", help="Only write reports/index.html.")
    args = command.parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    reports_dir = Path(config["outputs"]["reports_dir"])
    if args.build_only:
        print(build_browser(reports_dir))
    else:
        serve_reports(reports_dir, args.port)
