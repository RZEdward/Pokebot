"""Inspect transcript parsing and safe skill dispatch with a sample phrase."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from hitmonlee_sim.commands import dispatch_intent, parse_transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="Text received from a speech recognizer")
    parser.add_argument("--confidence", type=float, default=1.0)
    args = parser.parse_args()

    intent = parse_transcript(args.transcript, confidence=args.confidence, source="cli_demo")
    dispatch = dispatch_intent(intent)
    print(json.dumps({"intent": asdict(intent), "dispatch": asdict(dispatch)}, indent=2))


if __name__ == "__main__":
    main()
