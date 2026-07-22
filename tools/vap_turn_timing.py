#!/usr/bin/env python3
"""Repository wrapper for zoom1_dialogue_tts.vap_cli."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zoom1_dialogue_tts.vap_cli import main


if __name__ == "__main__":
    main()
