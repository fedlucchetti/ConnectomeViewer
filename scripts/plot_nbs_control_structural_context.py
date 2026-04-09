#!/usr/bin/env python3
"""Thin wrapper to launch the mrsitoolbox NBS structural-context plotting script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


VIEWER_ROOT = Path(__file__).resolve().parents[1]
MRSITOOLBOX_ROOT = VIEWER_ROOT.parent / 'mrsitoolbox'
TARGET_SCRIPT = MRSITOOLBOX_ROOT / 'experiments' / 'nbs_metsim_rc_overlap' / 'plot_nbs_control_structural_context.py'


def main() -> None:
    if not TARGET_SCRIPT.exists():
        raise FileNotFoundError(f'Could not find target script: {TARGET_SCRIPT}')
    root_text = str(MRSITOOLBOX_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    runpy.run_path(str(TARGET_SCRIPT), run_name='__main__')


if __name__ == '__main__':
    main()
