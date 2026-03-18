#!/usr/bin/env python3
"""Compatibility entry point for gradient rendering dialogs."""

try:
    from .plot_msmode import (
        GradientClassificationDialog,
        GradientScatterDialog,
        GradientSurfaceDialog,
        MSModeSurfaceDialog,
    )
except Exception:
    from plot_msmode import (
        GradientClassificationDialog,
        GradientScatterDialog,
        GradientSurfaceDialog,
        MSModeSurfaceDialog,
    )


__all__ = [
    "GradientSurfaceDialog",
    "GradientScatterDialog",
    "GradientClassificationDialog",
    "MSModeSurfaceDialog",
]
