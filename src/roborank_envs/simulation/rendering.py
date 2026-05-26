from __future__ import annotations

import math
import os


DEFAULT_RENDER_INTERVAL_SEC = 0.1


def configured_render_interval_sec(default: float = DEFAULT_RENDER_INTERVAL_SEC) -> float:
    try:
        fallback = float(default)
    except (TypeError, ValueError):
        fallback = DEFAULT_RENDER_INTERVAL_SEC

    try:
        configured = float(os.environ.get("ROBORANK_RENDER_INTERVAL_SEC", str(fallback)))
    except ValueError:
        return fallback

    return configured if math.isfinite(configured) and configured > 0.0 else fallback
