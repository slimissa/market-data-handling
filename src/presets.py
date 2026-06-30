"""
presets.py — Signal and pipeline presets for different market types.
QuantOS Market Data Pipeline

Usage:
    from presets import get_preset, list_presets, apply_preset

    config = get_preset("crypto")
    apply_preset(pipeline.config, "crypto")

Presets are deep-merged OVER the pipeline defaults. They do not replace the
entire config — they override only the keys that genuinely differ for the
target market type. Keys not mentioned in a preset retain their defaults.

IMPORTANT on threshold values:
    The crypto and forex presets below are starting points derived from
    general quant finance principles (higher vol -> wider thresholds,
    mean-reverting FX pairs -> tighter zscore windows), NOT from a
    validated backtest on those asset classes. Treat them as informed
    initial guesses and expect to calibrate them against real data before
    using them in any decision-making context.
"""

from copy import deepcopy
from typing import Dict, Any


# ======================================================================
# Preset definitions
# ======================================================================

PRESET_EQUITIES: Dict[str, Any] = {
    # The pipeline's own defaults — documented here for completeness so you
    # can see exactly what crypto/forex are changing.
    "_description": "US equity defaults. Daily data, US/Eastern timezone.",
    "timezone": "US/Eastern",
    "interval": "1d",
    "signals": {
        "rsi": {"oversold": 30, "overbought": 70, "exit": 50, "smoothing": 1},
        "zscore": {"entry_threshold": 2.0, "exit_threshold": 0.0, "window": 60},
        "bollinger": {"squeeze_percentile": 20.0, "max_holding_bars": 10},
        "vol_scale": {"floor": 0.0, "ceiling": 2.0},
        "holding": {"min_bars": 2, "max_bars": 20},
        "ensemble": {"method": "majority_vote"},
    },
    "features": {
        "volatility": {"windows": [5, 21, 63]},
        "bollinger": {"window": 20, "num_std": 2.0},
        "price_zscore": {"windows": [20, 60]},
    },
    "regime_filter": {
        "vol_percentile": 20.0,
        "max_trend_annual": 0.06,
        "bb_percentile": 25.0,
    },
}

PRESET_CRYPTO: Dict[str, Any] = {
    # Crypto markets: 24/7 trading, much higher volatility (daily vol
    # 3-5x equities), fat tails, momentum-driven. Adjustments:
    #   - Wider RSI bands (crypto rarely hits 30/70 without extreme moves)
    #   - Wider zscore thresholds (2.5 sigma instead of 2.0)
    #   - Shorter holding windows (crypto trends revert faster)
    #   - Tighter vol percentile gates (in crypto, "high vol" is relative)
    #   - Wider Bollinger bands (2.5 std instead of 2.0)
    "_description": "Crypto defaults. Higher vol, wider thresholds, UTC timezone.",
    "timezone": "UTC",
    "interval": "1d",
    "signals": {
        "rsi": {"oversold": 25, "overbought": 75, "exit": 50, "smoothing": 2},
        "zscore": {"entry_threshold": 2.5, "exit_threshold": 0.5, "window": 30},
        "bollinger": {"squeeze_percentile": 30.0, "max_holding_bars": 7},
        "vol_scale": {"floor": 0.0, "ceiling": 1.5},  # cap at 1.5x — crypto already volatile
        "holding": {"min_bars": 1, "max_bars": 10},
        "ensemble": {"method": "majority_vote"},
    },
    "features": {
        "volatility": {"windows": [3, 14, 30]},   # shorter windows for faster crypto moves
        "bollinger": {"window": 20, "num_std": 2.5},
        "price_zscore": {"windows": [14, 30]},
    },
    "regime_filter": {
        "vol_percentile": 35.0,   # looser — in crypto even "low vol" is high by equity standards
        "max_trend_annual": 0.15,
        "bb_percentile": 35.0,
    },
}

PRESET_FOREX: Dict[str, Any] = {
    # Forex markets: highly mean-reverting, institutional-dominated, low
    # directional vol. Adjustments:
    #   - Tighter RSI bands (FX rarely makes extreme directional moves)
    #   - Shorter zscore window (FX pairs revert quickly)
    #   - Longer holding windows (FX trends are slow to develop and long)
    #   - Tighter vol percentile gates (FX vol clustering is persistent)
    "_description": "Forex defaults. Mean-reverting, tighter thresholds, UTC timezone.",
    "timezone": "UTC",
    "interval": "1d",
    "signals": {
        "rsi": {"oversold": 35, "overbought": 65, "exit": 50, "smoothing": 1},
        "zscore": {"entry_threshold": 1.5, "exit_threshold": 0.0, "window": 20},
        "bollinger": {"squeeze_percentile": 15.0, "max_holding_bars": 15},
        "vol_scale": {"floor": 0.0, "ceiling": 2.0},
        "holding": {"min_bars": 3, "max_bars": 30},   # FX trends take longer
        "ensemble": {"method": "majority_vote"},
    },
    "features": {
        "volatility": {"windows": [5, 21, 63]},
        "bollinger": {"window": 20, "num_std": 1.8},   # tighter bands for lower-vol FX
        "price_zscore": {"windows": [10, 40]},
    },
    "regime_filter": {
        "vol_percentile": 25.0,
        "max_trend_annual": 0.04,   # FX trends are weak — only gate on very weak trends
        "bb_percentile": 20.0,
    },
}

# ── Registry ─────────────────────────────────────────────────────────

_PRESETS: Dict[str, Dict[str, Any]] = {
    "equities": PRESET_EQUITIES,
    "crypto":   PRESET_CRYPTO,
    "forex":    PRESET_FOREX,
}


# ======================================================================
# Public API
# ======================================================================

def list_presets() -> Dict[str, str]:
    """Return {name: description} for all available presets."""
    return {name: p.get("_description", "") for name, p in _PRESETS.items()}


def get_preset(name: str) -> Dict[str, Any]:
    """Return a deep copy of the named preset dict."""
    name = name.lower().strip()
    if name not in _PRESETS:
        available = sorted(_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available: {available}"
        )
    return deepcopy(_PRESETS[name])


def apply_preset(config: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Deep-merge a preset over an existing config dict IN PLACE.

    The preset overrides only the keys it defines. Keys in `config` that
    are not mentioned in the preset are preserved unchanged.

    Args:
        config:      The pipeline config dict (modified in place).
        preset_name: One of 'equities', 'crypto', 'forex'.

    Returns:
        The modified config (same object, mutated).
    """
    preset = get_preset(preset_name)
    _deep_merge(config, preset)
    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge `override` into `base` in place.
    Non-dict values in `override` replace values in `base`.
    Dict values are merged recursively. Private keys (starting with '_')
    are skipped (used for metadata like _description).
    """
    for key, value in override.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base