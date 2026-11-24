"""
VectorBT Adapter for Factor Synthesis Backtesting

Provides VectorBT-based portfolio backtesting using from_orders with targetvalue.
Replaces removed Qlib backtest functionality.
"""

from .alignment_data_writer import AlignmentDataWriter
from .backtest_runner import VectorBTBacktestRunner
from .slot_provider import SlotDataProvider
from .universe_filter import (
    UniverseFilterError,
    UniverseFilterPlugin,
    apply_universe_filter,
    resolve_universe_filter_plugin,
)

__all__ = [
    "AlignmentDataWriter",
    "SlotDataProvider",
    "UniverseFilterError",
    "UniverseFilterPlugin",
    "VectorBTBacktestRunner",
    "apply_universe_filter",
    "resolve_universe_filter_plugin",
]
