"""
VectorBT Adapter for Factor Synthesis Backtesting

Provides VectorBT-based portfolio backtesting using from_orders with targetpercent.
Replaces removed Qlib backtest functionality.
"""

from .backtest_runner import VectorBTBacktestRunner
from .alignment_data_writer import AlignmentDataWriter

__all__ = ['VectorBTBacktestRunner', 'AlignmentDataWriter']
