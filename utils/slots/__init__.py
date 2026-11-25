"""MarketSlot component contracts, registry and declaration loader."""
from .contracts import (
    AuxFeatureSlot,
    BarFeedSlot,
    CalendarSlot,
    ExecutionProfile,
    InstrumentSpec,
    LabelExecutionProfile,
    MarketSlot,
    MicrostructureSlot,
    PortfolioExecutionProfile,
    UniverseSlot,
)
from .registry import SlotRegistry
from .schema import ProviderConfig, SlotConfig, load_slot_config, slot_config_hash

__all__ = [
    'AuxFeatureSlot', 'BarFeedSlot', 'CalendarSlot', 'ExecutionProfile', 'InstrumentSpec',
    'LabelExecutionProfile', 'MarketSlot', 'MicrostructureSlot', 'PortfolioExecutionProfile',
    'UniverseSlot', 'SlotRegistry', 'ProviderConfig', 'SlotConfig', 'load_slot_config',
    'slot_config_hash',
]
