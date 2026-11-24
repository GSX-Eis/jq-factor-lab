"""Registry of explicitly supplied market adapters."""
from .contracts import MarketSlot


class SlotRegistry:
    def __init__(self) -> None:
        self._slots: dict[str, MarketSlot] = {}

    def register(self, slot: MarketSlot) -> None:
        if not isinstance(slot, MarketSlot):
            raise TypeError('register expects a MarketSlot')
        if slot.slot_id in self._slots:
            raise ValueError(f'MarketSlot already registered: {slot.slot_id!r}')
        self._slots[slot.slot_id] = slot

    def get(self, slot_id: str) -> MarketSlot:
        try:
            return self._slots[slot_id]
        except KeyError:
            raise KeyError(f'Unknown MarketSlot: {slot_id!r}') from None
