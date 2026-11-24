"""Physical market database declarations and validated handles."""

from .bindings import CrossMarketAttachSpec, open_market_db, resolve_market_db_path
from .errors import MarketDbProfileError
from .profile import MarketDbProfile, load_market_db_profile_yaml
from .storage import (
    adopt_market_db_sidecar,
    init_market_db,
    market_db_sidecar_path,
    migrate_market_db_profile,
    read_market_db_profile,
)

__all__ = [
    "CrossMarketAttachSpec",
    "MarketDbProfile",
    "MarketDbProfileError",
    "adopt_market_db_sidecar",
    "init_market_db",
    "load_market_db_profile_yaml",
    "market_db_sidecar_path",
    "migrate_market_db_profile",
    "open_market_db",
    "read_market_db_profile",
    "resolve_market_db_path",
]
