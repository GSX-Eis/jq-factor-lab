"""Input and source identities and atomic disk caching for FactorBase."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Callable, Optional

import numpy as np
import pandas as pd


def factor_core_source_hashes(facade: Path) -> list[str]:
    """Hash extracted FactorBase sources in deterministic filename order."""
    return [
        hashlib.sha256(dependency.read_bytes()).hexdigest()
        for dependency in sorted(facade.with_name("factor_core").glob("*.py"))
    ]


def _get_cache_inputs(self) -> Optional[dict]:
    """Declare every factor-specific parameter and additional data input.

    Implement on the concrete factor class to enable calculation caching.
    Values must be JSON parameters or top-level DataFrames. Read additional
    data before returning and calculate from that same snapshot. None means
    the dependencies cannot be identified completely, so caching is disabled.
    """
    return None


def _get_cache_key(
    self,
    factor_base_cls: type,
    stock_filter_cls: type,
    apply_pit_adjustment: Callable,
    raw_data: Optional[pd.DataFrame] = None,
    *,
    fallback_keys: Optional[list[str]] = None,
) -> Optional[str]:
    """Hash declared inputs; never infer configuration from runtime state."""
    import inspect

    if raw_data is None:
        return None
    # A child can add parameters/dependencies absent from its parent's hook.
    if "_get_cache_inputs" not in vars(type(self)):
        self.logger.warning("Cache disabled: %s must declare _get_cache_inputs", type(self).__name__)
        return None
    # Dependency read failures must propagate, even if an old cache exists.
    inputs = self._get_cache_inputs()
    if inputs is None:
        self.logger.warning("Cache disabled: _get_cache_inputs did not identify all inputs")
        return None

    source_revision = None
    if self._duckdb_adapter is not None:
        source_revision = self._duckdb_adapter.cache_revision()
        if source_revision is None:
            return None  # Unmanaged Parquet must not reuse derived disk caches.
    elif self.db_connection is not None:
        from utils.data_revision import connection_revision
        source_revision = connection_revision(self.db_connection)

    def frame_digest(frame):
        # Include values, ordering, dtypes, categorical metadata, axes and attrs.
        digest = hashlib.sha256()

        class HashWriter:
            def write(self, data):
                digest.update(data)
                return len(data)

        # Stream into the digest instead of retaining the entire pickle too.
        pickle.dump(frame, HashWriter(), protocol=4)
        return digest.hexdigest()

    try:
        if not isinstance(inputs, dict) or any(not isinstance(key, str) for key in inputs):
            raise TypeError("_get_cache_inputs must return a dictionary with string keys")
        declared = {}
        for key, value in inputs.items():
            if isinstance(value, pd.DataFrame):
                declared[key] = ["dataframe", frame_digest(value)]
            else:
                encoded = json.dumps(value, sort_keys=True, allow_nan=False)
                normalized = json.loads(encoded)
                if normalized != value:
                    raise TypeError("Cache input " + key + " must use JSON values without coercion")
                declared[key] = ["json", normalized]
        fields = (
            "factor_name", "start_date", "end_date", "stock_pool", "all_stocks",
            "use_dynamic_pool", "stock_pools_by_period", "dynamic_pool_mode",
            "max_stocks_per_month", "sort_by", "ascending",
            "rebalance_period_months", "rebalance_schedule", "data_frequency",
            "fundamental_table", "price_table", "db_backend", "data_bindings",
            "pit_enabled", "pit_factor_column", "pit_window",
        )
        parameters = {name: getattr(self, name) for name in fields}
        parameters["pool_schedule"] = getattr(self, "pool_schedule", None)
        for name in ('analysis_mode', 'output_frequency', 'session_view', 'label_config',
                     'data_version', 'source_revision', 'calendar_hash', 'rth_reference_calendar'):
            parameters[name] = getattr(self, name, None)
        metadata = getattr(self, 'instrument_metadata', None)
        parameters['instrument_metadata'] = frame_digest(metadata) if isinstance(metadata, pd.DataFrame) else metadata
        calendars = getattr(self, 'session_calendar', None)
        if calendars is not None:
            parameters['session_calendar'] = (calendars.calendar_hash if hasattr(calendars, 'calendar_hash')
                else {instrument: calendar.calendar_hash for instrument, calendar in calendars.items()})
        if self._duckdb_adapter is not None or self.db_connection is not None:
            parameters["data_access"] = self._get_local_data_config().as_dict()
            parameters["source_policy"] = self.source_policy
        for name in ("duckdb_path", "parquet_dir"):
            value = getattr(self, name)
            parameters[name] = str(Path(value).resolve()) if value is not None else None
        filters = []
        for condition in self.stock_filters or []:
            if type(condition) is not stock_filter_cls:
                raise TypeError("stock_filters must contain StockFilter instances")
            value = condition.value
            if condition.operator in {"in", "not_in"} and isinstance(value, (set, frozenset)):
                value = sorted(value, key=lambda member: json.dumps(member, sort_keys=True, allow_nan=False))
            filters.append({"field": condition.field, "operator": condition.operator,
                            "value": value})
        parameters["stock_filters"] = filters
        sources = []
        previous_public = None
        implementations = [cls for cls in type(self).__mro__ if cls is not object]
        implementations.append(apply_pit_adjustment)
        bar_grid = getattr(getattr(self, '_frequency_route', None), 'bar_grid', False)
        if bar_grid:
            from backtest_and_analysis.time_series.factor_bridge import session_grid
            implementations.append(session_grid)
        for implementation in implementations:
            source_file = inspect.getsourcefile(implementation)
            if not source_file:
                raise ValueError("Cannot identify source for " + implementation.__qualname__)
            path = Path(source_file)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if implementation is factor_base_cls and not bar_grid:
                from utils.legacy_cache_identity import factor_base_digest
                legacy_digest = factor_base_digest(source_file)
                sources.append(legacy_digest)
                if legacy_digest != digest:
                    if fallback_keys is not None:
                        mapping_path = Path(__file__).parents[1] / "legacy_cache_identity.json"
                        mapping = json.loads(mapping_path.read_text())
                        if digest == mapping["equivalent_sha256"] and legacy_digest == mapping["baseline_sha256"]:
                            previous_public = mapping.get("previous_public_sha256")
                            facade_index = len(sources) - 1
                    continue  # The legacy mapping already tracks the extracted files.
            else:
                sources.append(digest)
            if implementation is factor_base_cls:
                # Bar and edited facades use source hashes outside the frozen mapping.
                sources.extend(factor_core_source_hashes(path))
        identity = {"schema": 3, "class": type(self).__qualname__,
                    "source": sources, "parameters": parameters,
                    "runtime": [pd.__version__, np.__version__],
                    "inputs": declared, "source_revision": source_revision,
                    "raw_data": frame_digest(raw_data)}

        def render_key():
            encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
            return "factor-v3-" + hashlib.sha256(encoded.encode()).hexdigest()

        key = render_key()
        if previous_public is not None:
            sources[facade_index] = previous_public
            fallback_keys.append(render_key())
        return key
    except (TypeError, ValueError, OSError, pickle.PicklingError) as exc:
        self.logger.warning("Cache disabled: %s", exc)
        return None


def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
    """完整写入临时文件后原子替换缓存，失败时保留原文件。

    Args:
        data: 要缓存的数据
        cache_key: 缓存键
    """
    if not self.cache_dir or not cache_key:
        return

    try:
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with tempfile.TemporaryDirectory(prefix=".factor-cache-", dir=self.cache_dir) as temporary_dir:
            temporary_path = os.path.join(temporary_dir, "data.pkl")
            with open(temporary_path, 'wb') as f:
                pickle.dump(data, f)
            os.replace(temporary_path, cache_path)
        self.logger.debug(f"数据已缓存到: {cache_path}")
    except Exception as e:
        self.logger.warning(f"缓存数据失败: {e}")


def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
    """从缓存加载数据

    Args:
        cache_key: 缓存键

    Returns:
        已规范化的因子宽表；缓存不存在、不可读或结构无效时返回None
    """
    if not self.cache_dir or not cache_key:
        return None

    cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("缓存结果必须是非空因子 DataFrame")
        if (not isinstance(data.index, pd.DatetimeIndex) or data.index.name != 'date'
                or data.index.hasnans or not data.index.is_unique):
            raise ValueError("缓存结果必须使用名为 date 的非空且唯一的日期索引")
        if (not data.columns.is_unique
                or not all(isinstance(code, str) and code for code in data.columns)):
            raise ValueError("缓存结果股票列必须是唯一的非空字符串")
        self.logger.debug(f"从缓存加载数据: {cache_path}")
        return data
    except Exception as e:
        self.logger.warning(f"加载缓存失败: {e}")
        return None
