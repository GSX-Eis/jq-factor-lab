# 最小因子模板

此处只提供接口骨架、输入检查和合成数据检查。没有因子公式、训练配置、研究窗口或表现阈值。将第一段保存到自己的因子模块，改名并实现 `calculate_factor` 中的计算部分；未实现时会明确抛出 `NotImplementedError`，不能直接用于评估或聚宽运行。

## 类骨架与输入检查

示例检查日频 `close` 输入，保留缺失价格；算法需要其他字段时应补充相应校验。复权数据由运行入口准备，直接调用计算方法时由调用者负责。

```python
import numpy as np
import pandas as pd
from utils.factor_base import FactorBase


def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "close"}
    if not required.issubset(data.columns):
        raise ValueError(f"missing columns: {sorted(required - set(data.columns))}")
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("missing date")
    if frame["date"].dt.tz is not None:
        raise ValueError("convert dates to the agreed exchange timezone first")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("daily input must use trading dates without intraday times")
    if not frame["code"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("code must be a non-empty string")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("duplicate date/code")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise").astype(float)
    observed = frame["close"].dropna()
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("observed close must be finite and positive")
    return frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


class MyFactor(FactorBase):
    def _get_raw_data_for_factor(self) -> pd.DataFrame:
        return self._fetch_raw_data(
            fields=["close", "factor"], table_name="price_data_daily"
        )

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = prepare_input(data)
        prices = frame.pivot(index="date", columns="code", values="close").sort_index()
        # Implement your formula using only data available at each output date.
        # Return a numeric DataFrame with the same date/code axis convention as prices.
        raise NotImplementedError("implement the factor calculation before running")
```

不要为了让模板通过而把占位符替换为全零输出。实现后为类名补充 [requirements 声明](FACTOR_INTERFACE.md)。

## 合成数据检查

以下代码可与第一段放入同一测试模块；若分文件保存，需要从自己的因子模块导入 `prepare_input`。`test_input_contract` 可直接运行。`check_factor_contract` 接收已实现因子的计算函数，检查输出格式、输入顺序无关性和追加未来数据后历史结果是否变化。

```python
import pytest


def synthetic_prices():
    dates = pd.bdate_range("2024-01-02", periods=8)
    return pd.DataFrame([
        {"date": date, "code": code, "close": float(10 + day + asset), "factor": 1.0}
        for day, date in enumerate(dates)
        for asset, code in enumerate(["SYNTH_A", "SYNTH_B"])
    ])


def test_input_contract():
    data = synthetic_prices()
    pd.testing.assert_frame_equal(
        prepare_input(data), prepare_input(data.sample(frac=1, random_state=0))
    )
    with pytest.raises(ValueError, match="duplicate"):
        prepare_input(pd.concat([data, data.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="missing columns"):
        prepare_input(data.drop(columns="close"))
    for value in [np.inf, -np.inf, 0.0, -1.0]:
        bad = data.copy()
        bad.loc[0, "close"] = value
        with pytest.raises(ValueError, match="finite and positive"):
            prepare_input(bad)
    missing = data.copy()
    missing.loc[0, "close"] = np.nan
    assert prepare_input(missing)["close"].isna().sum() == 1


def check_factor_contract(calculate):
    data = synthetic_prices()
    dates = pd.DatetimeIndex(sorted(data["date"].unique()), name="date")
    codes = sorted(data["code"].unique())

    def checked(frame):
        result = calculate(frame.copy())
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "date"
        assert result.index.is_unique and result.index.is_monotonic_increasing
        assert result.columns.is_unique
        assert all(isinstance(code, str) for code in result.columns)
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)
        assert result.index.isin(frame["date"]).all()
        assert result.columns.isin(codes).all()
        assert not np.isinf(result.to_numpy(dtype=float)).any()
        return result

    full = checked(data)
    assert full.notna().to_numpy().any(), "increase fixture length if warm-up consumes all rows"
    shuffled = checked(data.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(full, shuffled)
    cutoff = dates[-2]
    prefix = checked(data[data["date"] <= cutoff])
    # Reindex to expose missing historical rows as well as changed values.
    earlier = dates[dates <= cutoff]
    pd.testing.assert_frame_equal(
        full.reindex(index=earlier, columns=codes),
        prefix.reindex(index=earlier, columns=codes),
    )
```

实现后在测试中调用 `check_factor_contract(instance.calculate_factor)`，其中 `instance` 是按实际运行配置构造的因子实例。固定日期、8 行日期与两只虚构股票仅用于小型测试夹具；若预热期更长，应扩大夹具，并加入算法自己的边界用例。这段检查不包含 IC 或收益判定，也不是完整的前瞻偏差证明。

输入带复权事件时，还需要按每个截断时点重新构造复权输入后比较历史输出；这里的 `factor=1` 只检查纯计算接口。现有 [公开演示测试](../tests/test_public_demo.py) 另有复权与执行滞后的合成用例。相关语义见 [数据语义](SEMANTICS.md)。
