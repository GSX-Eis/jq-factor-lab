"""Classroom example: five-session close-to-close return."""
from utils.factor_base import FactorBase


class DemoMomentum(FactorBase):
    def _get_cache_inputs(self):
        # The five-session window is fixed in source; all data arrives in raw_data.
        return {}

    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(["close", "factor"], table_name="price_data_daily")

    def calculate_factor(self, data):
        prices = data.pivot(index="date", columns="code", values="close").sort_index()
        return prices.pct_change(periods=5, fill_method=None)
