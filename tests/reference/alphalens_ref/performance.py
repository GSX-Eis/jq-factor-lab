#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Excerpted from alphalens-reloaded 0.4.6, alphalens/performance.py.
# Unused functions and imports were omitted for this test-only copy.


import numpy as np
import pandas as pd
from scipy import stats

from . import utils


def factor_information_coefficient(factor_data, group_adjust=False, by_group=False):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_adjust : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):
        f = group["factor"]
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)].apply(
            lambda x: stats.spearmanr(x, f)[0]
        )
        return _ic

    date_idx = factor_data.index.names.index("date")
    freq = factor_data.index.levels[date_idx].freq

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values("date")]

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data, grouper + ["group"])
    if by_group:
        grouper.append("group")

    ic = factor_data.groupby(grouper, observed=True).apply(src_ic)
    if by_group:
        return ic
    else:
        return ic.asfreq(freq)


def mean_return_by_quantile(
    factor_data,
    by_date=False,
    by_group=False,
    demeaned=True,
    group_adjust=False,
):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    by_date : bool
        If True, compute quantile bucket returns separately for each date.
    by_group : bool
        If True, compute quantile bucket returns separately for each group.
    demeaned : bool
        Compute demeaned mean returns (long short portfolio)
    group_adjust : bool
        Returns demeaning will occur on the group level.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values("date")] + ["group"]
        factor_data = utils.demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = utils.demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ["factor_quantile", factor_data.index.get_level_values("date")]

    if by_group:
        grouper.append("group")

    group_stats = factor_data.groupby(grouper, observed=True)[
        utils.get_forward_returns_columns(factor_data.columns)
    ].agg(["mean", "std", "count"])

    mean_ret = group_stats.T.xs("mean", level=1).T

    if not by_date:
        grouper = [mean_ret.index.get_level_values("factor_quantile")]
        if by_group:
            grouper.append(mean_ret.index.get_level_values("group"))
        group_stats = mean_ret.groupby(grouper).agg(["mean", "std", "count"])
        mean_ret = group_stats.T.xs("mean", level=1).T

    std_error_ret = group_stats.T.xs("std", level=1).T / np.sqrt(
        group_stats.T.xs("count", level=1).T
    )

    return mean_ret, std_error_ret


def compute_mean_returns_spread(mean_returns, upper_quant, lower_quant, std_err=None):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.

    Parameters
    ----------
    mean_returns : pd.DataFrame
        DataFrame of mean period wise returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.
    upper_quant : int
        Quantile of mean return from which we
        wish to subtract lower quantile mean return.
    lower_quant : int
        Quantile of mean return we wish to subtract
        from upper quantile mean return.
    std_err : pd.DataFrame, optional
        Period wise standard error in mean return by quantile.
        Takes the same form as mean_returns.

    Returns
    -------
    mean_return_difference : pd.Series
        Period wise difference in quantile returns.
    joint_std_err : pd.Series
        Period wise standard error of the difference in quantile returns.
        if std_err is None, this will be None
    """

    mean_return_difference = mean_returns.xs(
        upper_quant, level="factor_quantile"
    ) - mean_returns.xs(lower_quant, level="factor_quantile")

    if std_err is None:
        joint_std_err = None
    else:
        std1 = std_err.xs(upper_quant, level="factor_quantile")
        std2 = std_err.xs(lower_quant, level="factor_quantile")
        joint_std_err = np.sqrt(std1**2 + std2**2)

    return mean_return_difference, joint_std_err


def quantile_turnover(quantile_factor, quantile, period=1):
    """
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.

    Parameters
    ----------
    quantile_factor : pd.Series
        DataFrame with date, asset and factor quantile.
    quantile : int
        Quantile on which to perform turnover analysis.
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    # keep DateTimeIndex frequency information
    date_idx = quantile_factor.index.names.index("date")
    freq = quantile_factor.index.levels[date_idx].freq

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = (
        quant_names.groupby(level=["date"])
        .apply(lambda x: set(x.index.get_level_values("asset")))
        .asfreq(freq)
    )

    name_shifted = quant_name_sets.shift(periods=period)

    new_names = (quant_name_sets - name_shifted).dropna()

    def f(xs):
        return 0 if pd.isna(xs) else len(xs)

    def g(xs):
        return 1 if pd.isna(xs) else len(xs)

    quant_turnover = (new_names.apply(f) / quant_name_sets.apply(g)).rename(quantile)
    return quant_turnover


def factor_rank_autocorrelation(factor_data, period=1):
    """
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    autocorr : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.
    """
    # grouper = [factor_data.index.get_level_values('date')]

    date_idx = factor_data.index.names.index("date")
    freq = factor_data.index.levels[date_idx].freq

    asset_ranks_by_day = (
        factor_data.groupby(level="date")["factor"]
        .rank()
        .reset_index()
        .pivot(index="date", columns="asset", values="factor")
        .asfreq(freq)
    )

    # asset_factor_rank = ranks

    asset_shifted = asset_ranks_by_day.shift(period)

    return (
        asset_ranks_by_day.corrwith(asset_shifted, axis=1).rename(period).asfreq(freq)
    )
