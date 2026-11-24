"""Contract selection from prior session observations and the exchange calendar."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol

import pandas as pd


def expiry_date(value: str) -> pd.Timestamp:
    day = pd.Timestamp(value)
    return day + pd.offsets.MonthEnd(0) if len(value) == 7 else day


@dataclass(frozen=True)
class RollContract:
    instrument_id: str
    contract_month: str
    expiry: pd.Timestamp


@dataclass(frozen=True)
class RollInputs:
    contracts: tuple[RollContract, ...]
    incumbent: str | None
    decision_sessions: tuple[str, ...]
    effective_session: str
    forward_only: bool
    observe: Callable[[str, str, str], float]
    sessions_to_expiry: Callable[[str, pd.Timestamp], int]


class RollPolicy(Protocol):
    """Pick a concrete contract. Observations are limited to decision sessions."""

    name: str

    def select(self, inputs: RollInputs) -> str: ...


@dataclass(frozen=True)
class _LiquidityRollPolicy:
    name: str
    field: str

    def select(self, inputs: RollInputs) -> str:
        winners = []
        effective = pd.Timestamp(inputs.effective_session)
        contracts = tuple(contract for contract in inputs.contracts
                          if self.field != 'volume' or contract.expiry > effective)
        by_id = {contract.instrument_id: contract for contract in contracts}
        forced_roll = inputs.incumbent is not None and inputs.incumbent not in by_id
        decision_sessions = inputs.decision_sessions[-1:] if forced_roll else inputs.decision_sessions
        for decision in decision_sessions:
            decision_date = pd.Timestamp(decision)
            scores = [
                (-inputs.observe(decision, contract.instrument_id, self.field),
                 contract.contract_month, contract.instrument_id)
                for contract in contracts if contract.expiry >= decision_date
            ]
            if not scores:
                raise ValueError(f'Missing {self.field}: Session={decision} has no unexpired contracts')
            winners.append(min(scores)[2])
        incumbent = inputs.incumbent or winners[0]
        candidate = winners[-1]
        if forced_roll:
            return candidate
        if len(set(winners)) == 1 and (not inputs.forward_only or
                by_id[candidate].contract_month > by_id[incumbent].contract_month):
            return candidate
        return incumbent


@dataclass(frozen=True)
class DaysBeforeExpiry:
    """Switch at the opening of the Nth trading session before expiry."""

    days: int = 5
    name = 'days_before_expiry'

    def __post_init__(self) -> None:
        if isinstance(self.days, bool) or not isinstance(self.days, Integral) or self.days < 0:
            raise ValueError('days_before_expiry must be a non-negative integer')

    def select(self, inputs: RollInputs) -> str:
        effective = pd.Timestamp(inputs.effective_session)
        contracts = sorted(inputs.contracts, key=lambda contract: (
            contract.expiry, contract.contract_month, contract.instrument_id))
        incumbent = next((contract for contract in contracts
                          if contract.instrument_id == inputs.incumbent), None)
        if incumbent is None:
            incumbent = next((contract for contract in contracts if contract.expiry >= effective), None)
        if incumbent is None:
            raise ValueError(f'No unexpired contracts for Session={inputs.effective_session}')
        for contract in contracts:
            if contract.expiry <= incumbent.expiry or contract.expiry < effective:
                continue
            if (incumbent.expiry < effective or
                    inputs.sessions_to_expiry(inputs.effective_session, incumbent.expiry) <= self.days):
                return contract.instrument_id
            break
        if incumbent.expiry < effective:
            raise ValueError(f'No successor for {incumbent.instrument_id} in Session={inputs.effective_session}')
        return incumbent.instrument_id


def resolve_roll_policy(rule: str | RollPolicy) -> RollPolicy:
    if isinstance(rule, str):
        policies = {
            'max_oi': _LiquidityRollPolicy('max_oi', 'open_interest'),
            'max_volume': _LiquidityRollPolicy('max_volume', 'volume'),
            'days_before_expiry': DaysBeforeExpiry(),
        }
        if rule not in policies:
            raise ValueError(f'rule must be one of {tuple(policies)} or a RollPolicy')
        return policies[rule]
    if (not isinstance(getattr(rule, 'name', None), str) or not rule.name.strip()
            or rule.name != rule.name.strip() or not callable(getattr(rule, 'select', None))):
        raise ValueError('RollPolicy requires a name and a select method')
    return rule
