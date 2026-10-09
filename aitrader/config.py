"""Instrument specs and global settings.

Prices for XAUUSD come from COMEX gold futures (GC=F) when no broker CSV is
supplied, so absolute levels differ from spot by the futures basis (usually a
few dollars). Load your own broker export with ``--csv`` for exact levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data_cache"
REPORT_DIR = ROOT / "reports"
JOURNAL_DIR = ROOT / "journal"
MODEL_DIR = ROOT / "models"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    yahoo: str
    pip: float                # price value of one pip / point used in reports
    spread: float             # typical round-trip-relevant spread (price units)
    slippage: float           # per-side slippage assumption (price units)
    contract_size: float      # units per 1.0 lot
    digits: int
    has_volume: bool
    cot_code: str             # CFTC contract market code
    usd_side: int             # +1 if long instrument == short USD
    smt_peer: str             # correlated/inverse market for SMT divergence
    smt_inverse: bool         # True if the peer moves inversely
    news_currencies: tuple = field(default_factory=tuple)


INSTRUMENTS: dict[str, Instrument] = {
    "EURUSD": Instrument(
        symbol="EURUSD", yahoo="EURUSD=X", pip=0.0001, spread=0.00008,
        slippage=0.00002, contract_size=100_000, digits=5, has_volume=False,
        cot_code="099741", usd_side=+1, smt_peer="DX-Y.NYB", smt_inverse=True,
        news_currencies=("USD", "EUR"),
    ),
    "XAUUSD": Instrument(
        symbol="XAUUSD", yahoo="GC=F", pip=0.1, spread=0.30,
        slippage=0.08, contract_size=100, digits=2, has_volume=True,
        cot_code="088691", usd_side=+1, smt_peer="SI=F", smt_inverse=False,
        news_currencies=("USD",),
    ),
}

# Daily intermarket series (Yahoo) and FRED macro series
INTERMARKET_YAHOO = {
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "silver": "SI=F",
    "vix": "^VIX",
    "spx": "^GSPC",
    "eurusd": "EURUSD=X",
    "gold": "GC=F",
}
FRED_SERIES = {
    "real10y": "DFII10",     # 10y TIPS real yield
    "us2y": "DGS2",
    "breakeven10y": "T10YIE",
}


@dataclass
class StrategyConfig:
    # signal rules (strategic daily bias + tactical H1 desk agreement)
    bias_min: float = 0.5               # |daily bias| needed to trade at all
    comp_align: float = 0.05            # H1 desk composite must agree by this much
    min_agreement: float = 0.55         # used by the legacy desk-only signal
    meta_threshold: float = 0.52        # ML meta-label probability gate
    use_meta: bool = False              # meta-filter added no value out-of-sample
    swing_target_vol: float = 0.10      # annualised vol target for swing mode
    # risk
    risk_per_trade: float = 0.005       # 0.5% equity at risk per trade
    max_risk_per_trade: float = 0.01
    atr_stop_mult: float = 1.5
    max_stop_atr: float = 3.0
    min_stop_atr: float = 0.8
    rr_target: float = 2.5
    partial_at_r: float = 1.0           # take 50% and move stop to BE
    partial_fraction: float = 0.5
    time_stop_bars: int = 72
    dd_throttle: float = 0.06           # halve risk beyond this drawdown
    dd_halt: float = 0.12               # stop opening trades beyond this
    avoid_hours_utc: tuple = (21, 22, 23)   # rollover / thin liquidity
    cooldown_bars: int = 3
    # ML
    ml_horizon: int = 48
    wf_initial_frac: float = 0.4
    wf_step_bars: int = 750
    embargo_bars: int = 48


DEFAULT_CONFIG = StrategyConfig()
