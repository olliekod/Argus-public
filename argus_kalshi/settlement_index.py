from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .paper_model import estimate_kalshi_taker_fee_usd


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _norm_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    return side if side in {"yes", "no"} else ""


@dataclass(frozen=True, slots=True)
class SettlementRecord:
    ticker: str
    side: str
    won: bool
    entry_price_cents: int
    exit_price_cents: int
    pnl_usd: float
    fees_usd: float
    settlement_method: str
    timestamp: float
    source: str


class SettlementIndex:
    """Lookup table for settlement outcomes and historical exit prices."""

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str], List[SettlementRecord]] = defaultdict(list)
        self._binary_outcomes: Dict[str, Optional[bool]] = {}

    def add_record(self, record: SettlementRecord) -> None:
        key = (record.ticker, record.side)
        self._records[key].append(record)
        self._binary_outcomes[record.ticker] = self.infer_binary_from_records(
            [r for (ticker, _), rows in self._records.items() if ticker == record.ticker for r in rows]
        )

    @classmethod
    def from_jsonl_files(cls, paths: List[str]) -> "SettlementIndex":
        idx = cls()
        by_ticker: Dict[str, List[SettlementRecord]] = defaultdict(list)
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") != "settlement":
                        continue
                    ticker = str(obj.get("ticker") or obj.get("market_ticker") or "").strip()
                    side = _norm_side(obj.get("side"))
                    if not ticker or not side:
                        continue
                    won = bool(obj.get("won"))
                    record = SettlementRecord(
                        ticker=ticker,
                        side=side,
                        won=won,
                        entry_price_cents=_as_int(
                            obj.get("entry_price_cents", obj.get("fill_price_cents", obj.get("entry_price")))
                        ),
                        exit_price_cents=_as_int(
                            obj.get(
                                "exit_price_cents",
                                obj.get("settlement_price_cents", obj.get("exit_price", 100 if won else 0)),
                            )
                        ),
                        pnl_usd=_as_float(obj.get("pnl_usd", obj.get("pnl"))),
                        fees_usd=_as_float(obj.get("fees_usd", obj.get("fee_usd", obj.get("fee")))),
                        settlement_method=str(obj.get("settlement_method") or obj.get("method") or "unknown"),
                        timestamp=_as_float(obj.get("timestamp")),
                        source=str(obj.get("source") or ""),
                    )
                    idx._records[(ticker, side)].append(record)
                    by_ticker[ticker].append(record)
        for ticker, rows in by_ticker.items():
            idx._binary_outcomes[ticker] = idx.infer_binary_from_records(rows)
        return idx

    def get_binary_outcome(self, ticker: str) -> Optional[bool]:
        return self._binary_outcomes.get(ticker)

    def get_records(self, ticker: str, side: str) -> List[SettlementRecord]:
        return list(self._records.get((ticker, _norm_side(side)), ()))

    def get_scalp_pnl(
        self,
        entry_price_cents: int,
        exit_price_cents: int,
        quantity_contracts: int,
        source: str = "mispricing_scalp",
    ) -> Tuple[float, float]:
        del source
        qty_centicx = max(0, int(quantity_contracts)) * 100
        gross = (int(exit_price_cents) - int(entry_price_cents)) * max(0, int(quantity_contracts)) * 0.01
        fee = estimate_kalshi_taker_fee_usd(int(entry_price_cents), qty_centicx)
        fee += estimate_kalshi_taker_fee_usd(int(exit_price_cents), qty_centicx)
        return gross, fee

    def get_hold_pnl(
        self,
        ticker: str,
        side: str,
        entry_price_cents: int,
        quantity_contracts: int,
    ) -> Optional[Tuple[float, float]]:
        outcome = self.get_binary_outcome(ticker)
        if outcome is None:
            return None
        side = _norm_side(side)
        qty = max(0, int(quantity_contracts))
        yes_won = outcome
        side_won = yes_won if side == "yes" else not yes_won
        gross = ((100 - int(entry_price_cents)) if side_won else -int(entry_price_cents)) * qty * 0.01
        fee = estimate_kalshi_taker_fee_usd(int(entry_price_cents), qty * 100)
        return gross, fee

    def tickers_with_outcomes(self) -> Set[str]:
        return {ticker for ticker, outcome in self._binary_outcomes.items() if outcome is not None}

    @staticmethod
    def infer_binary_from_records(records: List[SettlementRecord]) -> Optional[bool]:
        if not records:
            return None
        expiry_votes: List[bool] = []
        inferred_votes: List[bool] = []
        for record in records:
            implied_yes_won = (record.side == "yes" and record.won) or (record.side == "no" and not record.won)
            if record.settlement_method == "contract_expiry":
                expiry_votes.append(implied_yes_won)
            else:
                inferred_votes.append(implied_yes_won)
        if expiry_votes:
            counts = Counter(expiry_votes)
            return True if counts[True] >= counts[False] else False
        if inferred_votes:
            counts = Counter(inferred_votes)
            if counts[True] == counts[False]:
                return None
            return True if counts[True] > counts[False] else False
        return None

    def coverage_stats(self) -> Dict[str, Any]:
        unique_tickers = {ticker for ticker, _side in self._records.keys()}
        return {
            "unique_tickers": len(unique_tickers),
            "tickers_with_binary_outcomes": len(self.tickers_with_outcomes()),
            "records": sum(len(rows) for rows in self._records.values()),
        }
