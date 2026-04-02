# Created by Oliver Meihls

# Regime Gate for Kalshi Trading
#
# Lightweight regime-aware gating that reads cached regime state from
# SharedFarmState and decides whether scalp/hold entries are allowed.
#
# Regime data originates from the core Argus RegimeDetector (topics
# ``regimes.symbol`` / ``regimes.market``).  A bus bridge in runner.py
# mirrors those events into the Kalshi bus so they flow into the
# FarmDispatcher's single subscriber, which updates SharedFarmState.
#
# If regime data is unavailable (bridge down, standalone Kalshi mode),
# the fallback mode applies:
# - "conservative" (default): treat as VOL_SPIKE + LIQ_LOW
# - "permissive": allow all (effective disable)
#
# This module has NO direct bus subscriptions.  It reads regime state
# from SharedFarmState only, consistent with the shared-state farm
# architecture (no per-bot subscriptions).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import KalshiConfig
    from .shared_state import SharedFarmState


@dataclass
class GateResult:
    # Result of a regime gate check.
    allowed: bool
    reason: str
    qty_multiplier: float = 1.0
    max_hold_override_s: Optional[float] = None


class RegimeGate:
    # Regime-aware entry gate for scalp and hold strategies.
    #
    # All decisions are based on pre-cached regime strings in SharedFarmState.
    # No bus subscriptions are created here.

    def __init__(self, cfg: KalshiConfig, shared: SharedFarmState) -> None:
        self._cfg = cfg
        self._shared = shared
        # Gate counters for diagnostics
        self.counters: Dict[str, int] = {
            "regime_block_vol": 0,
            "regime_block_liquidity": 0,
            "regime_block_risk": 0,
            "regime_data_missing": 0,
            "regime_bridge_unavailable": 0,
            "regime_spike_microstructure_fail": 0,
            "regime_allowed": 0,
        }

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def _get_regime(self, asset: str) -> tuple:
        # Return (vol_regime, liq_regime, risk_regime) strings for asset.
        #
        # If missing, returns fallback values based on config.
        vol = self._shared.regime_vol.get(asset, "")
        liq = self._shared.regime_liq.get(asset, "")
        risk = self._shared.regime_risk
        last_update = self._shared.regime_last_update.get(asset, 0.0)

        if not vol or not liq or last_update == 0.0:
            self._count("regime_data_missing")
            if self._cfg.regime_fallback_mode == "conservative":
                return ("VOL_SPIKE", "LIQ_LOW", "UNKNOWN")
            else:
                return ("VOL_NORMAL", "LIQ_NORMAL", "NEUTRAL")

        return (vol, liq, risk)

    def gate_scalp(
        self,
        asset: str,
        spread_cents: int = 0,
        depth: int = 0,
        reprice_move_cents: int = 0,
        projected_net_edge_cents: int = 0,
    ) -> GateResult:
        # Check whether a scalp entry is allowed under current regime.
        #
        # Args:
        # asset: Asset symbol (e.g. "BTC").
        # spread_cents: Current orderbook spread in cents.
        # depth: Current best-level depth.
        # reprice_move_cents: Recent repricing impulse magnitude.
        # projected_net_edge_cents: Projected net profit after fees.
        #
        # Returns:
        # GateResult indicating whether scalp is allowed.
        if not self._cfg.enable_regime_gating:
            self._count("regime_allowed")
            return GateResult(allowed=True, reason="gating_disabled")

        vol, liq, risk = self._get_regime(asset)

        # LIQ_LOW / LIQ_DRIED: block all scalps
        if liq in ("LIQ_LOW", "LIQ_DRIED"):
            self._count("regime_block_liquidity")
            return GateResult(allowed=False, reason=f"regime_block_liquidity:{liq}")

        # VOL_SPIKE: strict microstructure checks
        if vol == "VOL_SPIKE":
            cfg = self._cfg
            fails = []
            if spread_cents > cfg.scalp_spike_max_spread_cents:
                fails.append(f"spread={spread_cents}>{cfg.scalp_spike_max_spread_cents}")
            if depth < cfg.scalp_spike_depth_min:
                fails.append(f"depth={depth}<{cfg.scalp_spike_depth_min}")
            if reprice_move_cents < cfg.scalp_spike_reprice_min:
                fails.append(f"reprice={reprice_move_cents}<{cfg.scalp_spike_reprice_min}")
            if projected_net_edge_cents < cfg.scalp_spike_min_edge_cents:
                fails.append(f"edge={projected_net_edge_cents}<{cfg.scalp_spike_min_edge_cents}")

            if fails:
                self._count("regime_spike_microstructure_fail")
                return GateResult(
                    allowed=False,
                    reason=f"regime_spike_microstructure_fail:{','.join(fails)}",
                )

            # Spike allowed but with reduced size and shorter hold
            self._count("regime_allowed")
            return GateResult(
                allowed=True,
                reason="spike_strict_pass",
                qty_multiplier=cfg.scalp_spike_qty_multiplier,
                max_hold_override_s=cfg.scalp_spike_max_hold_minutes * 60.0,
            )

        # VOL_NORMAL / VOL_HIGH / VOL_LOW: allow
        self._count("regime_allowed")
        return GateResult(allowed=True, reason="regime_ok")

    def gate_hold(
        self,
        asset: str,
        time_to_settle_s: float,
        edge: float,
    ) -> GateResult:
        # Check whether a hold entry is allowed under current regime.
        #
        # Args:
        # asset: Asset symbol (e.g. "BTC").
        # time_to_settle_s: Seconds until contract settlement.
        # edge: Current entry edge (fraction, e.g. 0.05 = 5%).
        #
        # Returns:
        # GateResult indicating whether hold is allowed.
        if not self._cfg.enable_regime_gating:
            self._count("regime_allowed")
            return GateResult(allowed=True, reason="gating_disabled")

        vol, liq, risk = self._get_regime(asset)

        # VOL_SPIKE: only near-expiry high-conviction entries
        if vol == "VOL_SPIKE":
            cfg = self._cfg
            if time_to_settle_s > cfg.hold_spike_entry_horizon_s:
                self._count("regime_block_vol")
                return GateResult(
                    allowed=False,
                    reason=f"regime_block_vol:spike_too_early:{time_to_settle_s:.0f}s>{cfg.hold_spike_entry_horizon_s:.0f}s",
                )
            if edge < cfg.hold_spike_min_edge:
                self._count("regime_block_vol")
                return GateResult(
                    allowed=False,
                    reason=f"regime_block_vol:spike_low_edge:{edge:.3f}<{cfg.hold_spike_min_edge}",
                )

        # LIQ_LOW: reduce exposure
        qty_mult = 1.0
        if liq in ("LIQ_LOW", "LIQ_DRIED"):
            qty_mult *= 0.5
            if risk == "RISK_OFF":
                qty_mult *= self._cfg.risk_off_qty_multiplier
            self._count("regime_allowed")
            return GateResult(
                allowed=True,
                reason=f"regime_reduced:{liq}:{risk}",
                qty_multiplier=qty_mult,
            )

        # RISK_OFF alone: reduce exposure
        if risk == "RISK_OFF":
            self._count("regime_allowed")
            return GateResult(
                allowed=True,
                reason="regime_risk_off",
                qty_multiplier=self._cfg.risk_off_qty_multiplier,
            )

        self._count("regime_allowed")
        return GateResult(allowed=True, reason="regime_ok")

    def get_diagnostics(self) -> Dict[str, object]:
        # Return gate counters and current regime snapshot for diagnostics.
        regime_snapshot = {}
        for asset in sorted(set(self._shared.regime_vol) | set(self._shared.regime_liq)):
            regime_snapshot[asset] = {
                "vol": self._shared.regime_vol.get(asset, "?"),
                "liq": self._shared.regime_liq.get(asset, "?"),
                "risk": self._shared.regime_risk,
            }
        return {
            "counters": dict(self.counters),
            "regime_state": regime_snapshot,
        }
