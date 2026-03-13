"""
Greeks Engine
=============

**European** Black-Scholes model for calculating option Greeks.
Provides Delta, Gamma, Theta, Vega, Rho, and Probability of Profit.

.. note::

    This engine uses the **European** Black-Scholes model as an
    approximation for American-style equity options.  American options
    can theoretically be worth more than their European counterpart
    (early-exercise premium), so Greeks computed here may differ from
    exchange-provided values — especially for deep-in-the-money puts
    with high interest rates.

Now with:
- Optional GPU acceleration via ``gpu_engine``
- Internal IV solver (Brent's method) with illiquid-quote guard
- Dynamic risk-free rate helper (Treasury yield)
- Derived-vs-provider tagging on every ``Greeks`` result

Illiquid quote guard
--------------------
Before attempting IV inversion the engine checks:
1. ``bid > 0``  — a zero bid means no market
2. ``spread_pct <= max_spread_pct`` — wide spreads produce unreliable IVs
3. ``mid >= min_premium`` — near-zero premiums blow up the Newton step

If any check fails the IV solve is skipped and ``Greeks.source`` is set
to ``"failed_illiquid"``.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from scipy.stats import norm
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Illiquid-quote thresholds (conservative defaults)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_SPREAD_PCT = 0.50   # 50 % of mid
_DEFAULT_MIN_PREMIUM = 0.01     # $0.01 minimum mid-price


# ---------------------------------------------------------------------------
# Greeks result types
# ---------------------------------------------------------------------------

@dataclass
class Greeks:
    """Container for option Greeks.

    Attributes
    ----------
    source : str
        One of:
        - ``"provider"``  — IV came directly from the exchange/API.
        - ``"derived"``   — IV was solved internally via Brent inversion.
        - ``"failed_illiquid"`` — IV solve skipped due to illiquid quote.
        - ``"failed_solve"``    — IV solve did not converge.
        - ``"unknown"``   — source not yet classified (legacy path).

        Backtesters should treat ``"derived"`` conservatively (wider bands,
        flag in reporting).
    solver_converged : bool or None
        True if IV solver converged, False if it failed, None if solver
        was not used (provider IV or no solve attempted).
    quote_quality_score : float
        0.0-1.0 score reflecting input quote quality.  Based on spread
        width, bid presence, and premium level.  0.0 = unusable.
    """
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float   # Per 1% IV change
    rho: float
    source: str = "unknown"
    iv_used: Optional[float] = None  # IV that was actually used
    solver_converged: Optional[bool] = None
    quote_quality_score: float = 0.0


@dataclass
class SpreadGreeks:
    """Container for spread Greeks (net of short and long legs)."""
    net_delta: float
    net_gamma: float
    net_theta: float  # Per day (positive = time decay helps us)
    net_vega: float   # Per 1% IV change (negative = IV drop helps us)
    short_source: str = "unknown"
    long_source: str = "unknown"


# ---------------------------------------------------------------------------
# Dynamic risk-free rate helper
# ---------------------------------------------------------------------------

def fetch_risk_free_rate(fallback: float = 0.045) -> float:
    """Fetch the current 13-week US Treasury yield as a risk-free proxy.

    Falls back to *fallback* on any network / parse error so the engine
    is never blocked by an external service.
    """
    try:
        import urllib.request
        import json

        # FRED API — 13-week T-bill secondary market rate (DGS3MO is free)
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            "?series_id=DTB3&sort_order=desc&limit=5"
            "&file_type=json&api_key=DEMO_KEY"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        for obs in data.get("observations", []):
            val = obs.get("value", ".")
            if val != ".":
                rate = float(val) / 100.0  # FRED reports as percent
                if 0.0 < rate < 0.20:
                    logger.debug("Fetched T-bill rate: %.4f", rate)
                    return rate
    except Exception as exc:
        logger.debug("fetch_risk_free_rate failed, using fallback: %s", exc)
    return fallback


class GreeksEngine:
    """
    **European** Black-Scholes Greeks calculator.

    .. warning::

        This is a European approximation.  For American-style equity
        options the true Greeks can differ — especially for deep ITM
        puts.  The ``Greeks.source`` field tags every result so
        downstream consumers can apply conservative handling.

    Formulas use standard Black-Scholes model with:
    - S: Spot price
    - K: Strike price
    - T: Time to expiration (years)
    - r: Risk-free rate
    - sigma: Implied volatility
    """
    
    # Default risk-free rate (current ~4.5% as of 2026)
    DEFAULT_RISK_FREE_RATE = 0.045
    
    def __init__(
        self,
        risk_free_rate: float | None = None,
        *,
        auto_refresh_rate: bool = False,
        max_spread_pct: float = _DEFAULT_MAX_SPREAD_PCT,
        min_premium: float = _DEFAULT_MIN_PREMIUM,
    ):
        """
        Initialize Greeks engine.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4.5%).
                If *auto_refresh_rate* is True this is ignored and the
                rate is fetched from FRED on construction.
            auto_refresh_rate: If True, fetch the 13-week T-bill rate
                on startup.  Falls back to DEFAULT_RISK_FREE_RATE.
            max_spread_pct: Maximum bid-ask spread as fraction of mid
                before the quote is flagged illiquid (default 0.50).
            min_premium: Minimum mid-price (in $) to attempt IV solve
                (default 0.01).
        """
        if auto_refresh_rate:
            self.r = fetch_risk_free_rate(self.DEFAULT_RISK_FREE_RATE)
        else:
            self.r = risk_free_rate if risk_free_rate is not None else self.DEFAULT_RISK_FREE_RATE
        self._max_spread_pct = max_spread_pct
        self._min_premium = min_premium

        # Solver metrics counters
        self._solve_attempts = 0
        self._solve_successes = 0
        self._solve_failures = 0
        self._illiquid_rejections = 0

        logger.debug("Greeks Engine initialized with r=%.4f (european approximation)", self.r)
    
    @staticmethod
    def compute_quote_quality_score(
        bid: float, ask: float,
        *, max_spread_pct: float = 0.50, min_premium: float = 0.01,
    ) -> float:
        """Compute a 0.0-1.0 quote quality score.

        Scoring:
        - 0.0 if bid <= 0 (no market)
        - 0.0 if mid < min_premium (near-zero premium)
        - Penalized by spread width (wider = lower score)
        - 1.0 for a tight, liquid quote
        """
        if bid <= 0:
            return 0.0
        mid = (bid + ask) / 2.0
        if mid < min_premium:
            return 0.0
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else 1.0
        if spread_pct > max_spread_pct:
            return round(max(0.0, 0.3 * (1.0 - spread_pct / max_spread_pct)), 4)
        # Linear scale: 0% spread → 1.0, max_spread_pct → 0.5
        return round(max(0.0, 1.0 - 0.5 * (spread_pct / max_spread_pct)), 4)

    def solver_metrics(self) -> Dict[str, Any]:
        """Return solver performance metrics."""
        total = self._solve_attempts
        return {
            "solve_attempts": total,
            "solve_successes": self._solve_successes,
            "solve_failures": self._solve_failures,
            "illiquid_rejections": self._illiquid_rejections,
            "success_rate": round(self._solve_successes / max(1, total), 4),
            "failure_rate": round(self._solve_failures / max(1, total), 4),
            "illiquid_rate": round(self._illiquid_rejections / max(1, total), 4),
        }

    def _d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return self._d1(S, K, T, sigma) - sigma * math.sqrt(T)

    # ------------------------------------------------------------------
    # European BS price (for IV inversion)
    # ------------------------------------------------------------------

    def _bs_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: Literal["call", "put"],
    ) -> float:
        """European Black-Scholes theoretical price."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(0.0, (S - K) if option_type == "call" else (K - S))
        d1 = self._d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-self.r * T) * norm.cdf(d2)
        return K * math.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # ------------------------------------------------------------------
    # Illiquid-quote guard
    # ------------------------------------------------------------------

    def is_quote_liquid(
        self,
        bid: float,
        ask: float,
        *,
        max_spread_pct: float | None = None,
        min_premium: float | None = None,
    ) -> Tuple[bool, str]:
        """Check whether a bid/ask quote is liquid enough for IV solving.

        Returns ``(is_liquid, reason)`` where *reason* is empty on success.
        """
        msp = max_spread_pct if max_spread_pct is not None else self._max_spread_pct
        mp = min_premium if min_premium is not None else self._min_premium

        if bid <= 0:
            return False, "zero_bid"
        mid = (bid + ask) / 2.0
        if mid < mp:
            return False, f"mid_below_min({mid:.4f}<{mp})"
        spread = ask - bid
        if mid > 0 and (spread / mid) > msp:
            return False, f"spread_too_wide({spread / mid:.2%}>{msp:.0%})"
        return True, ""

    # ------------------------------------------------------------------
    # IV solver (Brent's method)
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: Literal["call", "put"] = "put",
        *,
        bid: float | None = None,
        ask: float | None = None,
        sigma_low: float = 0.001,
        sigma_high: float = 5.0,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> Tuple[Optional[float], str]:
        """Solve for implied volatility via Brent's method.

        If *bid* and *ask* are provided the illiquid guard is applied
        first.  If the quote fails the guard, ``(None, "failed_illiquid")``
        is returned immediately.

        Returns
        -------
        (iv, source) where source is one of ``"derived"``,
        ``"failed_illiquid"``, or ``"failed_solve"``.
        """
        self._solve_attempts += 1

        # Guard: illiquid quotes
        if bid is not None and ask is not None:
            ok, reason = self.is_quote_liquid(bid, ask)
            if not ok:
                self._illiquid_rejections += 1
                logger.debug(
                    "IV solve skipped (illiquid): %s  S=%.2f K=%.2f T=%.4f",
                    reason, S, K, T,
                )
                return None, "failed_illiquid"

        if T <= 0 or S <= 0 or market_price <= 0:
            self._solve_failures += 1
            return None, "failed_solve"

        # Intrinsic check
        intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
        if market_price < intrinsic - 1e-10:
            self._solve_failures += 1
            return None, "failed_solve"

        def objective(sigma: float) -> float:
            return self._bs_price(S, K, T, sigma, option_type) - market_price

        try:
            iv = brentq(objective, sigma_low, sigma_high, xtol=tol, maxiter=max_iter)
            self._solve_successes += 1
            return iv, "derived"
        except (ValueError, RuntimeError):
            self._solve_failures += 1
            return None, "failed_solve"

    # ------------------------------------------------------------------
    # All-in-one: Greeks from raw quote (provider or derived)
    # ------------------------------------------------------------------

    def greeks_from_quote(
        self,
        S: float,
        K: float,
        T: float,
        option_type: Literal["call", "put"] = "put",
        *,
        provider_iv: float | None = None,
        bid: float | None = None,
        ask: float | None = None,
    ) -> Greeks:
        """Compute Greeks using provider IV first, falling back to internal IV.

        Priority:
        1. If *provider_iv* is supplied and > 0, use it (``source="provider"``).
        2. Else if *bid*/*ask* are supplied, compute mid and attempt IV
           solve (``source="derived"`` or ``"failed_*"``).
        3. Else return zero Greeks with ``source="unknown"``.

        This is the **recommended entry point** for the backtester.
        """
        # Compute quote quality score if bid/ask available
        quality = 0.0
        if bid is not None and ask is not None:
            quality = self.compute_quote_quality_score(bid, ask)

        # Path 1: provider-supplied IV
        if provider_iv is not None and provider_iv > 0:
            greeks = self.calculate_all_greeks(S, K, T, provider_iv, option_type)
            greeks.source = "provider"
            greeks.iv_used = provider_iv
            greeks.solver_converged = None  # solver not used
            greeks.quote_quality_score = quality
            return greeks

        # Path 2: derive IV from market mid
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
            iv, source = self.implied_volatility(
                mid, S, K, T, option_type, bid=bid, ask=ask
            )
            if iv is not None:
                greeks = self.calculate_all_greeks(S, K, T, iv, option_type)
                greeks.source = source   # "derived"
                greeks.iv_used = iv
                greeks.solver_converged = True
                greeks.quote_quality_score = quality
                return greeks
            # Failed solve — return zero Greeks with failure tag
            return Greeks(
                delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
                source=source, iv_used=None,
                solver_converged=False,
                quote_quality_score=quality,
            )

        # Path 3: nothing usable
        return Greeks(
            delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0,
            source="unknown", iv_used=None,
            solver_converged=None,
            quote_quality_score=0.0,
        )

    def calculate_delta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> float:
        """
        Calculate option delta.
        
        Delta measures the rate of change of option price with respect
        to changes in the underlying price.
        
        For puts: Delta is negative (price goes up, put value goes down).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Delta value (-1 to 0 for puts, 0 to 1 for calls)
        """
        if T <= 0:
            # At expiration
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1 = self._d1(S, K, T, sigma)
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1  # Put delta is N(d1) - 1
    
    def calculate_gamma(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float
    ) -> float:
        """
        Calculate option gamma.
        
        Gamma measures the rate of change of delta. Same for calls and puts.
        
        Returns:
            Gamma value (always positive)
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    def calculate_theta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> float:
        """
        Calculate option theta (per day).
        
        Theta measures time decay. Negative means option loses value over time.
        
        Returns:
            Theta per day (negative for long options)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        # First term (same for calls and puts)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        
        if option_type == 'call':
            theta_annual = term1 - self.r * K * math.exp(-self.r * T) * norm.cdf(d2)
        else:
            theta_annual = term1 + self.r * K * math.exp(-self.r * T) * norm.cdf(-d2)
        
        # Convert to per-day
        return theta_annual / 365
    
    def calculate_vega(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float
    ) -> float:
        """
        Calculate option vega (per 1% IV change).
        
        Vega measures sensitivity to volatility changes.
        Same for calls and puts.
        
        Returns:
            Vega per 1% IV change
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, sigma)
        vega_full = S * norm.pdf(d1) * math.sqrt(T)
        
        # Convert to per 1% change
        return vega_full / 100
    
    def calculate_rho(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> float:
        """
        Calculate option rho (sensitivity to interest rates).
        
        Returns:
            Rho per 1% rate change
        """
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, sigma)
        
        if option_type == 'call':
            return K * T * math.exp(-self.r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * math.exp(-self.r * T) * norm.cdf(-d2) / 100
    
    def calculate_all_greeks(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: Literal['call', 'put'] = 'put'
    ) -> Greeks:
        """
        Calculate all Greeks for an option.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (decimal, e.g., 0.40 for 40%)
            option_type: 'call' or 'put'
            
        Returns:
            Greeks dataclass with all values
        """
        return Greeks(
            delta=self.calculate_delta(S, K, T, sigma, option_type),
            gamma=self.calculate_gamma(S, K, T, sigma),
            theta=self.calculate_theta(S, K, T, sigma, option_type),
            vega=self.calculate_vega(S, K, T, sigma),
            rho=self.calculate_rho(S, K, T, sigma, option_type),
            source="unknown",  # caller should override via greeks_from_quote
            iv_used=sigma,
        )
    
    def calculate_spread_greeks(
        self,
        S: float,
        short_strike: float,
        long_strike: float,
        T: float,
        short_iv: float,
        long_iv: float = None,
    ) -> SpreadGreeks:
        """
        Calculate net Greeks for a put credit spread.
        
        For a put credit spread:
        - We SELL the higher strike put (short)
        - We BUY the lower strike put (long)
        
        Net Greeks = Short Greeks (inverted sign) + Long Greeks
        
        Args:
            S: Spot price
            short_strike: Strike we sell (higher)
            long_strike: Strike we buy (lower)
            T: Time to expiration in years
            short_iv: IV of short put
            long_iv: IV of long put (defaults to short_iv)
            
        Returns:
            SpreadGreeks with net values
        """
        if long_iv is None:
            long_iv = short_iv
        
        # Short put (we sold it, so invert the sign)
        short_greeks = self.calculate_all_greeks(S, short_strike, T, short_iv, 'put')
        
        # Long put (we own it)
        long_greeks = self.calculate_all_greeks(S, long_strike, T, long_iv, 'put')
        
        # Net = -Short + Long (because we're short the first put)
        return SpreadGreeks(
            net_delta=-short_greeks.delta + long_greeks.delta,
            net_gamma=-short_greeks.gamma + long_greeks.gamma,
            net_theta=-short_greeks.theta + long_greeks.theta,  # Positive = good for us
            net_vega=-short_greeks.vega + long_greeks.vega,     # Negative = IV drop helps
            short_source=short_greeks.source,
            long_source=long_greeks.source,
        )
    
    def probability_of_profit(
        self,
        S: float,
        short_strike: float,
        credit: float,
        T: float,
        sigma: float,
        long_strike: Optional[float] = None,
        use_gpu: bool = True,
        use_heston: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate probability of profit and touch.
        
        Args:
            S: Current spot price
            short_strike: Strike we sold
            credit: Net credit received
            T: Time to expiration in years
            sigma: Implied volatility
            long_strike: Strike we bought
            use_gpu: If True, use GPU
            use_heston: If True, use stochastic volatility
            
        Returns:
            Dict with 'pop' (0-100) and 'touch_stop' (0-100)
        """
        break_even = short_strike - credit
        
        if T <= 0:
            return {'pop': 100.0 if S > break_even else 0.0, 'touch_stop': 0.0}
        
        if use_gpu:
            try:
                from .gpu_engine import get_gpu_engine
                engine = get_gpu_engine()
                
                ls = long_strike if long_strike else short_strike - (S * 0.1) # Default 10% wide
                
                if use_heston:
                    return engine.monte_carlo_pop_heston(
                        S=S, short_strike=short_strike, long_strike=ls,
                        credit=credit, T=T, v0=sigma**2
                    )
                else:
                    pop = engine.monte_carlo_pop(
                        S=S, short_strike=short_strike, long_strike=ls,
                        credit=credit, T=T, sigma=sigma
                    )
                    return {'pop': round(pop, 1), 'touch_stop': 0.0}
            except Exception as e:
                logger.debug(f"GPU unavailable, using analytical: {e}")
        
        # Fallback: Analytical
        d2 = self._d2(S, break_even, T, sigma)
        return {'pop': round(norm.cdf(d2) * 100, 1), 'touch_stop': 0.0}
    
    def expected_move(
        self,
        S: float,
        sigma: float,
        T: float,
    ) -> Tuple[float, float]:
        """
        Calculate expected move based on implied volatility.
        
        The "expected move" is the 1 standard deviation range.
        ~68% of outcomes are expected within this range.
        
        Args:
            S: Current spot price
            sigma: Implied volatility (annualized)
            T: Time to expiration in years
            
        Returns:
            Tuple of (low_price, high_price)
        """
        if T <= 0 or sigma <= 0:
            return (S, S)
        
        # Expected move = S * sigma * sqrt(T)
        move = S * sigma * math.sqrt(T)
        
        return (round(S - move, 2), round(S + move, 2))
    
    @staticmethod
    def dte_to_years(dte: int) -> float:
        """Convert days to expiration to years."""
        return dte / 365


# Test function
def test_greeks():
    """Test the Greeks engine."""
    engine = GreeksEngine()
    
    print("=" * 60)
    print("GREEKS ENGINE TEST (European approximation)")
    print("=" * 60)
    
    # Example: IBIT at $42, selling $38 put, 10 DTE, 50% IV
    S = 42.0
    K = 38.0
    T = 10 / 365
    sigma = 0.50
    
    print(f"\nExample: IBIT ${S}, ${K} Put, 10 DTE, {sigma*100:.0f}% IV")
    
    greeks = engine.calculate_all_greeks(S, K, T, sigma, 'put')
    print(f"\nSingle Put Greeks (source={greeks.source}):")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.4f}")
    print(f"  Theta: ${greeks.theta:.4f}/day")
    print(f"  Vega:  ${greeks.vega:.4f}/1% IV")
    
    # Spread: Sell $38, Buy $36
    short_strike = 38.0
    long_strike = 36.0
    spread_greeks = engine.calculate_spread_greeks(S, short_strike, long_strike, T, sigma)
    
    print(f"\nPut Spread Greeks (Sell ${short_strike}, Buy ${long_strike}):")
    print(f"  Net Delta: {spread_greeks.net_delta:.4f}")
    print(f"  Net Gamma: {spread_greeks.net_gamma:.4f}")
    print(f"  Net Theta: ${spread_greeks.net_theta:.4f}/day (+ is good)")
    print(f"  Net Vega:  ${spread_greeks.net_vega:.4f}/1% IV (- is good)")
    
    # PoP
    credit = 0.50
    pop = engine.probability_of_profit(S, short_strike, credit, T, sigma)
    print(f"\nProbability of Profit: {pop['pop']:.1f}%")
    
    # Expected move
    low, high = engine.expected_move(S, sigma, T)
    print(f"Expected Move (1 SD): ${low:.2f} - ${high:.2f}")

    # IV solve demo
    print(f"\n--- IV Solver Demo ---")
    # Price a put, then back-solve for IV
    bs_price = engine._bs_price(S, K, T, sigma, "put")
    iv_solved, src = engine.implied_volatility(bs_price, S, K, T, "put")
    print(f"  BS price={bs_price:.4f}  solved IV={iv_solved:.6f}  source={src}")

    # Illiquid guard demo
    g = engine.greeks_from_quote(S, K, T, "put", bid=0.0, ask=0.05)
    print(f"  Illiquid quote (bid=0): source={g.source}")

    g2 = engine.greeks_from_quote(S, K, T, "put", provider_iv=0.50)
    print(f"  Provider IV: source={g2.source}, iv_used={g2.iv_used}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_greeks()
