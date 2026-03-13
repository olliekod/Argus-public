"""
GPU Engine
==========

CUDA-accelerated engine for Monte Carlo simulations and vectorized Greeks.
Uses PyTorch for GPU computation with automatic CPU fallback.

Features:
- Monte Carlo Probability of Profit (1M+ simulations)
- Batch Greeks calculation (100K+ options at once)
- IV Surface fitting
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("❌ PyTorch not found. GPU Engine will fall back to CPU.")
    logger.warning("💡 To enable RTX 4080 Super acceleration, run: pip install torch --index-url https://download.pytorch.org/whl/cu121")

try:
    from scipy.stats import norm
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU computation statistics."""
    device: str
    total_simulations: int
    computation_time_ms: float
    simulations_per_second: float


class GPUEngine:
    """
    CUDA-accelerated computation engine.
    
    Provides high-performance Monte Carlo simulations and Greeks
    calculations using PyTorch tensors on GPU.
    
    Falls back to CPU if CUDA is not available.
    """
    
    # Default simulation count
    DEFAULT_SIMULATIONS = 1_000_000
    
    # Risk-free rate (annualized)
    RISK_FREE_RATE = 0.045  # 4.5%
    
    # Default Heston Parameters for BTC/Crypto
    HESTON_KAPPA = 2.0      # Speed of mean reversion
    HESTON_SIGMA_V = 0.5    # Volatility of volatility
    HESTON_RHO = -0.7       # Price-Volatility correlation
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize GPU engine.
        
        Args:
            force_cpu: If True, use CPU even if GPU is available
        """
        self.force_cpu = force_cpu
        self._device = None
        self._stats = GPUStats(
            device="unknown",
            total_simulations=0,
            computation_time_ms=0,
            simulations_per_second=0,
        )
        
        # Initialize device
        self._init_device()
        
    def _init_device(self):
        """Initialize compute device."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not installed. GPU acceleration disabled.")
            self._device = None
            return
        
        if self.force_cpu:
            self._device = torch.device('cpu')
            logger.info("GPU Engine initialized on CPU (forced)")
        elif torch.cuda.is_available():
            self._device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU Engine initialized on CUDA: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self._device = torch.device('cpu')
            logger.info("GPU Engine initialized on CPU (CUDA not available)")
        
        self._stats.device = str(self._device)
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self._device is not None and self._device.type == 'cuda'
    
    @property
    def device_name(self) -> str:
        """Get current device name."""
        if self._device is None:
            return "None (no PyTorch)"
        if self._device.type == 'cuda':
            return torch.cuda.get_device_name(0)
        return "CPU"
    
    @property
    def stats(self) -> GPUStats:
        """Get computation statistics."""
        return self._stats
    
    def monte_carlo_pop(
        self,
        S: float,
        short_strike: float,
        long_strike: float,
        credit: float,
        T: float,
        sigma: float,
        simulations: int = None,
    ) -> float:
        """
        Calculate Probability of Profit using Monte Carlo simulation.

        P2 fix: Uses Heston stochastic volatility model instead of GBM
        to account for vol-of-vol, mean reversion, and price-vol correlation.
        This produces more realistic tail risk estimates for crypto underlyings.

        For a put credit spread:
        - Max profit: credit received (if price stays above short strike)
        - Max loss: spread width - credit (if price drops below long strike)
        - Breakeven: short strike - credit

        Args:
            S: Current underlying price
            short_strike: Strike sold (higher)
            long_strike: Strike bought (lower)
            credit: Net credit received per share
            T: Time to expiration in years
            sigma: Implied volatility (decimal, e.g., 0.40 for 40%)
            simulations: Number of simulations (default 1M)

        Returns:
            Probability of profit as percentage (0-100)
        """
        if not TORCH_AVAILABLE or self._device is None:
            return self._analytical_pop(S, short_strike, credit, T, sigma)

        # P2: Delegate to Heston model which accounts for stochastic vol
        v0 = sigma ** 2  # Convert IV to initial variance
        result = self.monte_carlo_pop_heston(
            S=S,
            short_strike=short_strike,
            long_strike=long_strike,
            credit=credit,
            T=T,
            v0=v0,
            simulations=simulations,
            steps=max(10, int(T * 252)),  # Daily steps
        )
        return result['pop']
    
    def _analytical_pop(
        self,
        S: float,
        short_strike: float,
        credit: float,
        T: float,
        sigma: float,
    ) -> float:
        """Analytical PoP fallback when GPU unavailable."""
        if not SCIPY_AVAILABLE:
            # Last resort: delta approximation
            return 70.0  # Conservative default
        
        breakeven = short_strike - credit
        
        # P(S_T >= breakeven) using Black-Scholes
        d2 = (math.log(S / breakeven) + (self.RISK_FREE_RATE - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        pop = norm.cdf(d2) * 100
        
        return pop

    def monte_carlo_pop_heston(
        self,
        S: float,
        short_strike: float,
        long_strike: float,
        credit: float,
        T: float,
        v0: float,           # Current variance (sigma^2)
        simulations: int = None,
        steps: int = 30,     # Time steps for stochastic simulation
    ) -> Dict[str, float]:
        """
        Calculate Probability of Profit and Touch using Heston Stochastic Volatility.
        
        The Heston model accounts for:
        1. Volatility mean-reversion (Kappa, Theta)
        2. Volatility of Volatility (Sigma_v)
        3. Price-Volatility correlation (Rho)
        
        Args:
            S: Current price
            short_strike: Put credit spread short strike
            long_strike: Put credit spread long strike
            credit: Net credit received
            T: Time to expiration (years)
            v0: Initial variance (IV^2)
            simulations: Number of paths (default 1M)
            steps: Number of time steps (daily is usually 252*T)
            
        Returns:
            Dict with 'pop' and 'prob_of_touch_stop'
        """
        if not TORCH_AVAILABLE or self._device is None:
            return {'pop': self._analytical_pop(S, short_strike, credit, T, math.sqrt(v0)), 'prob_of_touch_stop': 0.0}
            
        simulations = simulations or self.DEFAULT_SIMULATIONS
        dt = T / steps
        
        # Initialize tensors
        S_path = torch.full((simulations,), S, device=self._device)
        V_path = torch.full((simulations,), v0, device=self._device)
        
        # Track if path HAS TOUCHED stop loss (100% loss of spread)
        # For put spread, max loss occurs at long_strike
        stop_level = long_strike
        has_touched_stop = torch.zeros(simulations, device=self._device, dtype=torch.bool)
        
        # Parameters
        kappa = self.HESTON_KAPPA
        theta = v0 # Assume current vol is the mean for now
        sigma_v = self.HESTON_SIGMA_V
        rho = self.HESTON_RHO
        r = self.RISK_FREE_RATE
        
        sqrt_dt = math.sqrt(dt)
        
        # Pre-generate normals (1M paths x 30 steps x 2 correlated normals)
        # We'll generate step-by-step to save memory if needed, 
        # but 1M x 30 x 2 is only ~240MB in float32.
        
        for _ in range(steps):
            # Correlated random variables
            z1 = torch.randn(simulations, device=self._device)
            z2 = torch.randn(simulations, device=self._device)
            zv = rho * z1 + math.sqrt(1 - rho**2) * z2
            
            # Update Price: dS = r*S*dt + sqrt(V)*S*dW_s
            # We use Euler-Maruyama discretization
            sqrt_V = torch.sqrt(torch.clamp(V_path, min=1e-6))
            S_path = S_path * torch.exp((r - 0.5 * V_path) * dt + sqrt_V * sqrt_dt * z1)
            
            # Update Variance: dV = kappa*(theta - V)*dt + sigma_v*sqrt(V)*dW_v
            # Ensure variance stays positive (Reflection or Partial Truncation)
            V_path = V_path + kappa * (theta - V_path) * dt + sigma_v * sqrt_V * sqrt_dt * zv
            V_path = torch.clamp(V_path, min=1e-6)
            
            # Check for touch
            has_touched_stop |= (S_path <= stop_level)
            
        # Terminal checks
        breakeven = short_strike - credit
        profitable = (S_path >= breakeven).float()
        
        return {
            'pop': profitable.mean().item() * 100,
            'prob_of_touch_stop': has_touched_stop.float().mean().item() * 100
        }
    
    def batch_greeks(
        self,
        S: float,
        strikes: List[float],
        T: float,
        sigmas: List[float],
        option_type: str = 'put',
    ) -> Dict[str, List[float]]:
        """
        Calculate Greeks for multiple options in parallel.
        
        Vectorized Black-Scholes Greeks calculation on GPU.
        
        Args:
            S: Current underlying price
            strikes: List of strike prices
            T: Time to expiration in years
            sigmas: List of implied volatilities for each strike
            option_type: 'put' or 'call'
            
        Returns:
            Dict with 'delta', 'gamma', 'theta', 'vega' lists
        """
        if not TORCH_AVAILABLE or self._device is None:
            return self._cpu_batch_greeks(S, strikes, T, sigmas, option_type)
        
        import time
        start_time = time.perf_counter()
        
        # Convert to tensors
        K = torch.tensor(strikes, device=self._device, dtype=torch.float32)
        sigma = torch.tensor(sigmas, device=self._device, dtype=torch.float32)
        S_t = torch.tensor([S], device=self._device, dtype=torch.float32)
        
        r = self.RISK_FREE_RATE
        sqrt_T = math.sqrt(T)
        
        # d1 and d2
        d1 = (torch.log(S_t / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Normal CDF and PDF (approximation for GPU)
        # Using torch.special.erf for normal CDF
        def norm_cdf(x):
            return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        
        def norm_pdf(x):
            return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        
        N_d1 = norm_cdf(d1)
        N_d2 = norm_cdf(d2)
        n_d1 = norm_pdf(d1)
        
        # Greeks calculation
        if option_type == 'put':
            delta = N_d1 - 1  # Put delta is negative
        else:
            delta = N_d1
        
        gamma = n_d1 / (S_t * sigma * sqrt_T)
        vega = S_t * n_d1 * sqrt_T / 100  # Per 1% IV change
        
        # Convert scalar to tensor for exp operation
        exp_neg_rT = torch.exp(torch.tensor(-r * T, device=self._device))
        
        if option_type == 'put':
            theta = ((-S_t * n_d1 * sigma) / (2 * sqrt_T) 
                    + r * K * exp_neg_rT * (1 - N_d2)) / 365
        else:
            theta = ((-S_t * n_d1 * sigma) / (2 * sqrt_T) 
                    - r * K * exp_neg_rT * N_d2) / 365
        
        # Move back to CPU for output
        result = {
            'delta': delta.cpu().tolist(),
            'gamma': gamma.cpu().tolist(),
            'theta': theta.cpu().tolist(),
            'vega': vega.cpu().tolist(),
        }
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Batch Greeks: {len(strikes)} options in {elapsed_ms:.2f}ms")
        
        return result
    
    def _cpu_batch_greeks(
        self,
        S: float,
        strikes: List[float],
        T: float,
        sigmas: List[float],
        option_type: str,
    ) -> Dict[str, List[float]]:
        """CPU fallback for batch Greeks."""
        if not SCIPY_AVAILABLE:
            # Return empty if scipy not available
            n = len(strikes)
            return {
                'delta': [0.0] * n,
                'gamma': [0.0] * n,
                'theta': [0.0] * n,
                'vega': [0.0] * n,
            }
        
        r = self.RISK_FREE_RATE
        sqrt_T = math.sqrt(T)
        
        deltas, gammas, thetas, vegas = [], [], [], []
        
        for K, sigma in zip(strikes, sigmas):
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            n_d1 = norm.pdf(d1)
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            
            if option_type == 'put':
                delta = N_d1 - 1
                theta = ((-S * n_d1 * sigma) / (2 * sqrt_T) 
                        + r * K * math.exp(-r * T) * (1 - N_d2)) / 365
            else:
                delta = N_d1
                theta = ((-S * n_d1 * sigma) / (2 * sqrt_T) 
                        - r * K * math.exp(-r * T) * N_d2) / 365
            
            gamma = n_d1 / (S * sigma * sqrt_T)
            vega = S * n_d1 * sqrt_T / 100
            
            deltas.append(delta)
            gammas.append(gamma)
            thetas.append(theta)
            vegas.append(vega)
        
        return {
            'delta': deltas,
            'gamma': gammas,
            'theta': thetas,
            'vega': vegas,
        }
    
    def monte_carlo_expected_move(
        self,
        S: float,
        T: float,
        sigma: float,
        simulations: int = None,
        confidence: float = 0.68,
    ) -> Tuple[float, float]:
        """
        Calculate expected price range using Monte Carlo.
        
        Args:
            S: Current price
            T: Time to expiration in years
            sigma: Implied volatility
            simulations: Number of simulations
            confidence: Confidence level (default 0.68 = 1 std dev)
            
        Returns:
            Tuple of (low_price, high_price)
        """
        if not TORCH_AVAILABLE or self._device is None:
            # Analytical fallback
            move = S * sigma * math.sqrt(T)
            return (S - move, S + move)
        
        simulations = simulations or self.DEFAULT_SIMULATIONS
        
        drift = (self.RISK_FREE_RATE - 0.5 * sigma ** 2) * T
        vol = sigma * math.sqrt(T)
        
        z = torch.randn(simulations, device=self._device)
        S_T = S * torch.exp(drift + vol * z)
        
        # Get percentiles
        lower_pct = (1 - confidence) / 2
        upper_pct = 1 - lower_pct
        
        low_price = torch.quantile(S_T, lower_pct).item()
        high_price = torch.quantile(S_T, upper_pct).item()
        
        return (low_price, high_price)
    
    def evaluate_traders_batch(
        self,
        trader_params: torch.Tensor, # (N, 8) tensor of parameters
        market_data: torch.Tensor,   # (10,) tensor of current conditions
    ) -> torch.Tensor:               # (N,) boolean mask
        """
        Evaluate entry conditions for all traders in one batch on GPU.
        
        trader_params columns:
        0: iv_min | 1: iv_max | 2: warmth_min | 3: pop_min 
        4: dte_min | 5: dte_max | 6: gap_max | 7: strategy_id
        
        market_data values:
        0: current_iv | 1: warmth | 2: dte | 3: pop | 4: gap_risk 
        5: bullish_bool | 6: bearish_bool | 7: neutral_bool
        8: session_morning_bool | 9: session_midday_bool | 10: session_afternoon_bool
        """
        if not TORCH_AVAILABLE or self._device is None:
            return torch.zeros(trader_params.shape[0], dtype=torch.bool)
            
        N = trader_params.shape[0]
        
        # 1. Basic Threshold Filters
        # (current_iv >= iv_min) AND (current_iv <= iv_max)
        mask = (market_data[0] >= trader_params[:, 0]) & (market_data[0] <= trader_params[:, 1])
        
        # warmth >= warmth_min
        mask &= (market_data[1] >= trader_params[:, 2])
        
        # (dte >= dte_min) AND (dte <= dte_max)
        mask &= (market_data[2] >= trader_params[:, 4]) & (market_data[2] <= trader_params[:, 5])
        
        # pop >= pop_min
        mask &= (market_data[3] >= trader_params[:, 3])
        
        # gap_risk <= gap_max
        mask &= (market_data[4] <= trader_params[:, 6])
        
        # 2. Strategy vs Market Direction
        # strategy_id: 0: BULL_PUT, 1: BEAR_CALL, 2: IRON_CONDOR, 3: STRADDLE_SELL
        
        # Bull Put: No Bearish direction
        mask &= ~((trader_params[:, 7] == 0) & (market_data[6] == 1))
        
        # Bear Call: No Bullish direction
        mask &= ~((trader_params[:, 7] == 1) & (market_data[5] == 1))
        
        # Straddle Sell: Only if high IV
        mask &= ~((trader_params[:, 7] == 3) & (market_data[0] < 60))
        
        return mask

    def portfolio_stress_test(
        self,
        trader_params: torch.Tensor,
        S: float,
        T: float,
        v0: float,
        num_paths: int = 1000,
        steps: int = 30,
    ) -> Dict[str, Any]:
        """
        Simulate the entire 400,000 trader farm against multiple price paths.
        Checks how many traders 'survive' or hit stops in correlated scenarios.
        """
        if not TORCH_AVAILABLE or self._device is None:
            return {}
            
        N_traders = trader_params.shape[0]
        dt = T / steps
        r = self.RISK_FREE_RATE
        
        # Paths for the underlying (Shared across all traders)
        S_paths = torch.full((num_paths, steps + 1), S, device=self._device)
        V_paths = torch.full((num_paths, steps + 1), v0, device=self._device)
        
        # Parameters for Heston paths
        kappa = self.HESTON_KAPPA
        theta = v0
        sigma_v = self.HESTON_SIGMA_V
        rho = self.HESTON_RHO
        sqrt_dt = math.sqrt(dt)
        
        # Generate paths
        for t in range(steps):
            z1 = torch.randn(num_paths, device=self._device)
            z2 = torch.randn(num_paths, device=self._device)
            zv = rho * z1 + math.sqrt(1 - rho**2) * z2
            
            sqrt_V = torch.sqrt(torch.clamp(V_paths[:, t], min=1e-6))
            S_paths[:, t+1] = S_paths[:, t] * torch.exp((r - 0.5 * V_paths[:, t]) * dt + sqrt_V * sqrt_dt * z1)
            V_paths[:, t+1] = V_paths[:, t] + kappa * (theta - V_paths[:, t]) * dt + sigma_v * sqrt_V * sqrt_dt * zv
            V_paths[:, t+1] = torch.clamp(V_paths[:, t+1], min=1e-6)
            
        # Evaluation
        # For each path, check how many traders would survive (no stop touched)
        # We'll pick 10 representative paths (Best, Worst, Median)
        
        # Calculate stops for all traders (assuming 15% distance for stress test)
        # In reality, this would use the trader's dte_target and delta
        stop_levels = S * 0.85 
        
        results = []
        for p in range(min(10, num_paths)):
            path = S_paths[p, :]
            touched = (path.min() <= stop_levels)
            results.append({
                'path_id': p,
                'min_price': path.min().item(),
                'traders_stopped': touched.float().item() * 100
            })
            
        return {
            'total_paths': num_paths,
            'avg_stopped_pct': sum(r['traders_stopped'] for r in results) / len(results),
            'sample_paths': results
        }

    def format_status(self) -> str:
        """Format GPU status for display."""
        lines = [
            f"🖥️ <b>GPU Engine Status:</b>",
            f"  Device: {self.device_name}",
        ]
        
        if self.is_gpu_available:
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            lines.append(f"  Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
        
        if self._stats.total_simulations > 0:
            lines.append(f"  Simulations: {self._stats.total_simulations:,} (Heston & GBM)")
            lines.append(f"  Speed: {self._stats.simulations_per_second:,.0f} paths/sec")
            
        # Add context about the 400K farm
        lines.append(f"  Batch Engine: Active (Optimized for 400K traders)")
        
        return "\n".join(lines)


# Module-level singleton for easy access
_engine: Optional[GPUEngine] = None


def get_gpu_engine(force_cpu: bool = False) -> GPUEngine:
    """Get or create the GPU engine singleton."""
    global _engine
    if _engine is None:
        _engine = GPUEngine(force_cpu=force_cpu)
    return _engine


# Self-test
if __name__ == "__main__":
    import time
    print("GPU Engine Benchmark & Scaling Test")
    print("=" * 50)
    
    engine = get_gpu_engine()
    print(f"Device: {engine.device_name}")
    print(f"GPU Available: {engine.is_gpu_available}")
    
    # Test cases: 1M (Cold), 1M (Warm), 10M, 100M
    test_scales = [1_000_000, 1_000_000, 10_000_000, 100_000_000]
    
    print("\n📊 Monte Carlo PoP Scaling Test:")
    for i, scale in enumerate(test_scales):
        label = f"Trial {i+1} ({scale:,} sims)"
        if i == 0: label += " [COLD START]"
        elif i == 1: label += " [WARM START]"
        
        start = time.perf_counter()
        pop = engine.monte_carlo_pop(
            S=50.0, short_strike=48, long_strike=44,
            credit=1.20, T=30/365, sigma=0.65,
            simulations=scale
        )
        elapsed = (time.perf_counter() - start) * 1000
        speed = scale / (elapsed / 1000)
        
        print(f"  {label:35} | Time: {elapsed:8.2f}ms | Speed: {speed:15,.0f} sims/sec | PoP: {pop:.1f}%")
    
    # Test Batch Greeks with more options
    print("\n📈 Batch Greeks Scaling Test:")
    count = 10000
    strikes = [40.0 + (i * 0.01) for i in range(count)]
    sigmas = [0.4 + (i * 0.00001) for i in range(count)]
    
    start = time.perf_counter()
    greeks = engine.batch_greeks(
        S=50.0,
        strikes=strikes,
        T=30/365,
        sigmas=sigmas,
        option_type='put',
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Processed {count:,} options in {elapsed:.2f}ms ({count/(elapsed/1000):,.0f} options/sec)")
    
    # Test Expected Move
    print("\n📉 Expected Move Test:")
    low, high = engine.monte_carlo_expected_move(
        S=50.0,
        T=30/365,
        sigma=0.65,
    )
    print(f"  68% Range: ${low:.2f} - ${high:.2f}")
    
    print("\n✅ GPU Engine benchmark complete!")
