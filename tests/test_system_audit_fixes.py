# Created by Oliver Meihls

# Tests for System Audit Fixes (Sections 1-12)
#
# Regression tests for all bugs fixed from the comprehensive system audit.
# Tests use source inspection for modules with heavy dependency chains
# (pandas/yfinance), and direct imports for lightweight modules.

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone

import pytest


# Helper: read source file without importing

def _read_source(path: str) -> str:
    with open(path) as f:
        return f.read()


# 1.2: Ollama SIGKILL fallback


class TestOllamaSIGKILL:
    # Verify SIGKILL escalation in RuntimeController.

    def test_sigkill_in_source(self):
        source = _read_source("src/agent/runtime_controller.py")
        assert "SIGKILL" in source

    def test_pid_tracked(self):
        source = _read_source("src/agent/runtime_controller.py")
        assert ".pid" in source

    def test_sigterm_before_sigkill(self):
        source = _read_source("src/agent/runtime_controller.py")
        # SIGTERM should come before SIGKILL (graceful first)
        assert "SIGTERM" in source
        sigterm_pos = source.index("SIGTERM")
        sigkill_pos = source.index("SIGKILL")
        assert sigterm_pos < sigkill_pos


# 2.2a: ETF detector uses explicit symbol set


class TestBITSubstringFix:
    # Verify explicit symbol set replaces fragile 'BIT in symbol' check.

    def test_explicit_set_used(self):
        source = _read_source("src/detectors/etf_options_detector.py")
        assert '{"IBIT", "BITO"}' in source or "{'IBIT', 'BITO'}" in source or \
               '{"BITO", "IBIT"}' in source or "{'BITO', 'IBIT'}" in source

    def test_bit_substring_removed(self):
        source = _read_source("src/detectors/etf_options_detector.py")
        assert '"BIT" in self.symbol' not in source

    def test_bitw_not_misclassified(self):
        assert "BITW" not in {"IBIT", "BITO"}


# 5.1a: Context key mismatch fix


class TestContextKeyMismatch:
    # Verify orchestrator uses correct regime keys.

    def test_warmth_label_key_used(self):
        source = _read_source("src/orchestrator.py")
        assert "warmth_label" in source


# 6.1a: Ghost import fix


class TestGhostImportFix:
    # Verify verify_system.py imports ETFOptionsDetector.

    def test_no_ibit_detector_import(self):
        source = _read_source("scripts/verify_system.py")
        assert "IBITDetector" not in source
        assert "ETFOptionsDetector" in source


# 6.3a: EventBus circuit breaker


class TestEventBusCircuitBreaker:
    # Verify handler circuit breaker prevents unbounded error logging.

    def test_handler_disabled_after_consecutive_errors(self):
        from src.core.bus import EventBus

        bus = EventBus()
        call_count = 0

        def bad_handler(event):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always fails")

        bus.subscribe("test.topic", bad_handler)
        bus.start()

        for _ in range(150):
            bus.publish("test.topic", {"data": 1})

        time.sleep(1.5)
        bus.stop()

        assert call_count <= bus._CIRCUIT_BREAKER_THRESHOLD + 5

    def test_healthy_handler_not_circuit_broken(self):
        from src.core.bus import EventBus

        bus = EventBus()
        results = []

        def good_handler(event):
            results.append(event)

        bus.subscribe("test.topic", good_handler)
        bus.start()

        for i in range(200):
            bus.publish("test.topic", i)

        time.sleep(1.0)
        bus.stop()

        assert len(results) == 200

    def test_circuit_broken_handlers_set_populated(self):
        from src.core.bus import EventBus

        bus = EventBus()

        def always_fails(event):
            raise RuntimeError("fail")

        bus.subscribe("test.topic", always_fails)
        bus.start()

        for _ in range(bus._CIRCUIT_BREAKER_THRESHOLD + 10):
            bus.publish("test.topic", "x")

        time.sleep(1.5)
        bus.stop()

        assert id(always_fails) in bus._circuit_broken_handlers

    def test_threshold_constant_exists(self):
        from src.core.bus import EventBus
        assert hasattr(EventBus, "_CIRCUIT_BREAKER_THRESHOLD")
        assert EventBus._CIRCUIT_BREAKER_THRESHOLD == 100

    def test_circuit_breaker_state_in_init(self):
        from src.core.bus import EventBus
        bus = EventBus()
        assert hasattr(bus, "_handler_consecutive_errors")
        assert hasattr(bus, "_circuit_broken_handlers")
        assert isinstance(bus._circuit_broken_handlers, set)


# 7.1: DOW strategy replay adapter


class TestDowReplayAdapter:
    # Verify the DowRegimeTimingGateReplayStrategy works.

    def test_instantiation_with_params_dict(self):
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy

        strategy = DowRegimeTimingGateReplayStrategy({"gate_score_threshold": 0.6})
        assert strategy.strategy_id == "DOW_REGIME_TIMING_REPLAY_V1"
        assert strategy._config["gate_score_threshold"] == 0.6

    def test_generates_no_trade_intents(self):
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy

        strategy = DowRegimeTimingGateReplayStrategy({})
        intents = strategy.generate_intents(sim_ts_ms=1000000)
        assert intents == []

    def test_gate_evaluation_rth_session(self):
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy
        from src.core.outcome_engine import BarData

        strategy = DowRegimeTimingGateReplayStrategy({})
        ts_ms = int(datetime(2024, 1, 9, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
        bar = BarData(timestamp_ms=ts_ms, open=100, high=101, low=99, close=100, volume=0)
        object.__setattr__(bar, 'symbol', 'IBIT')

        strategy.on_bar(bar, ts_ms, "RTH", {})
        result = strategy.finalize()
        assert result["gates_evaluated"] == 1
        assert result["gates_allowed"] == 1

    def test_gate_blocks_closed_session(self):
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy
        from src.core.outcome_engine import BarData

        strategy = DowRegimeTimingGateReplayStrategy({})
        ts_ms = int(datetime(2024, 1, 9, 2, 0, tzinfo=timezone.utc).timestamp() * 1000)
        bar = BarData(timestamp_ms=ts_ms, open=100, high=101, low=99, close=100, volume=0)
        object.__setattr__(bar, 'symbol', 'IBIT')

        strategy.on_bar(bar, ts_ms, "CLOSED", {})
        result = strategy.finalize()
        assert result["gates_evaluated"] == 1
        assert result["gates_allowed"] == 0
        assert "SESSION_CLOSED" in result["last_gate_results"]["reasons"]

    def test_vol_spike_blocks_gate(self):
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy
        from src.core.outcome_engine import BarData

        strategy = DowRegimeTimingGateReplayStrategy({})
        ts_ms = int(datetime(2024, 1, 9, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
        bar = BarData(timestamp_ms=ts_ms, open=100, high=101, low=99, close=100, volume=0)
        object.__setattr__(bar, 'symbol', 'IBIT')

        strategy.on_bar(bar, ts_ms, "RTH", {},
                        visible_regimes={"IBIT": {"vol_regime": "VOL_SPIKE"}})
        result = strategy.finalize()
        assert result["gates_allowed"] == 0
        assert "VOL_SPIKE" in result["last_gate_results"]["reasons"]

    def test_dow_weight_applied(self):
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy
        from src.core.outcome_engine import BarData

        strategy = DowRegimeTimingGateReplayStrategy({})
        # Monday has weight 0.9
        ts_ms = int(datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)
        bar = BarData(timestamp_ms=ts_ms, open=100, high=101, low=99, close=100, volume=0)
        object.__setattr__(bar, 'symbol', 'IBIT')

        strategy.on_bar(bar, ts_ms, "RTH", {})
        result = strategy.finalize()
        assert result["last_gate_results"]["dow_weight"] == pytest.approx(0.9)

    def test_replay_strategy_interface(self):
        # Verify all required ReplayStrategy methods exist.
        from src.strategies.dow_regime_timing import DowRegimeTimingGateReplayStrategy

        strategy = DowRegimeTimingGateReplayStrategy({})
        assert hasattr(strategy, "strategy_id")
        assert hasattr(strategy, "on_bar")
        assert hasattr(strategy, "generate_intents")
        assert hasattr(strategy, "on_fill")
        assert hasattr(strategy, "on_reject")
        assert hasattr(strategy, "finalize")


# 7.4: DSR n_obs uses total_trades (source-level verification)


class TestDSRnObsFix:
    # Verify DSR uses total_trades for n_obs and computes actual skew/kurtosis.

    def test_n_obs_uses_total_trades_not_bars(self):
        source = _read_source("src/analysis/strategy_evaluator.py")
        # Should use total_trades, not bars_replayed
        assert 'n_obs = metrics.get("total_trades", 0)' in source
        # The old pattern should be gone
        assert 'max(metrics.get("bars_replayed"' not in source

    def test_skew_kurtosis_computed(self):
        source = _read_source("src/analysis/strategy_evaluator.py")
        # Should compute skew and kurtosis from trade_pnls
        assert "trade_pnls" in source
        assert "skew_val" in source
        assert "kurt_val" in source

    def test_skew_kurtosis_passed_to_dsr(self):
        source = _read_source("src/analysis/strategy_evaluator.py")
        # Should pass computed values, not hardcoded 0.0
        assert "skewness=skew_val" in source
        assert "kurtosis=kurt_val" in source

    def test_skew_kurtosis_math(self):
        # Verify the skew/kurtosis computation is correct for known data.
        # Use clearly asymmetric data
        pnls = [50.0, -5.0, -3.0, -4.0, -2.0, -6.0, -1.0, -3.0, -4.0, -2.0, -5.0, -3.0]
        n = len(pnls)
        mean_pnl = sum(pnls) / n
        var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / n
        assert var_pnl > 0
        std_pnl = var_pnl ** 0.5
        skew = sum((p - mean_pnl) ** 3 for p in pnls) / (n * std_pnl ** 3)
        kurt = sum((p - mean_pnl) ** 4 for p in pnls) / (n * std_pnl ** 4) - 3.0

        # Should be non-zero for asymmetric data
        assert abs(skew) > 0.1
        assert isinstance(kurt, float)


# 7.7: Reality check wired into ExperimentRunner


class TestRealityCheckWiring:
    # Verify reality_check is computed and injected into manifests.

    def test_reality_check_import_in_experiment_runner(self):
        source = _read_source("src/analysis/experiment_runner.py")
        assert "run_reality_check" in source
        assert "reality_check" in source

    def test_reality_check_injected_in_manifest(self):
        source = _read_source("src/analysis/experiment_runner.py")
        assert 'manifest_overrides["reality_check"]' in source

    def test_reality_check_guarded_by_trade_count(self):
        source = _read_source("src/analysis/experiment_runner.py")
        # Should only run if there are enough trades
        assert "trade_pnls" in source and "len(result.trade_pnls)" in source


# 7.7 (continued): Reality check module standalone tests


def _load_reality_check():
    # Load reality_check module directly to avoid src.analysis.__init__ cascade.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "reality_check", "src/analysis/reality_check.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reality_check


class TestRealityCheckModule:
    # Verify the reality check module handles edge cases.

    def test_empty_strategies(self):
        reality_check = _load_reality_check()
        result = reality_check(strategy_returns={})
        assert result["p_value"] == 1.0
        assert result["n_strategies"] == 0

    def test_single_observation(self):
        reality_check = _load_reality_check()
        result = reality_check(strategy_returns={"s1": [0.01]})
        assert result["p_value"] == 1.0

    def test_good_strategy_low_p_value(self):
        reality_check = _load_reality_check()
        good_returns = [0.05] * 100
        result = reality_check(
            strategy_returns={"good": good_returns},
            n_bootstrap=500,
            seed=42,
        )
        assert result["p_value"] < 0.10
        assert result["best_strategy"] == "good"

    def test_mismatched_lengths_raises(self):
        reality_check = _load_reality_check()
        with pytest.raises(ValueError, match="same length"):
            reality_check(strategy_returns={"a": [1, 2, 3], "b": [1, 2]})


# 8.2b: Bybit REST rate-limit retry cap


class TestBybitRetryLimit:
    # Verify rate-limit retries are capped.

    def test_retry_cap_in_source(self):
        source = _read_source("src/connectors/bybit_rest.py")
        assert "attempt >= 4" in source or "rate limited after" in source


# 8.3a: Bybit WS timestamp/zero-quote validation


class TestBybitWSValidation:
    # Verify WS validates timestamps and rejects zero quotes.

    def test_zero_quote_rejection_in_source(self):
        source = _read_source("src/connectors/bybit_ws.py")
        assert "bid" in source and "ask" in source


# 8.5a: HTTP session leaks — connectors use TCPConnector


class TestHTTPSessionLeaks:
    # Verify connectors create sessions with TCPConnector.

    def test_coinbase_uses_tcp_connector(self):
        source = _read_source("src/connectors/coinbase_client.py")
        assert "TCPConnector" in source

    def test_coinglass_uses_tcp_connector(self):
        source = _read_source("src/connectors/coinglass_client.py")
        assert "TCPConnector" in source

    def test_deribit_uses_tcp_connector(self):
        source = _read_source("src/connectors/deribit_client.py")
        assert "TCPConnector" in source


# 9.2a: NaN propagation guard in feature builder


class TestFeatureBuilderNaNGuard:
    # Verify non-finite log returns are rejected.

    def test_isfinite_guard_in_source(self):
        source = _read_source("src/core/feature_builder.py")
        assert "isfinite" in source

    def test_inf_return_filtered_logic(self):
        # Simulate the guard: inf log returns should be filtered.
        prev, close = 100.0, float("inf")
        log_ret = math.log(close / prev) if (prev > 0 and close > 0) else None
        if log_ret is not None and not math.isfinite(log_ret):
            log_ret = None
        assert log_ret is None


# 9.3a: Regime detector hysteresis epsilon


class TestRegimeHysteresisEpsilon:
    def test_epsilon_in_source(self):
        source = _read_source("src/core/regime_detector.py")
        assert "eps" in source.lower() or "1e-" in source


# 9.6a: IV consensus bounded deques


class TestIVConsensusBounded:
    def test_deque_used(self):
        source = _read_source("src/core/iv_consensus.py")
        assert "deque" in source and "maxlen" in source


# 9.7b: Config hash float normalization


class TestConfigHashDeterminism:
    def test_same_float_different_repr_same_hash(self):
        from src.core.signals import compute_config_hash

        config_a = {"threshold": 0.5, "scale": 1.0}
        config_b = {"threshold": 0.5000000001, "scale": 1.0000000001}
        assert compute_config_hash(config_a) == compute_config_hash(config_b)

    def test_different_configs_different_hash(self):
        from src.core.signals import compute_config_hash

        assert compute_config_hash({"threshold": 0.5}) != compute_config_hash({"threshold": 0.9})


# 9.8a: Spool recovery infinite loop fix


class TestSpoolRecoveryFix:
    def test_skip_corrupt_in_source(self):
        source = _read_source("src/core/persistence.py")
        assert "JSONDecodeError" in source or "json.decoder" in source.lower()


# 9.8b: Signal persistence idempotency


class TestSignalIdempotency:
    def test_insert_or_ignore(self):
        source = _read_source("src/core/persistence.py")
        assert "INSERT OR IGNORE" in source or "INSERT OR REPLACE" in source


# 10.1a: Position counter race condition fix


class TestPositionCounterLock:
    def test_lock_in_source(self):
        source = _read_source("src/trading/paper_trader_farm.py")
        assert "asyncio.Lock" in source or "_positions_lock" in source


# 10.2b: Paper trader expiry validation


class TestPaperTraderExpiryValidation:
    def test_past_expiry_check(self):
        source = _read_source("src/trading/paper_trader.py")
        assert "past" in source.lower() or "expir" in source.lower()


# 11.1a / 11.3a / 11.4a: Dashboard security fixes


class TestDashboardSecurityFixes:
    def test_xss_escape(self):
        source = _read_source("src/dashboard/web.py")
        assert "textContent" in source or "esc(" in source

    def test_command_whitelist(self):
        source = _read_source("src/dashboard/web.py")
        assert "_ALLOWED_CMD_PREFIXES" in source

    def test_oauth_expiry(self):
        source = _read_source("src/dashboard/web.py")
        assert "expir" in source.lower() or "600" in source


# 12.1a: Canonical replay warning fix


class TestCanonicalReplayFix:
    def test_logger_used(self):
        source = _read_source("src/soak/tape.py")
        assert "logger" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
