from argus_kalshi.runner import (
    _infer_asset_from_ticker,
    _standalone_classify_liq,
    _standalone_classify_vol,
    _standalone_risk_regime,
)


def test_infer_asset_from_ticker() -> None:
    assert _infer_asset_from_ticker("KXBTC15M-26MAR051500-00") == "BTC"
    assert _infer_asset_from_ticker("KXETH-26MAR0515-B2080") == "ETH"
    assert _infer_asset_from_ticker("KXSOL-26MAR0515-B140") == "SOL"
    assert _infer_asset_from_ticker("UNKNOWN") == ""


def test_vol_classifier_spike_and_high() -> None:
    assert _standalone_classify_vol(
        abs_ret_ema=0.0022, vol_baseline_ema=0.0005, last_jump_pct=0.0008
    ) == "VOL_SPIKE"
    assert _standalone_classify_vol(
        abs_ret_ema=0.0010, vol_baseline_ema=0.0005, last_jump_pct=0.0019
    ) == "VOL_HIGH"


def test_vol_classifier_normal_and_low() -> None:
    assert _standalone_classify_vol(
        abs_ret_ema=0.0005, vol_baseline_ema=0.0005, last_jump_pct=0.0003
    ) == "VOL_NORMAL"
    assert _standalone_classify_vol(
        abs_ret_ema=0.00015, vol_baseline_ema=0.0004, last_jump_pct=0.00005
    ) == "VOL_LOW"


def test_liq_classifier_avoids_overblocking() -> None:
    assert _standalone_classify_liq(spread_ema=2.2, depth_ema=520.0, ob_idle_s=1.0) == "LIQ_NORMAL"
    assert _standalone_classify_liq(spread_ema=4.5, depth_ema=210.0, ob_idle_s=2.0) == "LIQ_LOW"
    assert _standalone_classify_liq(spread_ema=8.0, depth_ema=80.0, ob_idle_s=13.0) == "LIQ_DRIED"


def test_risk_regime_requires_material_stress() -> None:
    assert _standalone_risk_regime(
        {"BTC": "VOL_NORMAL", "ETH": "VOL_HIGH"},
        {"BTC": "LIQ_NORMAL", "ETH": "LIQ_NORMAL"},
    ) == "NEUTRAL"
    assert _standalone_risk_regime(
        {"BTC": "VOL_SPIKE", "ETH": "VOL_NORMAL"},
        {"BTC": "LIQ_LOW", "ETH": "LIQ_NORMAL"},
    ) == "RISK_OFF"
