import pytest

from src.analysis.sweep_grid import expand_sweep_grid


def test_mixed_list_range_and_scalar_expands():
    raw = {
        "max_vol_regime": ["VOL_LOW", "VOL_NORMAL"],
        "min_vrp": {"min": 0.02, "max": 0.08, "step": 0.01},
        "max_vol_regime_single": "VOL_LOW",
    }

    expanded = expand_sweep_grid(raw)

    assert expanded["max_vol_regime"] == ["VOL_LOW", "VOL_NORMAL"]
    assert expanded["min_vrp"] == [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    assert expanded["max_vol_regime_single"] == ["VOL_LOW"]


def test_step_mode_includes_max_when_exact():
    expanded = expand_sweep_grid({"x": {"min": 0.0, "max": 0.3, "step": 0.1}})
    assert expanded["x"] == [0.0, 0.1, 0.2, 0.3]


def test_step_mode_excludes_values_beyond_max():
    expanded = expand_sweep_grid({"x": {"min": 0.0, "max": 0.25, "step": 0.1}})
    assert expanded["x"] == [0.0, 0.1, 0.2]


def test_num_steps_mode_generates_endpoints_and_count():
    expanded = expand_sweep_grid({"x": {"min": 1.0, "max": 2.0, "num_steps": 4}})
    assert expanded["x"] == [1.0, 1.25, 1.5, 1.75, 2.0]


def test_rounding_with_small_step_is_decimal_stable():
    expanded = expand_sweep_grid({"x": {"min": 0.05, "max": 0.06, "step": 0.005}})
    assert expanded["x"] == [0.05, 0.055, 0.06]


def test_scalar_becomes_single_item_list():
    assert expand_sweep_grid({"x": 5}) == {"x": [5]}


def test_empty_or_none_grid_returns_empty_mapping():
    assert expand_sweep_grid(None) == {}
    assert expand_sweep_grid({}) == {}


@pytest.mark.parametrize(
    "raw,match",
    [
        ([], "mapping/dict"),
        ({"x": {"max": 1, "step": 0.1}}, "min"),
        ({"x": {"min": 0, "max": 1}}, "step' or 'num_steps"),
        ({"x": {"min": 0, "max": 1, "step": 0}}, "must be > 0"),
        ({"x": {"min": 0, "max": 1, "num_steps": 0}}, "must be >= 1"),
        ({"x": {"min": "a", "max": 1, "step": 0.1}}, "must be numeric"),
        ({"x": {"min": 0, "max": 1, "step": 0.1, "num_steps": 2}}, "cannot specify both"),
    ],
)
def test_invalid_specs_raise_value_error(raw, match):
    with pytest.raises(ValueError, match=match):
        expand_sweep_grid(raw)
