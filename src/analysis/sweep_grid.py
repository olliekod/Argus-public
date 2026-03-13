from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from typing import Any


def expand_sweep_grid(raw: dict | None) -> dict[str, list[Any]]:
    """Normalize sweep YAML values into lists for parameter-grid expansion.

    Each key supports:
    - list: used as-is
    - dict range spec: {min,max,step?,num_steps?,round?}
    - scalar: wrapped as [value]
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Sweep grid must be a mapping/dict, got {type(raw).__name__}."
        )

    expanded: dict[str, list[Any]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            expanded[key] = value
            continue

        if isinstance(value, dict):
            expanded[key] = _expand_range_spec(key, value)
            continue

        expanded[key] = [value]

    return expanded


def _expand_range_spec(param_name: str, spec: dict[str, Any]) -> list[Any]:
    if "min" not in spec or "max" not in spec:
        raise ValueError(
            f"Sweep range for '{param_name}' must include both 'min' and 'max'."
        )

    min_dec = _to_decimal(spec["min"], param_name, "min")
    max_dec = _to_decimal(spec["max"], param_name, "max")
    if max_dec < min_dec:
        raise ValueError(
            f"Sweep range for '{param_name}' must satisfy max >= min."
        )

    round_decimals = _resolve_round_decimals(param_name, spec)

    has_step = "step" in spec
    has_num_steps = "num_steps" in spec
    if has_step and has_num_steps:
        raise ValueError(
            f"Sweep range for '{param_name}' cannot specify both 'step' and 'num_steps'."
        )
    if not has_step and not has_num_steps:
        raise ValueError(
            f"Sweep range for '{param_name}' must include either 'step' or 'num_steps'."
        )

    if has_step:
        return _expand_step_mode(param_name, min_dec, max_dec, spec, round_decimals)

    return _expand_num_steps_mode(param_name, min_dec, max_dec, spec, round_decimals)


def _expand_step_mode(
    param_name: str,
    min_dec: Decimal,
    max_dec: Decimal,
    spec: dict[str, Any],
    round_decimals: int,
) -> list[Any]:
    step_dec = _to_decimal(spec["step"], param_name, "step")
    if step_dec <= 0:
        raise ValueError(f"Sweep step for '{param_name}' must be > 0.")

    span = max_dec - min_dec
    steps_count = int((span / step_dec).to_integral_value(rounding=ROUND_FLOOR))
    values = [min_dec + (step_dec * i) for i in range(steps_count + 1)]

    tolerance = Decimal("1e-12")
    final = values[-1] if values else min_dec
    if abs(max_dec - final) <= tolerance:
        values[-1] = max_dec

    return [_finalize_number(v, round_decimals) for v in values]


def _expand_num_steps_mode(
    param_name: str,
    min_dec: Decimal,
    max_dec: Decimal,
    spec: dict[str, Any],
    round_decimals: int,
) -> list[Any]:
    num_steps_raw = spec["num_steps"]
    if isinstance(num_steps_raw, bool) or not isinstance(num_steps_raw, int):
        raise ValueError(
            f"Sweep num_steps for '{param_name}' must be an integer >= 1."
        )
    if num_steps_raw < 1:
        raise ValueError(
            f"Sweep num_steps for '{param_name}' must be >= 1."
        )

    step_dec = (max_dec - min_dec) / Decimal(num_steps_raw)
    values = [min_dec + (step_dec * i) for i in range(num_steps_raw + 1)]
    values[-1] = max_dec

    return [_finalize_number(v, round_decimals) for v in values]


def _resolve_round_decimals(param_name: str, spec: dict[str, Any]) -> int:
    if "round" in spec:
        round_raw = spec["round"]
        if isinstance(round_raw, bool) or not isinstance(round_raw, int):
            raise ValueError(
                f"Sweep round for '{param_name}' must be an integer >= 0."
            )
        if round_raw < 0:
            raise ValueError(
                f"Sweep round for '{param_name}' must be >= 0."
            )
        return round_raw

    if "step" in spec:
        step_raw = spec["step"]
        step_dec = _to_decimal(step_raw, param_name, "step")
        exponent = step_dec.normalize().as_tuple().exponent
        return max(0, -exponent)

    return 4


def _to_decimal(value: Any, param_name: str, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(
            f"Sweep {field} for '{param_name}' must be numeric, got {type(value).__name__}."
        )
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Sweep {field} for '{param_name}' must be numeric."
        ) from exc


def _finalize_number(value: Decimal, round_decimals: int) -> int | float:
    quant = Decimal("1").scaleb(-round_decimals)
    rounded = value.quantize(quant)
    if round_decimals == 0:
        return int(rounded)
    return float(rounded)
