# Containment Module (`contain.py`)

Vectorized, single-resource fire containment simulation based on BehavePlus §13.7. Uses fixed-step masked RK4 integration over NumPy arrays — all cells advance simultaneously, terminal cells frozen via `np.where` masking.

**Scope / limitations:**
- Single resource, single attack point
- No terrain adjustment
- Constant ROS (no growth acceleration)
- Simplified ellipse perimeter/area model

---

## Status Codes

Mirrors the ContainSim C++ enum:

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `STATUS_REPORTED` | Fire reported, simulation running |
| 3 | `STATUS_CONTAINED` | Fire contained — line built ≥ perimeter |
| 4 | `STATUS_OVERRUN` | Fire grew beyond 10× report size before attack |
| 5 | `STATUS_EXHAUSTED` | (reserved) |
| 8 | `STATUS_TIME_LIMIT` | Max integration steps reached without resolution |

---

## Components

### Module-level constants

```python
STATUS_REPORTED   = np.int32(1)
STATUS_CONTAINED  = np.int32(3)
STATUS_OVERRUN    = np.int32(4)
STATUS_EXHAUSTED  = np.int32(5)
STATUS_TIME_LIMIT = np.int32(8)
```

### `run_contain_sim_array()`

```python
run_contain_sim_array(
    report_spread_rate,   # ft/min — fire ROS at report time
    lw_ratio,             # dimensionless — fire length-to-width ratio
    report_size,          # acres — fire size at report time
    production_rate,      # chains/hr — suppression line production rate
    attack_time,          # minutes — lag from report to initial attack
    max_steps=500,        # max RK4 integration steps
    dist_step=0.01        # integration distance step (chains)
) -> dict
```

**Inputs:** scalar or NumPy array (all broadcast together).

**Returns dict:**

| Key | Dtype | Description |
|-----|-------|-------------|
| `'status'` | int32 | Final status code per cell |
| `'contained_time'` | float | Elapsed time to containment (min); 0 if not contained |
| `'final_perimeter'` | float | Fire perimeter at simulation end (chains) |
| `'final_area'` | float | Fire area at simulation end (acres) |

**Integration loop logic:**
1. `dt` = `dist_step / fire_ros` (time increment per step)
2. Perimeter → `π × head_dist × (1 + 1/lw_ratio)` (ellipse approx)
3. Area → `π × head_dist² / (4 × lw_ratio) / 10` (chains² → acres)
4. Suppression line accrues at `production_rate / 60` chains/min once `time ≥ attack_time`
5. **CONTAINED** when `line_built ≥ perimeter` and `time ≥ attack_time`
6. **OVERRUN** when `area > report_size × 10` before `attack_time`
7. **TIME_LIMIT_EXCEEDED** for any cell still active after `max_steps`

---

## Usage Example

### Scalar — single fire scenario

```python
from behave_fire.components.contain import run_contain_sim_array, STATUS_CONTAINED

result = run_contain_sim_array(
    report_spread_rate=30.0,   # ft/min
    lw_ratio=3.0,              # length-to-width
    report_size=2.0,           # acres at report
    production_rate=40.0,      # chains/hr
    attack_time=20.0,          # 20 min lag to attack
)

if result['status'][0] == STATUS_CONTAINED:
    print(f"Contained in {result['contained_time'][0]:.1f} min")
    print(f"Final perimeter: {result['final_perimeter'][0]:.2f} chains")
    print(f"Final area:      {result['final_area'][0]:.2f} acres")
else:
    print(f"Status code: {result['status'][0]}")
```

### Array — batch sweep over attack delay

```python
import numpy as np
from behave_fire.components.contain import (
    run_contain_sim_array,
    STATUS_CONTAINED, STATUS_OVERRUN, STATUS_TIME_LIMIT,
)

attack_times = np.arange(5, 61, 5)   # 5, 10, 15, ... 60 min

result = run_contain_sim_array(
    report_spread_rate=30.0,
    lw_ratio=3.0,
    report_size=2.0,
    production_rate=40.0,
    attack_time=attack_times,
)

status_labels = {
    STATUS_CONTAINED:  "CONTAINED",
    STATUS_OVERRUN:    "OVERRUN",
    STATUS_TIME_LIMIT: "TIME_LIMIT",
}

for t, s, ct, fp, fa in zip(
    attack_times,
    result['status'],
    result['contained_time'],
    result['final_perimeter'],
    result['final_area'],
):
    label = status_labels.get(s, f"STATUS_{s}")
    print(
        f"Attack @{t:3.0f} min → {label:<12} "
        f"contained_time={ct:6.1f} min  "
        f"perimeter={fp:6.2f} ch  area={fa:5.2f} ac"
    )
```

**Example output:**
```
Attack @  5 min → CONTAINED    contained_time=  12.3 min  perimeter=  3.41 ch  area= 0.18 ac
Attack @ 10 min → CONTAINED    contained_time=  24.7 min  perimeter=  6.82 ch  area= 0.71 ac
...
Attack @ 45 min → OVERRUN      contained_time=   0.0 min  perimeter= 30.11 ch  area=20.34 ac
```

---

## Unit Reference

| Parameter | Input unit | Internal unit | Conversion |
|-----------|-----------|--------------|------------|
| `report_spread_rate` | ft/min | chains/min | ÷ 66 |
| `production_rate` | chains/hr | chains/min | ÷ 60 |
| `attack_time` | minutes | minutes | — |
| `report_size` | acres | acres | — |
| Perimeter output | — | chains | — |
| Area output | — | acres | — |
