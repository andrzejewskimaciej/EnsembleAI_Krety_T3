# Heat Pump Grid Load Forecasting

Hackathon solution for Task 3 - predicting monthly electrical grid load for heat pump networks across Poland (May–October 2025).

## Problem

Given 7 months of 5-minute telemetry from heat pump devices (Oct 2024 - Apr 2025), forecast the average monthly grid load indicator (`x2`) per device for the summer–autumn season of 2025.

This is an **extrapolation task** - models must generalize beyond the training period into a new season.

## Scoring

Evaluated using **MAE (Mean Absolute Error)** — lower is better, 0 = perfect.

| Score | Months | Weight |
|---|---|---|
| Leaderboard (public) | May–Jun 2025 | - |
| Final score | May–Oct 2025 | 2/6 validation + 4/6 test |

> ⚠️ Feature `x2` is withheld in the validation and test sets.
