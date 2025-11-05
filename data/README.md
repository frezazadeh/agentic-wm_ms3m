# Data directory

Place your NumPy arrays here:

- `x.npy` with shape **(N, L, F)**: windows of past KPIs
- `y.npy` with shape **(N, F)**: next-step KPI vectors

The feature ordering must match `WMMS3MConfig.feature_names` in `src/wm_ms3m/core.py`.
