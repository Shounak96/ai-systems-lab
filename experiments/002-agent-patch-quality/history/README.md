# Experiment History

V4 is the completed evaluation and the primary result. The earlier iterations below document how the methodology evolved; their reports were moved here without changing their contents.

## V1 — five-scenario unified-diff pilot

- [Original README](v1/README.md)
- [Original report](v1/REPORT.md)
- [Result rows](../artifacts/results.csv)
- [Raw-metric audit summary](../artifacts/audit.json)
- Source: [`pilot.py`](../pilot.py) and [`audit_results.py`](../audit_results.py)

Some V1 raw JSON and its environment block embedded the author's absolute local workspace path. Those local originals remain ignored rather than published; the metric-bearing CSV and audit summary are unchanged.

## V2 — first official AutoSaddler engine integration

- [Report](v2/REPORT.md)
- [Comparison](../artifacts/v2/comparison.json)
- [Environment](../artifacts/v2/environment.json)
- Source: [`v2_integration.py`](../v2_integration.py)

V2 raw requests embedded volatile absolute traceback paths. They remain ignored; the comparison and environment summaries are unchanged.

## V3 — frozen five-scenario structured-mutation comparison

- [Report](v3/REPORT.md)
- [Preregistration](../artifacts/v3/preregistration.json)
- [Results](../artifacts/v3/results.json)
- [Audit](../artifacts/v3/audit.json)
- [Raw model records](../artifacts/v3/raw/)
- Source: [`v3_integration.py`](../v3_integration.py)

V3 introduced sanitized byte-identical model inputs, multiple structured scenarios, a preflight gate, and explicit separation of rejected versus evaluated mutations. Generated engine workspaces are excluded.
