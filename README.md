# Core BTR (`core-btr`)

**Deterministic, multi-lens Set Transformer retrieval pod for sequential tabular logs and native ARTIQ HDF5 telemetry.**

Standard text-serialized RAG pipelines flatten multi-row operational histories into prose, losing structural trajectories, inter-event cadence, and the distinction between explicit `0.0` measurements (e.g., dark-ion `0` photon counts or `$0.00` wire probes) and `NULL` / unprobed channels.

`core-btr` implements an un-flattened multi-sense constellation based on two-layer Set Transformers (Lee et al., 2019) and contrastive structural regimes (Royse, 2026):

* **Un-flattened Orthogonal Viewpoints:** Evaluates sequential event windows across independent 64-D Set Transformer lenses (`how_scale`, `who_roles`, `zero_vs_null`, `ordered_transitions`, `cadence_rhythm`, `extrema_lifecycle`, `seasonal_phase`, `role_value_binding`).
* **Differentiation Contract:** Enforces pairwise Spearman rank correlation `|rho| <= 0.60` across admitted lenses, a minimum 5-sense floor (`SensoryGapRefusal`), frozen weights, and SHA-256 weight-hash verification on cold boot.
* **Calibrated Scarcity & Fail-Closed Refusal:** Uses corpus-calibrated stopping cutoffs and per-lens veto floors to return exact unpadded counts on scarce regimes and `EMPTY_BELOW_CUTOFF` on unseen out-of-distribution faults—zero LLM calls, zero GPU required at runtime (~3.6 ms/query on CPU).
* **Drop-In REST + FastMCP Pod:** Exposes `/health`, `/similar/{entity_id}`, `/match-window`, and MCP tools (`lookup_historical_twins`, `match_live_event_window`, `diagnose_artiq_run_drift`) for air-gapped Docker or local workstation deployment.

---

## Verified Benchmark (Apple Silicon CPU, Zero GPU)

| Suite | Bad Tab RAG (`MiniLM` Row-to-Text) | Core BTR Constellation (5–6 Frozen Senses) | Scarcity Gate (`<=3` True Matches) | Outlier Refusal (`0` True Matches) | Live Query Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `verify_core.py` (Sequential Ledger / Ops) | 53.3% P@10 | **100.0% P@10** | **6/6** unpadded | **10/10** refused | ~3.5 ms |
| `duke_artiq_runs` (295 ARTIQ `.h5` Scans) | 45.0% P@10 | **100.0% P@10** | **4/5** unpadded | **10/10** refused | **3.62 ms** |

---

## Quickstart

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[bench]"

# 1. Verify core contract (Spearman <= 0.60, scarcity, outlier refusal, SHA-256 reload)
python verify_core.py

# 2. Generate mock ARTIQ .h5 runs, convert to tabular scan steps, and export a frozen pod
python make_mock_artiq.py
python artiq_h5_to_csv.py --h5-dir ./mock_duke_h5 --out duke_artiq_runs.csv
python csv_shootout.py \
  --csv duke_artiq_runs.csv \
  --entity-col rid \
  --time-col shot_ts \
  --label-col regime_tag \
  --scarce-label scarce_aom_thermal_droop \
  --novel-label novel_laser_unlock_spike \
  --export-bundle artifacts/duke_pod.pt

# 3. Serve REST + FastMCP endpoints
uvicorn server:app --host 127.0.0.1 --port 8000
```

## Architecture Layout

* `core_btr/views.py` — Domain-separated signed categorical hashing, explicit `0.0` vs `NaN` separation, and 8 physical viewpoint encoders.
* `core_btr/models.py` — Two-layer Set Transformer (`MAB` / `SAB` / `PMA`), view-native contrastive training, and byte-identical determinism probe.
* `core_btr/contract.py` — Pairwise Spearman redundancy gate (`|rho| <= 0.60`), `N >= 5` sense floor, Weighted Reciprocal Rank Fusion, and hardest-impostor stopping calibration.
* `core_btr/bundle.py` — Deterministic `CoreBTRPod` serialization and cold-start SHA-256 verification.
* `artiq_h5_to_csv.py` — Native ARTIQ `.h5` run directory adapter (`rid` grouping).
* `server.py` — FastAPI + `FastMCP` air-gapped inference server.

---

## Licensing (Dual-Licensed)

`core-btr` is source-available and dual-licensed:

1. **Academic, Non-Profit & Personal Research (Free):** Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) (`PolyForm-Noncommercial-1.0.0`). Educational institutions, universities, public research organizations, and government laboratories may use, modify, and run `core-btr` free of charge.
2. **Commercial & Enterprise Use (Paid Pod License):** Any use by a for-profit organization—including internal R&D, manufacturing telemetry, financial operations, or production deployment—requires a separate commercial license from **Aizzia**. Open an issue or contact Aizzia via [github.com/aizzia](https://github.com/aizzia) for commercial pilot and air-gapped pod licensing.
