# Core BTR (`core-btr`)

**Deterministic, multi-lens Set Transformer retrieval pod for sequential tabular logs and native ARTIQ HDF5 telemetry.**

Standard text-serialized RAG pipelines flatten multi-row operational histories into prose, losing structural trajectories, inter-event cadence, and the distinction between explicit `0.0` measurements (e.g., dark-ion `0` photon counts or `$0.00` wire probes) and `NULL` / unprobed channels.

`core-btr` implements an un-flattened multi-sense constellation based on two-layer Set Transformers (Lee et al., 2019) and contrastive structural regimes (Royse, 2026):

* **Un-flattened Orthogonal Viewpoints:** Evaluates sequential event windows across independent 64-D Set Transformer lenses (`how_scale`, `who_roles`, `zero_vs_null`, `ordered_transitions`, `cadence_rhythm`, `extrema_lifecycle`, `seasonal_phase`, `role_value_binding`).
* **Differentiation Contract:** Enforces pairwise Spearman rank correlation `|rho| <= 0.60` across admitted lenses, a minimum 5-sense floor (`SensoryGapRefusal`), frozen weights, and SHA-256 weight-hash verification on cold boot.
* **Calibrated Scarcity & Fail-Closed Refusal:** Uses corpus-calibrated stopping cutoffs and per-lens veto floors to return exact unpadded counts on scarce regimes and `EMPTY_BELOW_CUTOFF` on unseen out-of-distribution faults—zero LLM calls, zero GPU required at runtime (~3.6 ms/query on CPU).
* **Drop-In REST + FastMCP Pod:** Exposes `/health`, `/similar/{entity_id}`, `/match-window`, and MCP tools (`lookup_historical_twins`, `match_live_event_window`, `diagnose_artiq_run_drift`) for air-gapped Docker or local workstation deployment.

---

## Original benchmark (Apple Silicon CPU, zero GPU)

| Suite | Bad Tab RAG (`MiniLM` Row-to-Text) | Core BTR Constellation (5–6 Frozen Senses) | Scarcity Gate (`<=3` True Matches) | Outlier Refusal (`0` True Matches) | Live Query Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `verify_core.py` (Sequential Ledger / Ops) | 53.3% P@10 | **100.0% P@10** | **6/6** unpadded | **10/10** refused | ~3.5 ms |
| `duke_artiq_runs` (295 ARTIQ `.h5` Scans) | 45.0% P@10 | **100.0% P@10** | **4/5** unpadded | **10/10** refused | **3.62 ms** |

---

The historical BTR percentages above average precision among returned matches per ordinary query (an empty answer scores zero). The CSV scorecard also counts every requested slot, so withholding nine results and returning one correct match earns 1/10. The documented quickstart generates synthetic scans. It does not establish performance on experimental lab data.

On the seed-42 synthetic split, 42 ordinary queries requested ten matches each:

| Method | Correct / requested | Correct / returned | Empty queries |
|---|---:|---:|---:|
| Core BTR | 417/420 (99.3%) | 417/417 | 0/42 |
| Nearest neighbors on pooled input views | 412/420 (98.1%) | 412/420 | 0/42 |
| MiniLM on serialized rows | 189/420 (45.0%) | 189/420 | 0/42 |

The pooled comparator uses the same fitted, masked input views. It averages each view over valid rows, normalizes it, then ranks by equal-weight similarity, with no trained lenses or refusal cutoff. On the separate scarcity and novel queries, BTR passed 4/5 and 10/10 respectively. These two checks evaluate BTR’s refusal behavior; the table compares ordinary retrieval only.

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
  --evaluation-bundle artifacts/evaluation.pt \
  --export-bundle artifacts/duke_pod.pt

# 3. Serve REST + FastMCP endpoints
uvicorn server:app --host 127.0.0.1 --port 8000
```

Use a fresh directory for the quickstart: the mock generator replaces its own mock_duke_h5 directory. The evaluation bundle contains the 90 indexed entities used for scoring. The demo bundle includes all 283 non-novel entities and is intended for browsing the historical corpus. Use the evaluation bundle for held-out checks. For a fully local comparator pass after installing dependencies, add --skip-text-baseline to skip downloading MiniLM.

### Serve offline

Build once with network access. Mount a trusted, locally generated bundle read-only at runtime. The image contains the server and dependencies; each lab supplies its own index.

```bash
docker build -t core-btr .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" core-btr
```

To verify historical lookup, a held-out live match and a novel-window refusal through the HTTP application inside a network-isolated container, use the synthetic evaluation bundle generated above:

```bash
docker run --rm --network none -e PYTHONPATH=/app -e BTR_BUNDLE_PATH=/data/evaluation.pt \
  -v "$PWD/artifacts/evaluation.pt:/data/evaluation.pt:ro" \
  -v "$PWD/duke_artiq_runs.csv:/data/input.csv:ro" \
  -v "$PWD/tests/smoke_mock_pod.py:/tmp/smoke.py:ro" \
  core-btr python /tmp/smoke.py /data/input.csv
```

The loader uses PyTorch deserialization. Only load bundles from trusted sources. New lens fingerprints hash tensor names, shapes, types and weight bytes independently of PyTorch's archive format. Existing 16-character fingerprints retain their original validation routine. Fingerprints stored inside the bundle check consistency; authenticating a received bundle requires verifying its provenance separately.

## Architecture Layout

* `core_btr/views.py` — Domain-separated signed categorical hashing, explicit `0.0` vs `NaN` separation, and 8 physical viewpoint encoders.
* `core_btr/models.py` — Two-layer Set Transformer (`MAB` / `SAB` / `PMA`), view-native contrastive training, and byte-identical determinism probe.
* `core_btr/contract.py` — Pairwise Spearman redundancy gate (`|rho| <= 0.60`), `N >= 5` sense floor, Weighted Reciprocal Rank Fusion, and hardest-impostor stopping calibration.
* `core_btr/bundle.py` — Deterministic `CoreBTRPod` serialization and cold-start SHA-256 verification.
* `artiq_h5_to_csv.py` — Native ARTIQ `.h5` run directory adapter (`rid` grouping).
* `server.py` — FastAPI + `FastMCP` air-gapped inference server.

---

---

## Licensing & Commercial Terms

SPDX-License-Identifier: PolyForm-Small-Business-1.0.0

```text
Required Notice: Copyright 2026 Aizzia (https://tabsense.io)
```

* **Small Business & Indie Use:** Licensed under the unaltered [**PolyForm Small Business License 1.0.0**](./LICENSE) for organizations with fewer than 100 total employees/contractors and less than $1,000,000 USD (2019 CPI-adjusted) annual revenue.
* **Academic & Public Research Labs:** Accredited universities and government laboratories receive a free supplemental grant for non-commercial scientific research (see [COMMERCIAL.md](./COMMERCIAL.md)).
* **Enterprise & Financial Institution Licensing:** Organizations exceeding the Small Business headcount or revenue thresholds—including larger enterprises, commercial banks, credit unions, card issuers, and payment processors—must purchase a commercial license from Aizzia. Contact **licensing@aizzia.ai** or see [**COMMERCIAL.md**](./COMMERCIAL.md) for pricing and pilot terms.
* **Version Scope:** The PolyForm Small Business License 1.0.0 applies prospectively to commits and releases from this transition forward; any prior MIT-licensed commits or releases remain under the MIT License.
