import argparse
import glob
import json
import os
import h5py
import numpy as np
import pandas as pd


def convert_artiq_h5_dir(h5_folder: str, out_csv: str = "duke_artiq_runs.csv") -> pd.DataFrame:
    """
    Walks a directory of native ARTIQ .h5 result files and converts scan steps
    into sequential tabular rows grouped by `rid`.
    Preserves explicit 0.0 photon counts (dark/lost ion) strictly distinct from NaN (unprobed channel).
    """
    files = sorted(glob.glob(os.path.join(h5_folder, "**", "*.h5"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {h5_folder}")

    rows = []
    for fpath in files:
        try:
            with h5py.File(fpath, "r") as f:
                rid = int(f["rid"][()]) if "rid" in f else os.path.basename(fpath).split("-")[0]
                start_ts = float(f["start_time"][()]) if "start_time" in f else 1760000000.0

                exp_role = "CalibrationScan"
                beam_line = "radial_729nm"
                fault_regime = None
                if "expid" in f:
                    raw_exp = f["expid"][()]
                    exp_str = raw_exp.decode() if isinstance(raw_exp, bytes) else str(raw_exp)
                    try:
                        meta = json.loads(exp_str)
                        exp_role = meta.get("class_name", exp_role)
                        beam_line = meta.get("beam_channel", beam_line)
                        fault_regime = meta.get("regime_tag", None)
                    except Exception:
                        exp_role = exp_str[:32]

                if "datasets" not in f:
                    continue

                dsets_1d, scalars = {}, {}
                step_dt = None
                for k in f["datasets"].keys():
                    arr = np.array(f["datasets"][k])
                    if k == "shot_offsets_s" and arr.ndim == 1:
                        step_dt = arr.astype(float)
                        continue
                    if np.issubdtype(arr.dtype, np.number):
                        if arr.ndim == 1 and len(arr) >= 4:
                            dsets_1d[k] = arr.astype(float)
                        elif arr.ndim == 0:
                            scalars[k] = float(arr)
                        elif arr.ndim == 2 and arr.shape[0] >= 4:
                            dsets_1d[f"{k}_mean"] = arr.mean(axis=1).astype(float)
                            dsets_1d[f"{k}_zero_frac"] = (arr == 0).mean(axis=1).astype(float)

                if not dsets_1d:
                    continue

                n_steps = max(len(v) for v in dsets_1d.values())
                for step in range(n_steps):
                    offset_s = float(step_dt[step]) if (step_dt is not None and step < len(step_dt)) else step * 0.05
                    row = {
                        "rid": f"rid_{rid}",
                        "shot_ts": pd.to_datetime(start_ts + offset_s, unit="s").isoformat(),
                        "exp_class": exp_role,
                        "beam_channel": beam_line,
                    }
                    if fault_regime is not None:
                        row["regime_tag"] = fault_regime
                    for k, sc_val in list(scalars.items())[:4]:
                        row[f"env_{k}"] = sc_val
                    for k, arr in dsets_1d.items():
                        row[k] = float(arr[step]) if step < len(arr) else np.nan
                    rows.append(row)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[ARTIQ H5 Adapter] Converted {df['rid'].nunique()} runs ({len(df):,} scan steps) -> {out_csv}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert native ARTIQ .h5 run directory to Core BTR CSV")
    p.add_argument("--h5-dir", required=True)
    p.add_argument("--out", default="duke_artiq_runs.csv")
    args = p.parse_args()
    convert_artiq_h5_dir(args.h5_dir, args.out)
