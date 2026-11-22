import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def signed_categorical_code(namespace: str, value: str, bits: int = 32) -> np.ndarray:
    digest = hashlib.sha256(f"{namespace}::{value}".encode("utf-8")).digest()
    n_bytes = (bits + 7) // 8
    unpacked = np.unpackbits(np.frombuffer(digest[:n_bytes], dtype=np.uint8))[:bits]
    return unpacked.astype(np.float32) * 2.0 - 1.0


@dataclass
class SchemaContract:
    entity_col: str
    time_col: Optional[str] = None
    role_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    max_tokens: int = 80
    min_tokens: int = 4
    role_hash_bits: int = 32
    num_medians: Dict[str, float] = field(default_factory=dict)
    num_iqrs: Dict[str, float] = field(default_factory=dict)


class ViewpointBuilder:
    # 8 disjoint physical viewpoints (zero shared positional boilerplate across views)
    VIEW_NAMES = (
        "how_scale",
        "who_roles",
        "zero_vs_null",
        "ordered_transitions",
        "cadence_rhythm",
        "extrema_lifecycle",
        "seasonal_phase",
        "role_value_binding",
    )

    def __init__(self, contract: Optional[SchemaContract] = None):
        self.contract = contract

    def fit(
        self,
        df: pd.DataFrame,
        entity_col: str,
        time_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        max_tokens: int = 80,
    ) -> "ViewpointBuilder":
        excl = set(exclude_cols or []) | {entity_col}
        if time_col:
            excl.add(time_col)

        feat_cols = [c for c in df.columns if c not in excl]
        num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
        role_cols = [c for c in feat_cols if c not in num_cols]

        medians, iqrs = {}, {}
        for c in num_cols:
            valid = df[c].dropna()
            medians[c] = float(valid.median()) if len(valid) > 0 else 0.0
            q75 = float(valid.quantile(0.75)) if len(valid) > 0 else 1.0
            q25 = float(valid.quantile(0.25)) if len(valid) > 0 else 0.0
            iqrs[c] = max(1e-4, q75 - q25)

        self.contract = SchemaContract(
            entity_col=entity_col,
            time_col=time_col if (time_col and time_col in df.columns) else None,
            role_cols=role_cols[:6],
            num_cols=num_cols,
            max_tokens=max_tokens,
            num_medians=medians,
            num_iqrs=iqrs,
        )
        return self

    def transform_dataframe(
        self, df: pd.DataFrame, label_col: Optional[str] = None
    ) -> Tuple[List[str], Optional[List[str]], Dict[str, np.ndarray], np.ndarray]:
        if self.contract is None:
            raise RuntimeError("FAIL_CLOSED: ViewpointBuilder must be fit first.")

        c = self.contract
        work = df.copy()

        if c.time_col and c.time_col in work.columns:
            work["_ts"] = pd.to_datetime(work[c.time_col], errors="coerce")
            work = work.sort_values([c.entity_col, "_ts"], kind="mergesort").reset_index(drop=True)
            work["_gap_s"] = (
                work.groupby(c.entity_col)["_ts"]
                .diff()
                .dt.total_seconds()
                .fillna(0.0)
                .clip(lower=0.0)
            )
            tod_rad = 2.0 * np.pi * (work["_ts"].dt.hour.fillna(0) * 60 + work["_ts"].dt.minute.fillna(0)) / 1440.0
            dow_rad = 2.0 * np.pi * work["_ts"].dt.dayofweek.fillna(0) / 7.0
            work["_sin_tod"], work["_cos_tod"] = np.sin(tod_rad), np.cos(tod_rad)
            work["_sin_dow"], work["_cos_dow"] = np.sin(dow_rad), np.cos(dow_rad)
            work["_has_clock"] = (~work["_ts"].isna()).astype(np.float32)
        else:
            work["_gap_s"] = 0.0
            work["_sin_tod"] = work["_cos_tod"] = work["_sin_dow"] = work["_cos_dow"] = 0.0
            work["_has_clock"] = 0.0

        entity_ids, labels = [], ([] if label_col else None)
        accum = {v: [] for v in self.VIEW_NAMES}
        pad_masks = []

        for ent_id, grp in work.groupby(c.entity_col, sort=False):
            if len(grp) < c.min_tokens:
                continue
            grp = grp.tail(c.max_tokens)
            v_dict, p_mask = self._encode_group(grp)

            entity_ids.append(str(ent_id))
            if label_col is not None and label_col in grp.columns:
                non_null = grp[label_col].dropna()
                labels.append(str(non_null.iloc[-1]) if len(non_null) > 0 else "MISSING")

            for k in self.VIEW_NAMES:
                accum[k].append(v_dict[k])
            pad_masks.append(p_mask)

        if not entity_ids:
            raise ValueError(f"FAIL_CLOSED: Zero entities met min_tokens={c.min_tokens}.")

        views_np = {k: np.stack(accum[k], axis=0).astype(np.float32) for k in self.VIEW_NAMES}
        masks_np = np.stack(pad_masks, axis=0)
        return entity_ids, labels, views_np, masks_np

    def encode_single_window(self, events: List[dict]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if self.contract is None:
            raise RuntimeError("FAIL_CLOSED: ViewpointBuilder has no SchemaContract.")
        if len(events) < self.contract.min_tokens:
            raise ValueError(
                f"INSUFFICIENT_HISTORY: Got {len(events)} rows, require >= {self.contract.min_tokens}."
            )
        tmp = pd.DataFrame(events)
        tmp[self.contract.entity_col] = "__live__"
        _, _, views_np, masks_np = self.transform_dataframe(tmp)
        return {k: views_np[k][0] for k in self.VIEW_NAMES}, masks_np[0]

    def _encode_group(self, grp: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        c = self.contract
        n_rows = len(grp)

        if c.num_cols:
            raw_num = grp.reindex(columns=c.num_cols, fill_value=np.nan).to_numpy(dtype=np.float64)
            is_na = np.isnan(raw_num).astype(np.float32)
            is_zero = ((~np.isnan(raw_num)) & (raw_num == 0.0)).astype(np.float32)
            med = np.array([c.num_medians[col] for col in c.num_cols], dtype=np.float64)
            iqr = np.array([c.num_iqrs[col] for col in c.num_cols], dtype=np.float64)
            z_raw = np.where(np.isnan(raw_num), 0.0, (raw_num - med) / iqr)
            scaled_num = np.arcsinh(z_raw).astype(np.float32)
        else:
            scaled_num = np.zeros((n_rows, 1), dtype=np.float32)
            is_na = np.zeros((n_rows, 1), dtype=np.float32)
            is_zero = np.zeros((n_rows, 1), dtype=np.float32)

        if c.role_cols:
            role_blocks = []
            for rcol in c.role_cols:
                vals = grp[rcol].tolist() if rcol in grp.columns else [np.nan] * n_rows
                codes = [
                    signed_categorical_code(rcol, "__MISSING__" if pd.isna(v) else str(v), bits=c.role_hash_bits)
                    for v in vals
                ]
                role_blocks.append(np.stack(codes, axis=0))
            role_mat = np.concatenate(role_blocks, axis=1)
        else:
            role_mat = np.zeros((n_rows, 16), dtype=np.float32)

        prog = np.linspace(-1.0, 1.0, n_rows, dtype=np.float32)[:, None]
        role_shifted = np.roll(role_mat, shift=1, axis=0)
        role_changed = (np.mean(np.abs(role_mat - role_shifted), axis=1, keepdims=True) > 0).astype(np.float32)

        raw_gap_s = grp["_gap_s"].to_numpy(dtype=np.float32)[:, None]
        log_gap = np.log1p(raw_gap_s)
        gap_accel = np.diff(log_gap, axis=0, prepend=log_gap[:1])
        # Rhythm irregularity & sub-minute / sub-hour / multi-day octave bands
        gap_med = np.maximum(1e-3, np.median(raw_gap_s))
        gap_ratio = np.tanh(np.log((raw_gap_s + 1e-3) / gap_med))
        sub_min_flag = (raw_gap_s < 120.0).astype(np.float32) * 2.0 - 1.0
        multi_day_flag = (raw_gap_s > 100000.0).astype(np.float32) * 2.0 - 1.0

        num_diff = np.diff(scaled_num, axis=0, prepend=scaled_num[:1])
        num_dir = np.sign(num_diff).astype(np.float32)

        # Scale-invariant shape / extrema lifecycle
        col_min = np.min(scaled_num, axis=0, keepdims=True)
        col_max = np.max(scaled_num, axis=0, keepdims=True)
        span = np.maximum(1e-4, col_max - col_min)
        unit_shape = (scaled_num - col_min) / span

        clock_mat = grp[["_sin_tod", "_cos_tod", "_sin_dow", "_cos_dow", "_has_clock"]].to_numpy(dtype=np.float32)
        event_polarity = (1.0 - 2.0 * is_zero[:, :1]) * (1.0 - is_na[:, :1])

        raw_views = {
            "how_scale": np.concatenate([scaled_num, np.tanh(scaled_num / 3.0), np.ones_like(scaled_num)], axis=1),
            "who_roles": np.concatenate([role_mat, role_changed], axis=1),
            "zero_vs_null": np.concatenate([is_zero, is_na, is_zero * prog, is_na * prog], axis=1),
            "ordered_transitions": np.concatenate([num_dir, np.clip(num_diff, -3.0, 3.0), prog], axis=1),
            "cadence_rhythm": np.concatenate([gap_ratio, gap_accel, np.cos(log_gap * 1.5), sub_min_flag * 0.5], axis=1),
            "extrema_lifecycle": np.concatenate([unit_shape, unit_shape * prog], axis=1),
            "seasonal_phase": clock_mat,
            "role_value_binding": role_mat[:, :16] * event_polarity,
        }

        pad_len = c.max_tokens - n_rows
        padded_views = {}
        for k, mat in raw_views.items():
            if pad_len > 0:
                pad_block = np.zeros((pad_len, mat.shape[1]), dtype=np.float32)
                padded_views[k] = np.concatenate([mat, pad_block], axis=0)
            else:
                padded_views[k] = mat

        pad_mask = np.zeros(c.max_tokens, dtype=bool)
        if pad_len > 0:
            pad_mask[n_rows:] = True

        return padded_views, pad_mask
