from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr


class SensoryGapRefusal(RuntimeError):
    pass


@dataclass
class RosterAudit:
    admitted_lenses: List[str]
    rejected_lenses: Dict[str, str]
    max_pairwise_spearman: float
    lens_weights: Dict[str, float]
    tau_cutoff: float
    per_lens_floors: Dict[str, float] = field(default_factory=dict)


def pairwise_cosine_stream(slots: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    s64 = slots.astype(np.float64)
    return np.sum(s64[pair_i] * s64[pair_j], axis=1)


def select_differentiated_roster(
    dev_slots: Dict[str, np.ndarray],
    dev_labels: Optional[np.ndarray] = None,
    max_spearman: float = 0.60,
    min_senses: int = 5,
    seed: int = 42,
) -> RosterAudit:
    rng = np.random.default_rng(seed)
    n = len(next(iter(dev_slots.values())))
    n_pairs = min(5000, n * (n - 1) // 2)

    pair_i = rng.integers(0, n, size=n_pairs * 2)
    pair_j = rng.integers(0, n, size=n_pairs * 2)
    diff = pair_i != pair_j
    pair_i, pair_j = pair_i[diff][:n_pairs], pair_j[diff][:n_pairs]

    streams = {name: pairwise_cosine_stream(mat, pair_i, pair_j) for name, mat in dev_slots.items()}

    standalone_scores = {}
    chance_floor = 0.0
    if dev_labels is not None:
        _, counts = np.unique(dev_labels, return_counts=True)
        chance_floor = float(np.sum((counts / n) ** 2))

    for name, mat in dev_slots.items():
        s64 = mat.astype(np.float64)
        sim = s64 @ s64.T
        np.fill_diagonal(sim, -2.0)
        if dev_labels is not None:
            k = min(10, max(1, n // 5))
            top_k = np.argsort(-sim, axis=1)[:, :k]
            standalone_scores[name] = float(np.mean(dev_labels[top_k] == dev_labels[:, None]))
        else:
            top1 = np.max(sim, axis=1)
            med = np.median(sim, axis=1)
            standalone_scores[name] = float(np.mean(top1 - med))

    ordered_candidates = sorted(dev_slots.keys(), key=lambda x: standalone_scores[x], reverse=True)

    admitted: List[str] = []
    rejected: Dict[str, str] = {}
    worst_rho = 0.0

    for cand in ordered_candidates:
        c_stream = streams[cand]
        if np.std(c_stream) < 1e-12:
            rejected[cand] = "COLLAPSED_CONSTANT_OUTPUT"
            continue

        too_close = False
        for kept in admitted:
            rho, _ = spearmanr(c_stream, streams[kept])
            rho_val = 0.0 if np.isnan(rho) else abs(float(rho))
            if rho_val > max_spearman:
                rejected[cand] = f"REDUNDANT_WITH_{kept} (|rho|={rho_val:.3f} > {max_spearman})"
                too_close = True
                break

        if not too_close:
            for kept in admitted:
                rho, _ = spearmanr(c_stream, streams[kept])
                worst_rho = max(worst_rho, 0.0 if np.isnan(rho) else abs(float(rho)))
            admitted.append(cand)

    if len(admitted) < min_senses:
        raise SensoryGapRefusal(
            f"SENSORY_GAP_REFUSAL: Only {len(admitted)} lenses cleared |rho|<={max_spearman} "
            f"(admitted={admitted}, rejected={rejected}). Contract D4 requires >= {min_senses}."
        )

    if dev_labels is not None:
        lifts = {k: max(0.01, standalone_scores[k] - chance_floor) ** 2 for k in admitted}
    else:
        lifts = {k: max(0.05, standalone_scores[k]) for k in admitted}
    w_sum = sum(lifts.values())
    weights = {k: float(lifts[k] / w_sum) for k in admitted}

    tau, lens_floors = calibrate_global_stopping_cutoff(
        {k: dev_slots[k] for k in admitted}, weights=weights, labels=dev_labels
    )

    return RosterAudit(
        admitted_lenses=admitted,
        rejected_lenses=rejected,
        max_pairwise_spearman=worst_rho,
        lens_weights=weights,
        tau_cutoff=tau,
        per_lens_floors=lens_floors,
    )


def calibrate_global_stopping_cutoff(
    admitted_slots: Dict[str, np.ndarray],
    weights: Dict[str, float],
    labels: Optional[np.ndarray] = None,
) -> tuple[float, Dict[str, float]]:
    n = len(next(iter(admitted_slots.values())))
    w_cos = np.zeros((n, n), dtype=np.float64)
    per_mat_sim = {}
    for name, mat in admitted_slots.items():
        s64 = mat.astype(np.float64)
        sim = s64 @ s64.T
        per_mat_sim[name] = sim
        w_cos += weights[name] * sim

    eye = np.eye(n, dtype=bool)
    lens_floors: Dict[str, float] = {}

    if labels is not None:
        pos_mask = (labels[:, None] == labels[None, :]) & (~eye)
        neg_mask = (labels[:, None] != labels[None, :]) & (~eye)

        for name, sim in per_mat_sim.items():
            if weights[name] >= 0.10 and pos_mask.any():
                lens_floors[name] = float(np.percentile(sim[pos_mask], 3))
            else:
                lens_floors[name] = -1.0

        # Calibrate against each entity's #1 hardest impostor across the corpus
        hardest_neg = np.where(neg_mask, w_cos, -2.0).max(axis=1)
        valid_neg = hardest_neg[hardest_neg > -1.5]
        pos_q25 = np.array(
            [float(np.percentile(w_cos[i, pos_mask[i]], 25)) for i in range(n) if pos_mask[i].any()],
            dtype=np.float64,
        )

        if len(valid_neg) > 0 and len(pos_q25) > 0:
            # Calibrate directly between corpus-wide 95th-pct hardest impostor and true twin distribution
            imp_95 = float(np.percentile(valid_neg, 95))
            pos_med = float(np.percentile(pos_q25, 50))
            if imp_95 < pos_med:
                # Place tau just above the 95th-percentile hardest impostor!
                return float(imp_95 + 0.15 * (pos_med - imp_95)), lens_floors
            return float(np.percentile(valid_neg, 92)), lens_floors

    # Unsupervised fallback: calibrate against authentic top-1 nearest-neighbour distribution
    np.fill_diagonal(w_cos, -2.0)
    top1_sim = np.max(w_cos, axis=1)
    for name, sim in per_mat_sim.items():
        s_copy = sim.copy()
        np.fill_diagonal(s_copy, -2.0)
        lens_floors[name] = float(np.percentile(np.max(s_copy, axis=1), 5)) if weights[name] >= 0.10 else -1.0
    return float(np.percentile(top1_sim, 10)), lens_floors


def query_constellation(
    query_slots: Dict[str, np.ndarray],
    db_slots: Dict[str, np.ndarray],
    db_ids: List[str],
    audit: RosterAudit,
    top_k: int = 10,
    rrf_k: int = 60,
    exclude_id: Optional[str] = None,
) -> dict:
    n_db = len(db_ids)
    rrf = np.zeros(n_db, dtype=np.float64)
    w_cos = np.zeros(n_db, dtype=np.float64)
    per_lens: Dict[str, np.ndarray] = {}

    for name in audit.admitted_lenses:
        w = audit.lens_weights[name]
        q_v = query_slots[name].astype(np.float64)
        db_m = db_slots[name].astype(np.float64)
        cos = db_m @ q_v
        per_lens[name] = cos
        w_cos += w * cos
        ranks = np.argsort(np.argsort(-cos)) + 1
        rrf += w * (1.0 / (rrf_k + ranks))

    if exclude_id is not None:
        for idx, eid in enumerate(db_ids):
            if eid == exclude_id:
                rrf[idx] = -1.0
                w_cos[idx] = -2.0

    cand_order = np.argsort(-rrf)
    matches = []
    floors = getattr(audit, "per_lens_floors", {}) or {}

    for idx in cand_order:
        if len(matches) >= top_k:
            break
        if w_cos[idx] < audit.tau_cutoff:
            continue
        # Multi-sense veto: if ANY primary bearing lens says this candidate is out-of-regime, veto it!
        if any(per_lens[k][idx] < floors.get(k, -1.0) for k in audit.admitted_lenses):
            continue

        matches.append({
            "entity_id": db_ids[idx],
            "rrf_score": round(float(rrf[idx]), 6),
            "weighted_cosine": round(float(w_cos[idx]), 4),
            "lens_contributions": {k: round(float(per_lens[k][idx]), 4) for k in audit.admitted_lenses},
        })

    return {
        "status": "OK" if matches else "EMPTY_BELOW_CUTOFF",
        "admitted_sense_count": len(audit.admitted_lenses),
        "tau_cutoff": round(audit.tau_cutoff, 4),
        "count": len(matches),
        "matches": matches,
    }
