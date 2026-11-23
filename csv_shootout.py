import argparse
import os
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from core_btr import CoreBTRPod, query_constellation


def ordinary_scores(results, k=10):
    """Each query supplies the relevance flags of its returned matches."""
    correct = sum(sum(hits) for hits in results)
    returned = sum(map(len, results))
    requested = k * len(results)
    return {
        "queries": len(results),
        "correct": int(correct),
        "returned": returned,
        "requested": requested,
        "empty": sum(not len(hits) for hits in results),
        "precision_at_k": correct / requested if requested else None,
        "precision_returned": correct / returned if returned else None,
    }


def pooled_views(builder, frame):
    """Same fitted input views, equal weight, no learned lenses or refusal."""
    ids, _, views, padding = builder.transform_dataframe(frame)
    blocks = []
    for values in views.values():
        means = np.where(padding[:, :, None], 0, values).sum(axis=1)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        blocks.append(means / np.maximum(norms, 1e-12))
    return ids, np.concatenate(blocks, axis=1)


def run_csv_shootout(
    csv_path: str,
    entity_col: str,
    time_col: Optional[str] = None,
    label_col: Optional[str] = None,
    scarce_label: Optional[str] = None,
    novel_label: Optional[str] = None,
    export_bundle: Optional[str] = None,
    epochs: int = 35,
    evaluation_bundle: Optional[str] = None,
    skip_text_baseline: bool = False,
):
    if evaluation_bundle and export_bundle and Path(evaluation_bundle).resolve() == Path(export_bundle).resolve():
        raise ValueError("Evaluation and demo bundles require different paths.")
    df = pd.read_csv(csv_path)
    print(f"\n[CSV Shootout] Loaded {len(df):,} rows across {df[entity_col].nunique()} entities from '{csv_path}'")

    # Deterministic entity split: Train (50%) / DB Index (35%) / Held-Out Query (15%)
    rng = np.random.default_rng(42)
    ent_meta = df.groupby(entity_col, sort=False)[label_col].last() if label_col else pd.Series("unlabeled", index=df[entity_col].unique())

    train_ents, db_ents, query_ents = [], [], []
    for lbl, grp_ents in ent_meta.groupby(ent_meta):
        e_list = grp_ents.index.to_numpy().copy()
        rng.shuffle(e_list)
        if novel_label and lbl == novel_label:
            # Novel regime NEVER appears in train or DB -> tests fail-closed EMPTY_BELOW_CUTOFF!
            query_ents.extend(e_list[:10])
        elif scarce_label and lbl == scarce_label:
            # Scarce regime: train sees examples, DB holds ONLY 3 true matches, query holds 5
            train_ents.extend(e_list[:18])
            db_ents.extend(e_list[18:21])
            query_ents.extend(e_list[21:26])
        else:
            n = len(e_list)
            n_tr, n_db = int(n * 0.50), int(n * 0.35)
            train_ents.extend(e_list[:n_tr])
            db_ents.extend(e_list[n_tr : n_tr + n_db])
            query_ents.extend(e_list[n_tr + n_db :])

    train_df = df[df[entity_col].isin(set(train_ents))]
    db_df = df[df[entity_col].isin(set(db_ents))]
    query_df = df[df[entity_col].isin(set(query_ents))]

    pod = CoreBTRPod()
    audit = pod.fit_corpus(
        train_df, entity_col=entity_col, time_col=time_col, label_col=label_col, epochs=epochs
    )
    pod.index_dataframe(db_df)
    if evaluation_bundle:
        Path(evaluation_bundle).parent.mkdir(parents=True, exist_ok=True)
        pod.save(evaluation_bundle)

    print(f"Admitted Senses ({len(audit.admitted_lenses)}/8) : {audit.admitted_lenses}")
    print(f"Max Pairwise Spearman |rho|   : {audit.max_pairwise_spearman:.3f} (Contract <= 0.600)")
    print(f"Calibrated Cutoff (tau)       : {audit.tau_cutoff:.4f}")

    if label_col:
        db_lbl = {str(eid): str(label) for eid, label in db_df.groupby(entity_col)[label_col].last().items()}
        q_ids, q_lbls, q_slots = pod.encode_dataframe_queries(query_df, label_col=label_col)

        ord_p10, scarce_hits, scarce_total, empty_hits, empty_total = [], 0, 0, 0, 0
        btr_hits = []
        for i, (qid, qlbl) in enumerate(zip(q_ids, q_lbls)):
            sq = {k: q_slots[k][i] for k in audit.admitted_lenses}
            res = query_constellation(sq, pod.db_slots, pod.db_ids, audit, top_k=10)
            ret = [db_lbl[m["entity_id"]] for m in res["matches"]]

            if novel_label and qlbl == novel_label:
                empty_total += 1
                if res["status"] == "EMPTY_BELOW_CUTOFF" and res["count"] == 0:
                    empty_hits += 1
            elif scarce_label and qlbl == scarce_label:
                scarce_total += 1
                if 1 <= len(ret) <= 3 and all(r == qlbl for r in ret):
                    scarce_hits += 1
            else:
                btr_hits.append([r == qlbl for r in ret])
                ord_p10.append(np.mean([r == qlbl for r in ret]) if ret else 0.0)

        ord_pairs = [(qid, qlbl) for qid, qlbl in zip(q_ids, q_lbls) if qlbl not in {novel_label, scarce_label}]
        db_ids, pooled_db = pooled_views(pod.builder, db_df)
        query_ids, pooled_queries = pooled_views(pod.builder, query_df)
        queries = dict(zip(query_ids, pooled_queries))
        pooled_hits = []
        for qid, label in ord_pairs:
            nearest = np.argsort(-(pooled_db @ queries[qid]), kind="stable")[:10]
            pooled_hits.append([db_lbl[db_ids[i]] == label for i in nearest])
        comparisons = {"Core BTR": btr_hits, "Pooled-view nearest neighbors": pooled_hits}
        if not skip_text_baseline and ord_pairs:
            # Bad Tab RAG baseline (row-to-text serialization + MiniLM)
            feat_cols = [c for c in df.columns if c not in {entity_col, label_col, time_col}]
            from sentence_transformers import SentenceTransformer
            txt_model = SentenceTransformer("all-MiniLM-L6-v2")

            def serialize_entities(sub_df):
                blobs = {}
                for eid, g in sub_df.groupby(entity_col, sort=False):
                    rows_txt = [
                        " ".join(f"{c}={'NULL' if pd.isna(r[c]) else r[c]}" for c in feat_cols[:6])
                        for _, r in g.head(10).iterrows()
                    ]
                    blobs[str(eid)] = " | ".join(rows_txt)
                return blobs

            db_blobs = serialize_entities(db_df)
            q_blobs = serialize_entities(query_df)
            db_emb = txt_model.encode([db_blobs[eid] for eid in pod.db_ids], normalize_embeddings=True)
            q_emb = txt_model.encode([q_blobs[qid] for qid, _ in ord_pairs], normalize_embeddings=True)
            sim = q_emb @ db_emb.T
            text_hits = [
                [db_lbl[pod.db_ids[j]] == label for j in np.argsort(-sim[i])[:10]]
                for i, (_, label) in enumerate(ord_pairs)
            ]
            comparisons["MiniLM serialized rows"] = text_hits
        print("\n" + "=" * 72)
        print(f"  BENCHMARK SCORECARD: {os.path.basename(csv_path)}")
        print("=" * 72)
        for name, hits in comparisons.items():
            print(f"  {name} (ordinary queries): {ordinary_scores(hits)}")
        if ord_p10:
            print(f"  Legacy BTR mean query precision (empty scores zero): {np.mean(ord_p10):.1%}")
        if scarce_total:
            print(f"  Scarcity Gate (<=3 True Matches in DB)     : {scarce_hits}/{scarce_total}  clean unpadded")
        if empty_total:
            print(f"  Fail-Closed Refusal (Novel Hardware Fault) : {empty_hits}/{empty_total} refused cleanly")
        print("=" * 72)

    if export_bundle:
        os.makedirs(os.path.dirname(export_bundle) or ".", exist_ok=True)
        # Index full non-novel historical corpus before saving production pod
        full_hist = df if not novel_label else df[df[label_col] != novel_label]
        n_indexed = pod.index_dataframe(full_hist)
        pod.save(export_bundle)
        print(f"\n[Pod Exported] Indexed {n_indexed} historical runs -> {export_bundle}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--entity-col", required=True)
    p.add_argument("--time-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--scarce-label", default=None)
    p.add_argument("--novel-label", default=None)
    p.add_argument("--export-bundle", default=None)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--evaluation-bundle", help="Save the held-out scoring index before demo reindexing")
    p.add_argument("--skip-text-baseline", action="store_true", help="Skip MiniLM download and comparison")
    args = p.parse_args()
    run_csv_shootout(
        args.csv,
        args.entity_col,
        args.time_col,
        args.label_col,
        args.scarce_label,
        args.novel_label,
        args.export_bundle,
        args.epochs,
        args.evaluation_bundle,
        args.skip_text_baseline,
    )
