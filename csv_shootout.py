import argparse
import os
import time
from typing import Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from core_btr import CoreBTRPod, query_constellation


def run_csv_shootout(
    csv_path: str,
    entity_col: str,
    time_col: Optional[str] = None,
    label_col: Optional[str] = None,
    scarce_label: Optional[str] = None,
    novel_label: Optional[str] = None,
    export_bundle: Optional[str] = None,
    epochs: int = 35,
):
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

    t0 = time.time()
    pod = CoreBTRPod()
    audit = pod.fit_corpus(
        train_df, entity_col=entity_col, time_col=time_col, label_col=label_col, epochs=epochs
    )
    pod.index_dataframe(db_df)
    elapsed = time.time() - t0

    print(f"Admitted Senses ({len(audit.admitted_lenses)}/8) : {audit.admitted_lenses}")
    print(f"Max Pairwise Spearman |rho|   : {audit.max_pairwise_spearman:.3f} (Contract <= 0.600)")
    print(f"Calibrated Cutoff (tau)       : {audit.tau_cutoff:.4f}")

    if label_col:
        db_lbl = db_df.groupby(entity_col)[label_col].last().to_dict()
        q_ids, q_lbls, q_slots = pod.encode_dataframe_queries(query_df, label_col=label_col)

        ord_p10, scarce_hits, scarce_total, empty_hits, empty_total = [], 0, 0, 0, 0
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
                ord_p10.append(np.mean([r == qlbl for r in ret]) if ret else 0.0)

        # Bad Tab RAG baseline (row-to-text serialization + MiniLM)
        feat_cols = [c for c in df.columns if c not in {entity_col, label_col, time_col}]
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
        ord_pairs = [(qid, qlbl) for qid, qlbl in zip(q_ids, q_lbls) if qlbl not in {novel_label, scarce_label}]
        q_emb = txt_model.encode([q_blobs[qid] for qid, _ in ord_pairs], normalize_embeddings=True)
        sim = q_emb @ db_emb.T
        bad_rag_p10 = np.mean([
            np.mean([db_lbl[pod.db_ids[j]] == qlbl for j in np.argsort(-sim[idx])[:10]])
            for idx, (_, qlbl) in enumerate(ord_pairs)
        ]) * 100.0

        print("\n" + "=" * 72)
        print(f"  BENCHMARK SCORECARD: {os.path.basename(csv_path)}")
        print("=" * 72)
        print(f"  Bad Tab RAG (Serialized Rows + MiniLM)     : {bad_rag_p10:5.1f}%  P@10")
        print(f"  Core BTR Constellation ({len(audit.admitted_lenses)} Frozen Senses)   : {np.mean(ord_p10)*100:5.1f}%  P@10  ({elapsed:.1f}s)")
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
    )
