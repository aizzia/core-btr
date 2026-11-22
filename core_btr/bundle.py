from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from .contract import RosterAudit, query_constellation, select_differentiated_roster
from .models import SetTransformerLens, freeze_and_verify_lens, train_commissioned_lens
from .views import ViewpointBuilder


class CoreBTRPod:
    def __init__(self):
        self.builder: Optional[ViewpointBuilder] = None
        self.lenses: Dict[str, SetTransformerLens] = {}
        self.lens_hashes: Dict[str, str] = {}
        self.audit: Optional[RosterAudit] = None
        self.db_ids: List[str] = []
        self.db_slots: Dict[str, np.ndarray] = {}

    def fit_corpus(
        self,
        train_df: pd.DataFrame,
        entity_col: str,
        time_col: Optional[str] = None,
        label_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        epochs: int = 35,
        max_spearman: float = 0.60,
        min_senses: int = 5,
        device: str = "cpu",
    ) -> RosterAudit:
        excl = list(exclude_cols or []) + ([label_col] if label_col else [])
        self.builder = ViewpointBuilder().fit(
            train_df, entity_col=entity_col, time_col=time_col, exclude_cols=excl
        )
        _, raw_labels, tr_views, tr_masks = self.builder.transform_dataframe(train_df, label_col=label_col)

        y_int = None
        if raw_labels is not None:
            uniq = {lbl: i for i, lbl in enumerate(sorted(set(raw_labels)))}
            y_int = np.array([uniq[l] for l in raw_labels], dtype=np.int64)

        # Lenses train strictly on their OWN physical view (never on y_int!)
        tr_slots: Dict[str, np.ndarray] = {}
        for idx, v_name in enumerate(ViewpointBuilder.VIEW_NAMES):
            lens, w_hash = train_commissioned_lens(
                tr_views[v_name],
                tr_masks,
                epochs=epochs,
                seed=42 + idx * 17,
                device=device,
            )
            self.lenses[v_name] = lens.cpu()
            self.lens_hashes[v_name] = w_hash
            with torch.no_grad():
                x_t = torch.tensor(tr_views[v_name], dtype=torch.float32)
                m_t = torch.tensor(tr_masks, dtype=torch.bool)
                tr_slots[v_name] = self.lenses[v_name](x_t, m_t).numpy()

        # Question labels (y_int) are used ONLY for roster weighting & tau calibration
        self.audit = select_differentiated_roster(
            tr_slots, dev_labels=y_int, max_spearman=max_spearman, min_senses=min_senses
        )
        self.lenses = {k: self.lenses[k] for k in self.audit.admitted_lenses}
        self.lens_hashes = {k: self.lens_hashes[k] for k in self.audit.admitted_lenses}
        return self.audit

    def index_dataframe(self, db_df: pd.DataFrame) -> int:
        ids, _, views, masks = self.builder.transform_dataframe(db_df)
        self.db_ids = ids
        m_t = torch.tensor(masks, dtype=torch.bool)
        self.db_slots = {}
        with torch.no_grad():
            for name, lens in self.lenses.items():
                x_t = torch.tensor(views[name], dtype=torch.float32)
                self.db_slots[name] = lens(x_t, m_t).numpy()
        return len(self.db_ids)

    def encode_dataframe_queries(self, query_df: pd.DataFrame, label_col: Optional[str] = None):
        q_ids, q_lbls, views, masks = self.builder.transform_dataframe(query_df, label_col=label_col)
        m_t = torch.tensor(masks, dtype=torch.bool)
        q_slots = {}
        with torch.no_grad():
            for name, lens in self.lenses.items():
                x_t = torch.tensor(views[name], dtype=torch.float32)
                q_slots[name] = lens(x_t, m_t).numpy()
        return q_ids, q_lbls, q_slots

    def query_by_id(self, entity_id: str, top_k: int = 10) -> dict:
        if entity_id not in self.db_ids:
            return {"status": "NOT_IN_CORPUS", "entity_id": entity_id, "matches": []}
        idx = self.db_ids.index(entity_id)
        q_slots = {k: self.db_slots[k][idx] for k in self.audit.admitted_lenses}
        res = query_constellation(q_slots, self.db_slots, self.db_ids, self.audit, top_k=top_k, exclude_id=entity_id)
        res["lens_hashes"] = self.lens_hashes
        return res

    def query_live_events(self, events: List[dict], top_k: int = 10) -> dict:
        v_single, mask_single = self.builder.encode_single_window(events)
        m_t = torch.tensor(mask_single[None, :], dtype=torch.bool)
        q_slots = {}
        with torch.no_grad():
            for name, lens in self.lenses.items():
                x_t = torch.tensor(v_single[name][None, :], dtype=torch.float32)
                q_slots[name] = lens(x_t, m_t).numpy()[0]
        res = query_constellation(q_slots, self.db_slots, self.db_ids, self.audit, top_k=top_k)
        res["lens_hashes"] = self.lens_hashes
        return res

    def save(self, path: str):
        payload = {
            "contract": self.builder.contract,
            "audit": self.audit,
            "lens_in_dims": {k: m.in_dim for k, m in self.lenses.items()},
            "lens_states": {k: m.state_dict() for k, m in self.lenses.items()},
            "lens_hashes": self.lens_hashes,
            "db_ids": self.db_ids,
            "db_slots": self.db_slots,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "CoreBTRPod":
        raw = torch.load(path, map_location="cpu", weights_only=False)
        pod = cls()
        pod.builder = ViewpointBuilder(contract=raw["contract"])
        pod.audit = raw["audit"]
        pod.db_ids = raw["db_ids"]
        pod.db_slots = raw["db_slots"]
        pod.lens_hashes = raw["lens_hashes"]

        dummy_mask = torch.zeros((4, pod.builder.contract.max_tokens), dtype=torch.bool)
        for name, state in raw["lens_states"].items():
            in_dim = raw["lens_in_dims"][name]
            lens = SetTransformerLens(in_dim=in_dim, d_model=64, num_heads=4)
            lens.load_state_dict(state)
            verify_hash = freeze_and_verify_lens(
                lens, torch.randn(4, pod.builder.contract.max_tokens, in_dim), dummy_mask
            )
            if verify_hash != pod.lens_hashes[name]:
                raise RuntimeError(
                    f"FAIL_CLOSED: SHA-256 mismatch on lens '{name}' ({verify_hash} != {pod.lens_hashes[name]})"
                )
            pod.lenses[name] = lens
        return pod
