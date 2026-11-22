"""Exercise the HTTP application against the documented seed-42 synthetic evaluation bundle."""
import sys

import pandas as pd
from fastapi.testclient import TestClient

from server import app


def check(csv_path):
    rows = pd.read_csv(csv_path)
    labels = rows.groupby("rid").regime_tag.last().to_dict()
    with TestClient(app) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["indexed_entities"] == 90, health.text
        known = client.get("/similar/rid_50043")
        known.raise_for_status()
        matches = known.json()["matches"]
        assert matches and all(m["entity_id"] != "rid_50043" for m in matches), known.text
        for rid, expected in [("rid_50022", "OK"), ("rid_50286", "EMPTY_BELOW_CUTOFF")]:
            assert client.get(f"/similar/{rid}").status_code == 404
            window = rows.loc[rows.rid == rid].drop(columns=["rid", "regime_tag"]).astype(object)
            events = window.where(window.notna(), None).to_dict("records")
            response = client.post("/match-window", json={"events": events})
            response.raise_for_status()
            result = response.json()
            assert result["status"] == expected, result
            matches = result["matches"]
            if expected == "OK":
                assert matches and all(labels[m["entity_id"]] == labels[rid] for m in matches), result
            else:
                assert not matches, result
        print("Passed historical lookup, held-out match and held-out refusal.")


if __name__ == "__main__":
    check(sys.argv[1])
