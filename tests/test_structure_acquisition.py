import json

import httpx
import pytest

from autovs.structure_acquisition import acquire_rcsb_structures, rank_verified_holo_candidates


def _research():
    return {
        "target_uniprot_id": "P10415", "target_organism": "Homo sapiens",
        "verified_pdb_structures": [
            {"pdb_id": "6O0K", "resolution": 2.0, "deposition_year": 2018,
             "has_ligand": True, "uniprot_mapped": True},
            {"pdb_id": "1ABC", "resolution": 1.5, "deposition_year": 2020,
             "has_ligand": False, "uniprot_mapped": True},
            {"pdb_id": "2ABC", "resolution": 1.0, "deposition_year": 2021,
             "has_ligand": True, "uniprot_mapped": False},
        ],
    }


def test_structure_candidates_require_holo_and_uniprot_mapping():
    candidates = rank_verified_holo_candidates(_research())
    assert [item["pdb_id"] for item in candidates] == ["6O0K"]
    with pytest.raises(ValueError, match="UniProt"):
        rank_verified_holo_candidates({"target_organism": "Homo sapiens", "verified_pdb_structures": []})


def test_acquisition_uses_fixed_rcsb_origin_and_records_checksum(tmp_path, monkeypatch):
    research = tmp_path / "research.json"; research.write_text(json.dumps(_research()))
    pdb = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "HETATM    2  C1  LIG A 101       1.000   1.000   1.000  1.00 20.00           C\nEND\n"
    )
    seen = []

    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def get(self, url):
            seen.append(url)
            return httpx.Response(200, text=pdb, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = acquire_rcsb_structures(research, tmp_path / "download", selected_strategy_id="selected")
    assert seen == ["https://files.rcsb.org/download/6O0K.pdb"]
    assert result["candidate_structures"][0].endswith("6O0K.pdb")
    report = json.loads(result["acquisition_report"].read_text())
    assert report["downloaded"][0]["sha256"]
    assert report["selected_strategy_id"] == "selected"
