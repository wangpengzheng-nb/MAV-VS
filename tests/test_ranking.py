from autovs.ranking import rank_rows


def test_ranker_respects_score_direction_and_scaffold_diversity():
    rows = [
        {"source_id": "a", "smiles": "A", "scaffold": "same", "docking_affinity": -10, "plip_score": 8},
        {"source_id": "b", "smiles": "B", "scaffold": "same", "docking_affinity": -9, "plip_score": 7},
        {"source_id": "c", "smiles": "C", "scaffold": "same", "docking_affinity": -8, "plip_score": 6},
        {"source_id": "d", "smiles": "D", "scaffold": "other", "docking_affinity": -7, "plip_score": 5},
    ]
    ranked = rank_rows(rows, top_n=4, max_per_scaffold=2)
    assert ranked[0]["source_id"] == "a"
    assert {row["source_id"] for row in ranked} == {"a", "b", "d"}

