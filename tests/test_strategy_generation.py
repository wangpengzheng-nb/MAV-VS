from autovs.compiler import choose_executable_strategy
from autovs.schemas import InputManifest, LibraryAsset, TargetAsset
from src.agents.strategy_generator import StrategyGeneratorAgent


def _manifest():
    return InputManifest(
        query="a sufficiently detailed screening request",
        library_asset=LibraryAsset(source="user", path="library.smi", sha256="a" * 64),
        target_asset=TargetAsset(source="user", locked=True, path="protein.pdb", sha256="b" * 64),
    )


def test_strategy_context_infers_bcl2_ppi_and_selectivity(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    report = {
        "target_name": "Apoptosis regulator Bcl-2",
        "gene_symbol": "BCL2",
        "uniprot_id": "P10415",
        "_user_query": "寻找 BCL-2 抑制剂，不要作用于 BCLXL",
        "verified_pdb_structures": [{"pdb_id": "6QGG", "resolution": 1.5, "has_ligand": True}],
        "executive_summary": "BCL-2 has a BH3 hydrophobic groove and is a protein-protein interaction target.",
    }
    ctx = StrategyGeneratorAgent.build_strategy_context(report)
    assert ctx.target_class == "PPI"
    assert ctx.pocket_type == "shallow_groove"
    assert ctx.rule_category == "bRo5"
    assert ctx.selectivity_constraints


def test_strategy_context_infers_egfr_kinase():
    report = {
        "target_name": "Epidermal growth factor receptor",
        "gene_symbol": "EGFR",
        "uniprot_id": "P00533",
        "_user_query": "为人源 EGFR 寻找高选择性非共价抑制剂",
        "verified_pdb_structures": [{"pdb_id": "1M17", "resolution": 2.6, "has_ligand": True}],
        "chembl_activities": [{"standard_value": 20.0}, {"standard_value": 500.0}],
        "executive_summary": "EGFR is a kinase with ATP-binding cleft and hinge-binding inhibitor evidence.",
    }
    ctx = StrategyGeneratorAgent.build_strategy_context(report)
    assert ctx.target_class == "Kinase"
    assert ctx.pocket_type == "deep_cleft"
    assert ctx.rule_category == "Ro5"
    assert ctx.has_known_active_ligands is True


def test_no_holo_structure_marks_prediction_gap():
    report = {
        "target_name": "Novel target",
        "gene_symbol": "NT1",
        "structure_readiness": {"predicted_structure_required": True},
        "executive_summary": "No verified experimental holo pocket is available.",
    }
    ctx = StrategyGeneratorAgent.build_strategy_context(report)
    strategies = StrategyGeneratorAgent(api_key="").generate_strategies(report)["strategies"]
    assert ctx.predicted_structure_required is True
    assert any("target_structure_prediction" in item["required_capabilities"] for item in strategies)
    assert any("target_structure_prediction" in " ".join(item["missing_capabilities"]) for item in strategies)


def test_offline_fallback_generates_eight_diverse_gap_aware_strategies(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    report = {
        "target_name": "Apoptosis regulator Bcl-2",
        "gene_symbol": "BCL2",
        "_user_query": "寻找 BCL-2 抑制剂，不要作用于 BCLXL",
        "verified_pdb_structures": [{"pdb_id": "6QGG", "resolution": 1.5, "has_ligand": True}],
        "executive_summary": "BCL-2 has a BH3 hydrophobic groove.",
    }
    strategies = StrategyGeneratorAgent().generate_strategies(report)["strategies"]
    axes = [item["diversity_axis"] for item in strategies]
    assert len(strategies) == 8
    assert len(set(axes)) >= 7
    assert all(item["user_requirement_coverage"] for item in strategies)
    for item in strategies:
        if item["execution_status"] != "currently_executable":
            assert item["missing_capabilities"]


def test_executable_strategy_is_chosen_before_future_capability_strategy():
    future = {
        "strategy_name": "future pharmacophore",
        "execution_status": "future_capability_required",
        "missing_capabilities": ["pharmacophore_screening: not installed"],
        "pipeline": [{"step_id": "p", "action_type": "pharmacophore_screening"}],
    }
    executable = {
        "strategy_name": "current docking",
        "execution_status": "currently_executable",
        "pipeline": [{"step_id": "d", "action_type": "molecular_docking"}],
    }
    selected, plan, rejected = choose_executable_strategy(
        ["future pharmacophore", "current docking"], [future, executable],
        input_manifest=_manifest(),
    )
    assert selected["strategy_name"] == "current docking"
    assert plan.strategy_id == "current docking"
    assert rejected[0]["strategy_name"] == "future pharmacophore"
    assert rejected[0]["missing_capabilities"]
