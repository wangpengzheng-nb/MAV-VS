import json

import httpx
import pytest
from fastapi.testclient import TestClient

from src.agents.strategy_generator import StrategyGeneratorAgent
from src.agents.target_research.client import ResearchHttpClient
from src.agents.target_research.models import (
    ChemblActivity,
    ResearchSourceResult,
    TargetIdentity,
    TargetIntent,
    VerifiedStructure,
)
from src.agents.target_research.resolver import TargetResolver
from src.agents.target_research.service import TargetResearchService
from web_app.server import app
import web_app.server as server_module


def source(name="uniprot_identity", status="success"):
    return ResearchSourceResult(source=name, status=status)


def uniprot_record(accession, gene, protein, organism="Homo sapiens", taxon=9606, aliases=()):
    return {
        "primaryAccession": accession,
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "genes": [{"geneName": {"value": gene},
                   "synonyms": [{"value": item} for item in aliases]}],
        "proteinDescription": {"recommendedName": {"fullName": {"value": protein}}},
        "organism": {"scientificName": organism, "taxonId": taxon},
    }


@pytest.mark.parametrize("query,hint,expected_target,expected_effect", [
    ("为人源 EGFR 寻找高选择性非共价抑制剂", "", "EGFR", "inhibit"),
    ("寻找 BCL-2 抑制剂，不要作用于 BCLXL", "BCL-2", "BCL-2", "inhibit"),
    ("针对结核分枝杆菌 inhA 进行抑制剂筛选", "inhA", "inhA", "inhibit"),
    ("为 PTGER2 寻找激动剂", "PTGER2", "PTGER2", "activate"),
])
def test_intent_fallback_preserves_target_and_requirements(monkeypatch, query, hint, expected_target, expected_effect):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    intent = TargetResolver(ResearchHttpClient()).parse_intent(query, hint)
    assert intent.target_text == expected_target
    assert intent.desired_effect == expected_effect
    assert intent.raw_query == query
    assert intent.organism_hint


def test_uniprot_identity_uses_alias_organism_and_margin(monkeypatch):
    resolver = TargetResolver(ResearchHttpClient())
    payload = {"results": [
        uniprot_record("P43116", "PTGER2", "Prostaglandin E2 receptor EP2", aliases=("EP2",)),
        uniprot_record("P35408", "PTGER4", "Prostaglandin E2 receptor EP4", aliases=("EP4",)),
    ]}
    monkeypatch.setattr(resolver.http, "get_json", lambda *args, **kwargs: (payload, source()))
    intent = TargetIntent(raw_query="寻找人源 EP2 激动剂", target_text="EP2", organism_hint="Homo sapiens")
    result = resolver.resolve(intent)
    assert result["status"] == "resolved"
    assert result["identity"].uniprot_accession == "P43116"
    assert result["identity"].canonical_gene_symbol == "PTGER2"


def test_ambiguous_candidates_require_confirmation_and_reject_tampering(monkeypatch):
    resolver = TargetResolver(ResearchHttpClient())
    payload = {"results": [
        uniprot_record("P11111", "GENEA", "Shared receptor", aliases=("SHARED",)),
        uniprot_record("P22222", "GENEB", "Shared receptor", aliases=("SHARED",)),
    ]}
    monkeypatch.setattr(resolver.http, "get_json", lambda *args, **kwargs: (payload, source()))
    intent = TargetIntent(raw_query="寻找 SHARED 抑制剂", target_text="SHARED", organism_hint="Homo sapiens")
    result = resolver.resolve(intent)
    assert result["status"] == "needs_confirmation"
    assert resolver.resolve(intent, selected_accession="P22222")["status"] == "resolved"
    assert resolver.resolve(intent, selected_accession="P99999")["status"] == "invalid_selection"


def test_non_protein_and_multi_target_are_unsupported():
    resolver = TargetResolver(ResearchHttpClient())
    rna = TargetIntent(raw_query="筛选RNA配体", target_text="RNA", macromolecule_type="rna")
    multi = TargetIntent(raw_query="同时抑制 EGFR 和 KRAS", target_text="EGFR", multiple_targets_detected=True)
    assert resolver.resolve(rna)["status"] == "unsupported"
    assert resolver.resolve(multi)["status"] == "unsupported"


def test_report_without_holo_is_degraded_but_strategy_gets_prediction(monkeypatch):
    service = TargetResearchService()
    intent = TargetIntent(raw_query="寻找人源 EGFR 抑制剂", target_text="EGFR", organism_hint="Homo sapiens")
    identity = TargetIdentity(
        canonical_gene_symbol="EGFR", protein_name="Epidermal growth factor receptor",
        uniprot_accession="P00533", organism_name="Homo sapiens", taxonomy_id=9606,
        confidence=0.99, reviewed=True, match_evidence=["exact canonical gene symbol"],
    )
    monkeypatch.setattr(service.resolver, "parse_intent", lambda *args, **kwargs: intent)
    monkeypatch.setattr(service.resolver, "resolve", lambda *args, **kwargs: {
        "status": "resolved", "intent": intent, "identity": identity,
        "source_results": [source()],
    })
    monkeypatch.setattr(service, "_fetch_structures", lambda identity: (
        [VerifiedStructure(pdb_id="1ABC", has_ligand=False, uniprot_mapped=True)],
        [source("rcsb_search")],
    ))
    monkeypatch.setattr(service, "_fetch_chembl", lambda identity: ([], [source("chembl_target", "empty")]))
    monkeypatch.setattr(service, "_fetch_pubmed", lambda identity, intent: ([], [source("pubmed_search", "empty")]))
    monkeypatch.setattr(service, "_fetch_trials", lambda identity: ([], source("clinical_trials", "empty")))
    monkeypatch.setattr(service, "_summarize", lambda *args: "")
    report = service.research(intent.raw_query)
    assert report["status"] == "degraded"
    assert report["target_name"] == "Epidermal growth factor receptor"
    assert report["structure_readiness"]["experimental_holo_available"] is False
    assert report["structure_readiness"]["predicted_structure_required"] is True
    strategies = StrategyGeneratorAgent._ensure_structure_prediction(
        [{"strategy_name": "prediction route", "pipeline": []}], True,
    )
    assert strategies[0]["pipeline"][0]["action_type"] == "target_structure_prediction"


def test_http_client_retries_429_and_records_snapshot(tmp_path, monkeypatch):
    responses = [
        httpx.Response(429, text="rate limited", headers={"Retry-After": "0"}, request=httpx.Request("GET", "https://example.test/data")),
        httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", "https://example.test/data")),
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, params=None): return responses.pop(0)

    monkeypatch.setattr(httpx, "Client", FakeClient)
    data, result = ResearchHttpClient(tmp_path, attempts=3).get_json("fixture", "https://example.test/data")
    assert data == {"ok": True}
    assert result.status == "success"
    assert result.attempts == 2
    assert len(result.snapshot_paths) == 2


@pytest.mark.parametrize("response,status,error", [
    (httpx.Response(200, text="not-json", request=httpx.Request("GET", "https://example.test/data")), "invalid", "invalid_json"),
    (httpx.Response(400, text="bad query", request=httpx.Request("GET", "https://example.test/data")), "invalid", "http_400"),
])
def test_http_client_surfaces_invalid_responses(monkeypatch, response, status, error):
    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, params=None): return response
    monkeypatch.setattr(httpx, "Client", FakeClient)
    data, result = ResearchHttpClient(attempts=3).get_json("fixture", "https://example.test/data")
    assert data is None
    assert result.status == status
    assert result.error_type == error
    assert result.attempts == 1


def test_target_resolution_web_contract(monkeypatch):
    candidate = {
        "canonical_gene_symbol": "EGFR", "protein_name": "Epidermal growth factor receptor",
        "uniprot_accession": "P00533", "organism_name": "Homo sapiens", "taxonomy_id": 9606,
        "aliases": [], "reviewed": True, "confidence": 0.99, "match_evidence": ["exact gene"],
    }
    monkeypatch.setattr(server_module.TargetScoutAgent, "resolve_target", lambda self, *args, **kwargs: {
        "status": "needs_confirmation", "reason": "ambiguous", "intent": {
            "raw_query": args[0], "target_text": "EGFR", "organism_hint": "Homo sapiens",
            "organism_assumed": True, "macromolecule_type": "protein", "desired_effect": "inhibit",
            "requirements": [], "excluded_targets": [], "multiple_targets_detected": False,
        }, "candidates": [candidate],
    })
    response = TestClient(app).post("/api/targets/resolve", json={"query": "寻找一个人源 EGFR 抑制剂"})
    assert response.status_code == 200
    assert response.json()["status"] == "needs_confirmation"
    assert response.json()["candidates"][0]["uniprot_accession"] == "P00533"
