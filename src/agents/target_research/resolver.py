from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .client import ResearchHttpClient
from .models import (
    RequirementPriority,
    ScreeningRequirement,
    TargetIdentity,
    TargetIdentityCandidate,
    TargetIntent,
)


UNIPROT_RE = re.compile(r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b", re.I)
TARGET_PREFIX_RE = re.compile(r"(?:靶(?:点|向)(?:基因|蛋白|蛋白质)?|target(?:\s+gene|\s+protein)?)\s*[:：]?\s*([A-Za-z0-9][A-Za-z0-9_.-]{0,30})", re.I)
GENE_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,20}\b")
NOISE = {
    "PROTAC", "SMILES", "PDB", "JSON", "API", "IC50", "EC50", "ADMET", "RNA", "DNA",
    "MW", "LOGP", "SDF", "CSV", "HTVS", "VS", "MD", "FDA", "ATP", "THE", "AND",
}


def _norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


class TargetResolver:
    def __init__(self, client: ResearchHttpClient, *, model: str = "deepseek-chat"):
        self.http = client
        self.model = model

    def parse_intent(self, query: str, target_hint: str = "") -> TargetIntent:
        parsed = self._llm_intent(query, target_hint)
        if parsed is None:
            parsed = self._heuristic_intent(query, target_hint)
        if not parsed.organism_hint:
            pathogen = self._organism_from_query(query)
            if pathogen:
                parsed.organism_hint = pathogen
            else:
                parsed.organism_hint = "Homo sapiens"
                parsed.organism_assumed = True
        return parsed

    def _llm_intent(self, query: str, target_hint: str) -> TargetIntent | None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        prompt = {
            "query": query,
            "target_hint": target_hint,
            "instructions": (
                "Extract exactly one protein target and screening requirements. Do not invent an identifier. "
                "Return JSON with target_text, organism_hint, macromolecule_type, desired_effect, "
                "multiple_targets_detected, excluded_targets and requirements. Each requirement has "
                "category, priority (must/prefer/avoid), original_text, normalized_key, operator, value, unit."
            ),
        }
        try:
            client = OpenAI(api_key=api_key, base_url=f"{os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com').rstrip('/')}/v1")
            response = client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=1200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You extract biomedical search intent into strict JSON. Facts will be verified by APIs."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            )
            raw = json.loads(response.choices[0].message.content or "{}")
            raw["raw_query"] = query
            raw.setdefault("target_text", target_hint)
            raw.setdefault("organism_hint", "")
            raw.setdefault("macromolecule_type", "protein")
            raw.setdefault("desired_effect", "unknown")
            raw.setdefault("requirements", [])
            raw.setdefault("excluded_targets", [])
            raw.setdefault("multiple_targets_detected", False)
            return TargetIntent.model_validate(raw)
        except Exception:
            return None

    @staticmethod
    def _heuristic_intent(query: str, target_hint: str) -> TargetIntent:
        target = target_hint.strip()
        if not target:
            prefix = TARGET_PREFIX_RE.search(query)
            if prefix:
                target = prefix.group(1)
        if not target:
            accession = UNIPROT_RE.search(query)
            if accession:
                target = accession.group(0).upper()
        if not target:
            for token in GENE_TOKEN_RE.findall(query):
                if token.upper() not in NOISE and (token.isupper() or any(ch.isdigit() for ch in token)):
                    target = token
                    break

        # --- Post-process: replace organism abbreviations with functional descriptions ---
        # When the query mentions a viral/bacterial organism AND a functional protein
        # description (e.g. "SARS-CoV-2 main protease"), the gene-token heuristic
        # often picks the organism abbreviation ("SARS" or "SARS-CoV-2") instead of
        # "main protease". Detect this and extract the functional phrase instead.
        _organism_indicators = {
            "sars", "cov", "hcv", "hbv", "hiv", "ebov", "denv", "zikv", "flu", "iav",
            "mtb", "ecoli", "pberghei", "pfalciparum", "tbrucei", "tcrustulum",
        }
        _target_words = set(re.split(r"[-_\s]+", target.lower()))
        if target and (_target_words & _organism_indicators or target.lower() in _organism_indicators):
            # Look for a functional protein phrase: adjective + functional noun.
            _func_phrase = re.search(
                r"\b(main|papain[ -]?like|3cl|RNA[ -]dependent[ -]RNA|spike|envelope|"
                r"nucleocapsid|membrane|non[ -]?structural)\s+"
                r"(protease|proteinase|polymerase|glycoprotein|protein|helicase|nuclease)\b",
                query, re.I,
            )
            if _func_phrase:
                target = _func_phrase.group(0)
            else:
                # Standalone functional keywords commonly used for viral proteins.
                _standalone_func = re.search(
                    r"\b(3CLpro|3CL|Mpro|PLpro|RdRp|NS5B|NS3|NS5A|gp120|gp41)\b",
                    query, re.I,
                )
                if _standalone_func:
                    target = _standalone_func.group(0)
                else:
                    # Fallback: any multi-word English protein description
                    _simple_func = re.search(
                        r"\b([A-Za-z][a-z]+(?:\s+[a-z]+){1,3})\s+(protease|kinase|polymerase|"
                        r"receptor|channel|transporter|helicase|nuclease|glycoprotein)\b",
                        query, re.I,
                    )
                    if _simple_func:
                        target = _simple_func.group(0)

        lower = query.lower()
        effect = "unknown"
        if any(item in lower for item in ("抑制", "inhibit", "antagonist", "拮抗")):
            effect = "inhibit"
        elif any(item in lower for item in ("降解", "degrad", "protac")):
            effect = "degrade"
        elif any(item in lower for item in ("激活", "activate", "agonist", "激动")):
            effect = "activate"
        elif any(item in lower for item in ("结合", "bind")):
            effect = "bind"

        mol_type = "protein"
        if re.search(r"\bRNA\b|核糖核酸", query, re.I):
            mol_type = "rna"
        elif re.search(r"\bDNA\b|脱氧核糖核酸", query, re.I):
            mol_type = "dna"

        requirements: list[ScreeningRequirement] = []
        excluded: list[str] = []
        fragments = [item.strip() for item in re.split(r"[。；;\n]", query) if item.strip()]
        for fragment in fragments:
            priority = RequirementPriority.PREFER
            category = "other"
            if re.search(r"不要|禁止|排除|避免|not\s+|avoid|exclude", fragment, re.I):
                priority = RequirementPriority.AVOID
                category = "off_target" if re.search(r"同源|靶点|target", fragment, re.I) else "other"
                excluded.extend(re.findall(r"\b[A-Z][A-Z0-9-]{1,15}\b", fragment))
            elif re.search(r"必须|要求|需要|must|required", fragment, re.I):
                priority = RequirementPriority.MUST
            if re.search(r"选择性|selectiv", fragment, re.I):
                category = "selectivity"
            elif re.search(r"MW|分子量|LogP|理化|physicochemical", fragment, re.I):
                category = "physicochemical"
            elif re.search(r"共价|非共价|抑制|激动|拮抗|降解", fragment, re.I):
                category = "mechanism"
            if fragment != query or category != "other" or priority != RequirementPriority.PREFER:
                requirements.append(ScreeningRequirement(
                    category=category, priority=priority, original_text=fragment,
                ))

        target_mentions = set(re.findall(r"\b[A-Z][A-Z0-9-]{2,15}\b", query)) - NOISE
        target_mentions -= set(excluded)
        multiple = len(target_mentions) > 1 and bool(re.search(r"双靶|多靶|同时|\band\b|[+&]", query, re.I))
        return TargetIntent(
            raw_query=query, target_text=target, macromolecule_type=mol_type,
            desired_effect=effect, requirements=requirements, excluded_targets=sorted(set(excluded)),
            multiple_targets_detected=multiple,
        )

    @staticmethod
    def _organism_from_query(query: str) -> str:
        patterns = [
            (r"结核|M\.\s*tuberculosis|Mycobacterium tuberculosis", "Mycobacterium tuberculosis"),
            (r"SARS[- ]?CoV[- ]?2|新冠", "Severe acute respiratory syndrome coronavirus 2"),
            (r"大肠杆菌|E\.\s*coli|Escherichia coli", "Escherichia coli"),
            (r"疟原虫|Plasmodium falciparum", "Plasmodium falciparum"),
            (r"小鼠|mouse|Mus musculus", "Mus musculus"),
            (r"人源|人类|human|Homo sapiens", "Homo sapiens"),
            (r"\bHIV\b|艾滋", "Human immunodeficiency virus 1"),
        ]
        for pattern, organism in patterns:
            if re.search(pattern, query, re.I):
                return organism
        return ""

    def resolve(self, intent: TargetIntent, *, selected_accession: str = "") -> dict[str, Any]:
        if intent.macromolecule_type != "protein":
            return {"status": "unsupported", "reason": "v1 supports single protein targets only", "intent": intent}
        if intent.multiple_targets_detected:
            return {"status": "unsupported", "reason": "v1 supports one primary protein target per task", "intent": intent}
        if not intent.target_text:
            return {"status": "needs_confirmation", "reason": "no target could be extracted", "intent": intent, "candidates": []}

        candidates, sources = self._uniprot_candidates(intent)
        if selected_accession:
            selected = next((item for item in candidates if item.uniprot_accession.upper() == selected_accession.upper()), None)
            if not selected:
                return {
                    "status": "invalid_selection", "reason": "selected UniProt accession is not a candidate for this query",
                    "intent": intent, "candidates": candidates, "source_results": sources,
                }
            return {"status": "resolved", "intent": intent, "identity": self._identity(selected, intent),
                    "candidates": candidates, "source_results": sources}

        if not candidates:
            return {"status": "needs_confirmation", "reason": "UniProt returned no verifiable candidates",
                    "intent": intent, "candidates": [], "source_results": sources}
        margin = candidates[0].confidence - (candidates[1].confidence if len(candidates) > 1 else 0.0)
        if candidates[0].confidence >= 0.90 and margin >= 0.08:
            return {"status": "resolved", "intent": intent, "identity": self._identity(candidates[0], intent),
                    "candidates": candidates[:5], "source_results": sources}
        return {"status": "needs_confirmation", "reason": "multiple plausible target identities",
                "intent": intent, "candidates": candidates[:5], "source_results": sources}

    def _uniprot_candidates(self, intent: TargetIntent) -> tuple[list[TargetIdentityCandidate], list[Any]]:
        target = intent.target_text.strip()
        accession_match = UNIPROT_RE.fullmatch(target)
        sources = []
        if accession_match:
            url = f"https://rest.uniprot.org/uniprotkb/{target.upper()}.json"
            data, source = self.http.get_json("uniprot_identity", url)
            sources.append(source)
            records = [data] if data else []
        else:
            organism = intent.organism_hint.replace('"', "")
            query = f'({{target_query}}) AND (organism_name:"{organism}")'
            exact = f"gene_exact:{target}"
            params = {"query": query.format(target_query=exact), "format": "json", "size": 10}
            data, source = self.http.get_json("uniprot_identity", "https://rest.uniprot.org/uniprotkb/search", params=params)
            sources.append(source)
            records = list((data or {}).get("results", []))
            broad = f'(gene:{target} OR protein_name:"{target}")'
            params["query"] = query.format(target_query=broad)
            broad_data, broad_source = self.http.get_json(
                "uniprot_identity", "https://rest.uniprot.org/uniprotkb/search", params=params,
            )
            sources.append(broad_source)
            records.extend((broad_data or {}).get("results", []))
            if not records or " " in target:
                params["query"] = query.format(target_query=f'"{target}"')
                text_data, text_source = self.http.get_json(
                    "uniprot_identity", "https://rest.uniprot.org/uniprotkb/search", params=params,
                )
                sources.append(text_source)
                records.extend((text_data or {}).get("results", []))

        # --- Fallback: organism-only search for viral / microbial targets ---
        # When the target_text is a functional description (e.g. "main protease",
        # "spike glycoprotein") that does not match any gene or protein name, search
        # all reviewed proteins from the organism and filter by functional relevance.
        if not records and intent.organism_hint.lower() not in {"homo sapiens", ""}:
            fallback_query = f'(organism_name:"{organism}") AND (reviewed:true)'
            fb_params: dict[str, Any] = {"query": fallback_query, "format": "json", "size": 50}
            fb_data, fb_source = self.http.get_json(
                "uniprot_identity", "https://rest.uniprot.org/uniprotkb/search", params=fb_params,
            )
            sources.append(fb_source)
            fb_records = list((fb_data or {}).get("results", []))
            if fb_records:
                # Only keep records whose processed chains match the functional intent.
                target_lower = intent.target_text.lower()
                functional_roots = (
                    "proteas", "proteinase", "kinas", "receptor", "polymeras",
                    "helicas", "nuclease", "spike", "envelope", "membrane",
                    "nucleocapsid", "capsid", "integrase", "transcriptase",
                )
                filtered: list[dict[str, Any]] = []
                for rec in fb_records:
                    if not isinstance(rec, dict):
                        continue
                    features = rec.get("features", [])
                    chains = [f.get("description", "") for f in features
                              if f.get("type") in {"Chain", "Peptide"} and f.get("description")]
                    _fb_target_has_root = any(root in target_lower for root in functional_roots)
                    if _fb_target_has_root and any(
                        any(root in chain.lower() for root in functional_roots)
                        for chain in chains
                    ):
                        filtered.append(rec)
                    # Also check if the target text appears anywhere in the protein name.
                    desc = rec.get("proteinDescription") or {}
                    full_name = (((desc.get("recommendedName") or {}).get("fullName") or {}).get("value") or "")
                    if target_lower in full_name.lower():
                        filtered.append(rec)
                if filtered:
                    records = filtered
                    if fb_source.status == "success":
                        fb_source.status = "success"  # keep as success

        if not records:
            for source in sources:
                if source.status == "success":
                    source.status = "empty"

        candidates = [self._candidate(record, intent) for record in records if isinstance(record, dict)]
        candidates = [item for item in candidates if item.uniprot_accession]
        candidates.sort(key=lambda item: (-item.confidence, not item.reviewed, item.uniprot_accession))
        deduped: dict[tuple[str, int | None], TargetIdentityCandidate] = {}
        for item in candidates:
            group = (_norm(item.canonical_gene_symbol) or item.uniprot_accession, item.taxonomy_id)
            deduped.setdefault(group, item)
        return list(deduped.values())[:10], sources

    @staticmethod
    def _candidate(record: dict[str, Any], intent: TargetIntent) -> TargetIdentityCandidate:
        genes = record.get("genes") or []
        primary = ""
        aliases: list[str] = []
        for gene in genes:
            name = (gene.get("geneName") or {}).get("value", "")
            if name and not primary:
                primary = name
            if name:
                aliases.append(name)
            aliases.extend(item.get("value", "") for item in gene.get("synonyms", []) if item.get("value"))
        desc = record.get("proteinDescription") or {}
        protein_name = (((desc.get("recommendedName") or {}).get("fullName") or {}).get("value") or
                        ((desc.get("submissionNames") or [{}])[0].get("fullName") or {}).get("value", ""))
        for section in (desc.get("alternativeNames") or []):
            value = (section.get("fullName") or {}).get("value", "")
            if value:
                aliases.append(value)
            aliases.extend(item.get("value", "") for item in section.get("shortNames", []) if item.get("value"))
        processed_names = [
            feature.get("description", "") for feature in record.get("features", [])
            if feature.get("type") in {"Chain", "Peptide"} and feature.get("description")
        ]
        target_lower = intent.target_text.lower()
        # "proteas" must match both "protease" (papain-like protease) and
        # "proteinase" (3C-like proteinase / main protease).
        functional_roots = (
            "proteas", "proteinase", "kinas", "receptor", "polymeras", "helicas", "nuclease",
        )
        # Collect ALL matching processed chains, not just the first.
        # The target must contain at least one functional keyword AND the chain
        # name must contain at least one functional keyword (they do not need to
        # be the *same* keyword — e.g. "main protease" in the query should also
        # match "3C-like proteinase" chains).
        matched_chains: list[str] = []
        _target_has_root = any(root in target_lower for root in functional_roots)
        for name in processed_names:
            name_lower = name.lower()
            if _target_has_root and any(root in name_lower for root in functional_roots):
                matched_chains.append(name)
        matched_processed = ""
        # When the user asks for the *main* protease, prefer 3C-like / 3CL / nsp5
        # over papain-like / PLpro / nsp3.
        if matched_chains:
            if "main" in target_lower or "mpro" in target_lower or "3cl" in target_lower:
                preferred = [c for c in matched_chains
                             if re.search(r"3C[ -]?like|3CL|nsp5\b", c, re.I)]
                if preferred:
                    matched_chains = preferred
            matched_processed = matched_chains[0]
            if protein_name:
                aliases.append(protein_name)
            # Keep polyprotein chain names as aliases; do NOT lose the
            # canonical gene symbol assigned by UniProt.
            for chain in processed_names:
                if chain not in aliases:
                    aliases.append(chain)
            # Use the matched chain as the display protein name only when
            # the UniProt recommended name is a generic polyprotein label.
            is_polyprotein = bool(
                re.search(r"polyprotein|replicase\s+polyprotein|ORF1[ab]\s+polyprotein",
                          protein_name, re.I)
            )
            if is_polyprotein:
                protein_name = matched_processed
            nsp = re.search(r"\b(nsp\d+)\b", matched_processed, re.I)
            if nsp:
                nsp_label = nsp.group(1).lower()
                if nsp_label not in aliases:
                    aliases.append(nsp_label)
                # Only fall back to nspX when UniProt provides no gene symbol.
                if not primary:
                    primary = nsp_label
            # Ensure the matched chain is recorded in aliases.
            if matched_processed not in aliases:
                aliases.append(matched_processed)
        organism = record.get("organism") or {}
        organism_name = organism.get("scientificName", "")
        taxonomy_id = organism.get("taxonId")
        accession = record.get("primaryAccession", "")
        query_norm = _norm(intent.target_text)
        evidence: list[str] = []
        score = 0.0
        if UNIPROT_RE.fullmatch(intent.target_text) and accession.upper() == intent.target_text.upper():
            score, evidence = 0.99, ["exact UniProt accession"]
        elif query_norm and query_norm == _norm(primary):
            score, evidence = 0.94, ["exact canonical gene symbol"]
        elif query_norm and query_norm in {_norm(item) for item in aliases}:
            score, evidence = 0.90, ["exact UniProt alias"]
        elif query_norm and query_norm in _norm(protein_name):
            score, evidence = 0.82, ["protein name match"]
        elif matched_processed:
            score, evidence = 0.90, ["processed protein chain function match"]
        context_terms = {
            "receptor": ("receptor", "受体"),
            "kinase": ("kinase", "激酶"),
            "protease": ("protease", "蛋白酶"),
            "phosphatase": ("phosphatase", "磷酸酶"),
            "transporter": ("transporter", "转运"),
        }
        query_lower = intent.raw_query.lower()
        protein_lower = protein_name.lower()
        for canonical, terms in context_terms.items():
            if any(term.lower() in query_lower for term in terms):
                protein_class_match = canonical in protein_lower or (
                    canonical == "protease" and ("proteas" in protein_lower or "proteinase" in protein_lower)
                )
                if protein_class_match:
                    score += 0.04
                    evidence.append(f"{canonical} context match")
                else:
                    score = max(0.0, score - 0.08)
                    evidence.append(f"{canonical} context mismatch")
                break
        if organism_name.lower() == intent.organism_hint.lower():
            score += 0.04
            evidence.append("organism match" + (" (assumed)" if intent.organism_assumed else ""))
        reviewed = record.get("entryType", "").lower().startswith("uniprotkb reviewed")
        if reviewed:
            score += 0.02
            evidence.append("reviewed UniProt entry")
        return TargetIdentityCandidate(
            canonical_gene_symbol=primary, protein_name=protein_name, uniprot_accession=accession,
            organism_name=organism_name, taxonomy_id=taxonomy_id,
            aliases=sorted(set(item for item in aliases if item and item != primary)), reviewed=reviewed,
            confidence=min(round(score, 4), 1.0), match_evidence=evidence,
        )

    @staticmethod
    def _identity(candidate: TargetIdentityCandidate, intent: TargetIntent) -> TargetIdentity:
        return TargetIdentity(**candidate.model_dump(), organism_assumed=intent.organism_assumed)
