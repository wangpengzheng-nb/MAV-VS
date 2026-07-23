from __future__ import annotations

import gzip
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TERMINAL = {"succeeded", "failed", "cancelled"}


class ToolPending(RuntimeError):
    """A long-running external tool has been submitted but is not terminal yet."""

    def __init__(self, message: str, *, state_path: Path | None = None, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.state_path = state_path
        self.payload = payload or {}


@dataclass(frozen=True)
class AF3Env:
    server_url: str
    token: str | None = None


def af3_env_available() -> tuple[bool, str]:
    if not os.environ.get("AF3_SERVER_URL"):
        return False, "AF3_SERVER_URL is not set"
    has_auth = any(os.environ.get(name) for name in (
        "AF3_TOKEN", "AF3_ACCESS_TOKEN", "AF3_LOGIN_TOKEN", "AF3_PASSWORD",
    ))
    if not has_auth and os.environ.get("AF3_AUTH_DISABLED", "").lower() not in {"1", "true", "yes"}:
        return False, "AF3 auth material is not set"
    return True, ""


def load_af3_env() -> AF3Env:
    url = os.environ.get("AF3_SERVER_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("AF3_SERVER_URL is not set")
    token = os.environ.get("AF3_ACCESS_TOKEN") or os.environ.get("AF3_TOKEN")
    return AF3Env(server_url=url, token=token)


def af3_health() -> tuple[bool, str]:
    ok, reason = af3_env_available()
    if not ok:
        return False, reason
    env = load_af3_env()
    try:
        code, body = _request(env, "GET", "/health", timeout=10)
        if code != 200:
            return False, f"AF3 /health returned HTTP {code}"
        data = json.loads(body.decode("utf-8"))
    except Exception as exc:
        return False, f"AF3 health check failed: {type(exc).__name__}: {exc}"
    if not data.get("model_loaded"):
        return False, "AF3 model_loaded=false"
    if data.get("db_accessible") is False:
        return False, "AF3 database is not accessible"
    return True, "AF3 server is reachable and model_loaded=true"


def predict_structure(
    *,
    research_path: Path,
    work_dir: Path,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = parameters or {}
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "af3_state.json"
    input_path = work_dir / "af3_input.json"
    output_dir = work_dir / "af3_results"
    env = load_af3_env()

    research = json.loads(research_path.read_text(encoding="utf-8")) if research_path.is_file() else {}
    if not input_path.is_file():
        payload = _build_af3_input(research, params)
        input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    state = _read_state(state_path)
    if not state.get("job_id"):
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        problems = _validate_af3_input(payload)
        if problems:
            raise RuntimeError("invalid AF3 input: " + "; ".join(problems))
        job = _submit_job(env, payload, name=str(params.get("name") or payload.get("name") or "autovs_af3"))
        state = {
            "job_id": job["job_id"],
            "status": job.get("status", "queued"),
            "input_json": str(input_path),
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_state(state_path, state)

    wait_seconds = float(params.get("wait_seconds", os.environ.get("AF3_WAIT_SECONDS", "0")))
    poll_interval = max(15.0, float(params.get("poll_interval", 30)))
    deadline = time.monotonic() + wait_seconds
    meta = _job_meta(env, state["job_id"])
    while meta.get("status") not in TERMINAL and wait_seconds > 0 and time.monotonic() < deadline:
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        meta = _job_meta(env, state["job_id"])
    state.update({
        "status": meta.get("status", state.get("status")),
        "result_available": meta.get("result_available"),
        "result_missing_reason": meta.get("result_missing_reason"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _write_state(state_path, state)

    status = str(meta.get("status", "unknown"))
    if status not in TERMINAL:
        raise ToolPending(
            f"AF3 job {state['job_id']} is {status}; resume task after it finishes",
            state_path=state_path,
            payload=state,
        )
    if status != "succeeded":
        raise RuntimeError(f"AF3 job {state['job_id']} ended with status={status}: {meta.get('error') or ''}")
    if meta.get("result_available") is False:
        raise RuntimeError(f"AF3 result unavailable: {meta.get('result_missing_reason')}")

    archive = _download_result(env, state["job_id"], output_dir)
    structure = _find_structure(output_dir)
    if structure is None:
        raise RuntimeError("AF3 result contains no .cif/.mmcif/.pdb structure file")
    pdb = _ensure_pdb(structure, work_dir / "af3_predicted.pdb")
    report_path = work_dir / "af3_report.json"
    report_path.write_text(json.dumps({
        "job": meta,
        "state": state,
        "archive": str(archive),
        "structure": str(structure),
        "pdb": str(pdb),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "predicted_structure": pdb,
        "target_structure": pdb,
        "af3_state": state_path,
        "af3_report": report_path,
        "af3_result_archive": archive,
        "af3_result_dir": output_dir,
        "job_id": state["job_id"],
    }


def _request(env: AF3Env, method: str, path: str, *, body: Any = None, timeout: float = 30,
             accept_zip: bool = False) -> tuple[int, bytes]:
    if not path.startswith("/"):
        path = "/" + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/zip" if accept_zip else "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if env.token:
        headers["Authorization"] = f"Bearer {env.token}"
    req = urllib.request.Request(env.server_url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _submit_job(env: AF3Env, input_json: dict[str, Any], *, name: str) -> dict[str, Any]:
    code, body = _request(env, "POST", "/api/jobs", body={
        "name": name,
        "input_json": input_json,
        "run_data_pipeline": True,
        "run_inference": True,
    }, timeout=60)
    data = json.loads(body.decode("utf-8"))
    if code != 201:
        raise RuntimeError(f"AF3 submission failed HTTP {code}: {data.get('detail', data)}")
    return data


def _job_meta(env: AF3Env, job_id: str) -> dict[str, Any]:
    code, body = _request(env, "GET", f"/api/jobs/{job_id}", timeout=30)
    data = json.loads(body.decode("utf-8"))
    if code != 200:
        raise RuntimeError(f"AF3 status failed HTTP {code}: {data.get('detail', data)}")
    return data


def _download_result(env: AF3Env, job_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{job_id}.zip"
    if not archive.is_file():
        code, body = _request(env, "GET", f"/api/jobs/{job_id}/result", timeout=300, accept_zip=True)
        if code != 200:
            raise RuntimeError(f"AF3 download failed HTTP {code}: {body[:200]!r}")
        archive.write_bytes(body)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise RuntimeError(f"unsafe AF3 zip member: {member}")
            zf.extractall(output_dir)
    else:
        out = output_dir / f"{job_id}_result"
        try:
            with gzip.open(archive, "rb") as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        except gzip.BadGzipFile as exc:
            raise RuntimeError("AF3 result archive is neither zip nor gz") from exc
    return archive


def _build_af3_input(research: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    if params.get("input_json"):
        path = Path(str(params["input_json"])).expanduser()
        return json.loads(path.read_text(encoding="utf-8"))
    sequence = str(params.get("sequence") or _sequence_from_research(research) or "").strip().replace(" ", "")
    if not sequence:
        accession = str(research.get("target_uniprot_id") or research.get("uniprot_id") or "")
        sequence = _fetch_uniprot_sequence(accession) if accession else ""
    if not sequence:
        raise RuntimeError("AF3 target_structure_prediction requires sequence, input_json, or research UniProt accession")
    name = str(params.get("name") or research.get("gene_symbol") or research.get("target_name") or "autovs_target")
    chain_id = str(params.get("chain_id", "A"))
    return {
        "name": name,
        "modelSeeds": [int(params.get("seed", 1))],
        "sequences": [{"protein": {"id": [chain_id], "sequence": sequence}}],
        "dialect": "alphafold3",
        "version": 1,
    }


def _sequence_from_research(research: dict[str, Any]) -> str:
    for key in ("protein_sequence", "target_sequence", "sequence", "fasta_sequence"):
        value = research.get(key)
        if isinstance(value, str) and value.strip():
            if value.startswith(">"):
                return "".join(line.strip() for line in value.splitlines() if not line.startswith(">"))
            return value
    identity = research.get("identity") or {}
    for key in ("protein_sequence", "sequence"):
        value = identity.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _fetch_uniprot_sequence(accession: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    return "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))


def _validate_af3_input(payload: dict[str, Any]) -> list[str]:
    required = ("name", "modelSeeds", "sequences", "dialect", "version")
    problems = [f"missing {key}" for key in required if key not in payload]
    if payload.get("dialect") != "alphafold3":
        problems.append("dialect must be alphafold3")
    seqs = payload.get("sequences")
    if not isinstance(seqs, list) or not seqs:
        problems.append("sequences must be a non-empty list")
    else:
        total = 0
        for wrapper in seqs:
            if isinstance(wrapper, dict):
                for entity in wrapper.values():
                    if isinstance(entity, dict) and isinstance(entity.get("sequence"), str):
                        total += len(entity["sequence"])
        if total > 1100:
            problems.append(f"sequence total length {total} exceeds 1100")
    return problems


def _find_structure(root: Path) -> Path | None:
    candidates = []
    for suffix in ("*.pdb", "*.cif", "*.mmcif"):
        candidates.extend(root.rglob(suffix))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (0 if p.suffix.lower() == ".pdb" else 1, len(p.as_posix())))[0]


def _ensure_pdb(source: Path, output_pdb: Path) -> Path:
    if source.suffix.lower() == ".pdb":
        if source != output_pdb:
            shutil.copyfile(source, output_pdb)
        return output_pdb
    try:
        import gemmi
        st = gemmi.read_structure(str(source))
        st.write_pdb(str(output_pdb))
        return output_pdb
    except Exception as exc:
        raise RuntimeError(f"could not convert AF3 structure to PDB: {exc}") from exc


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
