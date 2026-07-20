from autovs.config import Settings
from autovs.pipeline import PipelineService
from autovs.schemas import TaskRequest


def test_pipeline_failure_identifies_stage_and_indexes_diagnostics(tmp_path):
    protein = tmp_path / "invalid.pdb"; protein.write_text("HEADER INVALID\nEND\n")
    library = tmp_path / "library.smi"; library.write_text("CCO ethanol\n")
    settings = Settings(raw={
        "service": {"database": str(tmp_path / "state.sqlite3"), "task_root": str(tmp_path / "tasks"),
                    "host": "127.0.0.1", "port": 8765},
        "executables": {}, "limits": {}, "environments": {}, "containers": {},
    }, config_path=tmp_path / "tools.toml")
    result = PipelineService(settings).run_sync(TaskRequest(
        query="a sufficiently long screening request", protein_path=str(protein), library_path=str(library),
    ), use_llm_planning=False)
    assert result["status"] == "failed"
    failed = next(item for item in result["progress"] if item["status"] == "failed")
    assert failed["phase_id"] == "input_validation"
    assert "no ATOM records" in failed["error"]
    assert any(item["status"] == "skipped" and "上游阶段失败" in item["message"]
               for item in result["progress"])
    artifact_names = {item["name"] for item in result["artifacts"]}
    assert {"failure_diagnostic", "pipeline_error", "failure_report_md", "failure_report_html"} <= artifact_names
