import asyncio

from autovs.mcp_server import create_server


def test_mcp_server_uses_local_named_service():
    server = create_server()
    assert server.name == "autovs_tools_mcp"


def test_mcp_tools_expose_controlled_network_annotation_and_binding_validation():
    async def inspect():
        server = create_server()
        tools = {tool.name: tool for tool in await server.list_tools()}
        assert tools["autovs_submit_step"].annotations.openWorldHint is True
        assert tools["autovs_validate_workflow"].annotations.readOnlyHint is True
        manifest = {
            "query": "a sufficiently detailed screening request",
            "library_asset": {"source": "user", "path": "library.smi", "sha256": "a" * 64},
            "target_asset": {"source": "user", "locked": True, "path": "target.pdb", "sha256": "b" * 64},
        }
        _, structured = await server.call_tool("autovs_validate_workflow", {
            "workflow": {"strategy_name": "bad", "pipeline": [
                {"step_id": "f", "action_type": "physicochemical_filtering", "description": "use ZINC"},
            ]},
            "input_manifest": manifest,
        })
        assert structured["valid"] is False
        assert "external library" in structured["error"]
    asyncio.run(inspect())
