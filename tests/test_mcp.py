from autovs.mcp_server import create_server


def test_mcp_server_uses_local_named_service():
    server = create_server()
    assert server.name == "autovs_tools_mcp"
