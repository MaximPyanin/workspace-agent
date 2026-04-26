import importlib

import workplace_agent


def test_version_is_set() -> None:
    assert isinstance(workplace_agent.__version__, str)
    assert workplace_agent.__version__ != ""


def test_submodules_importable() -> None:
    for name in (
        "workplace_agent.agent",
        "workplace_agent.cli",
        "workplace_agent.config",
        "workplace_agent.llm",
        "workplace_agent.logging_setup",
        "workplace_agent.mock_api",
        "workplace_agent.tools",
    ):
        assert importlib.import_module(name) is not None
