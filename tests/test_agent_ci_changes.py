"""Changed-surface routing keeps required CI checks proportional."""

from scripts.agent.ci_changes import classify


def test_policy_only_needs_no_expensive_jobs() -> None:
    assert classify(
        [
            "AGENTS.md",
            "CHANGELOG.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/bug_report.md",
        ]
    ) == (False, False, False, False, False)


def test_harness_only_needs_harness_job() -> None:
    assert classify([tests_path := "tests/test_agent_ci_changes.py"]) == (
        True,
        False,
        False,
        False,
        False,
    )
    assert tests_path.endswith(".py")
    assert classify(
        [
            ".codex/config.toml",
            "scripts/agent/start_change.py",
            "tests/test_agent_check.py",
            "tests/test_rtk_hook.py",
            "tests/test_codegraph_mcp.py",
        ]
    ) == (True, False, False, False, False)


def test_docs_use_lightweight_docs_job() -> None:
    assert classify(
        [
            "README.md",
            "docs/agent-harness.md",
            "itn/docs/deployment.md",
            "docs.json",
            "style.css",
            "site.js",
            "tests/test_docs_routes.py",
        ]
    ) == (False, True, False, False, False)


def test_python_runtime_does_not_force_rust_or_package_jobs() -> None:
    assert classify(["src/premove_itn/contextual.py"]) == (
        False,
        False,
        True,
        False,
        False,
    )
    assert classify(["tests/test_candidate_scorer.py"]) == (
        False,
        False,
        True,
        False,
        False,
    )


def test_rust_changes_cover_cross_language_and_package_behavior() -> None:
    assert classify(["rust/src/lib.rs"]) == (False, False, True, True, True)
    assert classify(["rust/Cargo.toml"]) == (False, False, True, True, True)


def test_dependency_routes_match_consumers() -> None:
    assert classify(["uv.lock"]) == (True, True, True, False, True)
    assert classify(["pyproject.toml"]) == (False, True, True, False, True)
    assert classify(["uv.toml"]) == (True, True, True, True, True)


def test_workflow_and_unknown_paths_require_all_jobs() -> None:
    assert classify(["scripts/agent/ci_changes.py"]) == (True, True, True, True, True)
    assert classify([".github/workflows/ci.yml"]) == (True, True, True, True, True)
    assert classify(["new-surface/tool.py"]) == (True, True, True, True, True)
    assert classify(["AGENTS.md", "src/premove_itn/contextual.py"]) == (
        False,
        False,
        True,
        False,
        False,
    )
