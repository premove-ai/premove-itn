"""Keep the Mintlify navigation aligned with the planned /itn/ route tree."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROUTES = {
    "/itn/",
    "/itn/benchmarks/",
    "/itn/docs/",
    "/itn/docs/getting-started/",
    "/itn/docs/python-api/",
    "/itn/docs/cli/",
    "/itn/docs/supported-forms/",
    "/itn/docs/how-it-works/",
    "/itn/docs/deployment/",
    "/itn/docs/troubleshooting/",
    "/itn/docs/learn/what-is-inverse-text-normalization/",
    "/itn/docs/learn/itn-for-voice-agents/",
}


def _public_route(page: str) -> str:
    if page == "index":
        return "/itn/"
    if page == "docs/index":
        return "/itn/docs/"
    return f"/itn/{page}/"


def test_mintlify_public_routes_match_the_site_plan() -> None:
    config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
    pages = [
        page
        for group in config["navigation"]["groups"]
        for page in group["pages"]
        if not page.startswith("docs/internals/")
    ]

    assert len(pages) == len(set(pages))
    assert {_public_route(page) for page in pages} == PUBLIC_ROUTES
    assert all((ROOT / f"{page}.md").is_file() for page in pages)
