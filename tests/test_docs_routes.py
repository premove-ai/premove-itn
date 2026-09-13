"""Keep the Mintlify navigation aligned with docs.premove.dev routes."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROUTES = {
    "/",
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
        return "/"
    if page == "itn/index":
        return "/itn/"
    if page == "itn/docs/index":
        return "/itn/docs/"
    return f"/{page}/"


def _navigable_pages(config: dict) -> list[str]:
    return [
        page
        for tab in config["navigation"]["tabs"]
        for group in tab.get("groups", [])
        for page in group["pages"]
    ] + [page for tab in config["navigation"]["tabs"] for page in tab.get("pages", [])]


def test_mintlify_public_routes_match_the_site_plan() -> None:
    config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
    assert config["seo"]["metatags"]["canonical"] == "https://docs.premove.dev"
    assert [tab["tab"] for tab in config["navigation"]["tabs"]] == [
        "Premove",
        "Premove ITN",
    ]
    pages = [
        page
        for page in _navigable_pages(config)
        if not page.startswith("itn/docs/internals/")
    ]

    assert len(pages) == len(set(pages))
    assert {_public_route(page) for page in pages} == PUBLIC_ROUTES
    assert all((ROOT / f"{page}.md").is_file() for page in pages)


def test_navigable_pages_have_search_and_llm_metadata() -> None:
    config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
    pages = _navigable_pages(config)
    titles = []

    for page in pages:
        content = (ROOT / f"{page}.md").read_text(encoding="utf-8")
        frontmatter = content.split("---", 2)
        assert len(frontmatter) == 3 and frontmatter[0] == "", page
        metadata = dict(
            line.split(": ", 1)
            for line in frontmatter[1].strip().splitlines()
            if ": " in line
        )
        assert metadata.get("title"), page
        assert metadata.get("description"), page
        titles.append(metadata["title"])
        assert not frontmatter[2].lstrip().startswith("# "), page

    assert len(titles) == len(set(titles))
