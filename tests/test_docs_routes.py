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
    return [page for group in config["navigation"]["groups"] for page in group["pages"]]


def _page_file(page: str) -> Path:
    return ROOT / ("index.mdx" if page == "index" else f"{page}.md")


def test_mintlify_public_routes_match_the_site_plan() -> None:
    config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
    assert config["seo"]["metatags"]["canonical"] == "https://docs.premove.dev"
    assert config["navigation"]["groups"][0]["group"] == "Overview"
    assert config["navbar"]["links"] == [
        {
            "type": "github",
            "href": "https://github.com/premove-ai",
        },
        {
            "label": "How I Got Here",
            "href": "https://www.aryamantodkar.com/blog/how-i-built-an-open-source-itn-model-that-beat-nvidia-thutmose-on-voice-agent-transcripts/",
        },
    ]
    index_content = (ROOT / "index.mdx").read_text(encoding="utf-8")
    assert "mode: center" in index_content.split("---", 2)[1]
    assert "I started building voice agents because I was tired" in index_content
    assert "[**Premove ITN**](/itn/)" in index_content
    assert "—" not in index_content
    assert "#footer" in (ROOT / "style.css").read_text(encoding="utf-8")
    style_css = (ROOT / "style.css").read_text(encoding="utf-8")
    assert "font-size: 2.5rem;" in style_css
    assert ".base-route #header p" in style_css
    assert ".base-route #header > div:last-child" in style_css
    assert 'background: url("/premove-icon.png")' in style_css
    assert "filter: brightness(0) invert(1);" in style_css
    assert "#theme-preference-menu-trigger::before" in style_css
    assert 'content: "☾";' in style_css
    assert 'content: "☀";' in style_css
    assert "#theme-preference-menu-item-system" in style_css
    assert "#theme-preference-menu-content" in style_css
    assert ".base-route #search-bar-entry" in style_css
    assert 'content: "↗";' in style_css
    assert "margin-bottom: 1.5rem !important;" in style_css
    assert '#content > [data-as="p"]:first-child' in style_css
    assert "margin-top: 1.5rem !important;" in style_css
    assert (ROOT / "premove-icon.png").is_file()
    site_js = (ROOT / "site.js").read_text(encoding="utf-8")
    assert '"https://github.com/premove-ai/premove-itn"' in site_js
    assert '"https://github.com/premove-ai"' in site_js
    assert 'const blogLabel = "How I Got Here";' in site_js
    assert "link.hidden = !isItn;" in site_js
    assert "document.querySelectorAll('#content a[href^=\"/itn\"]')" in site_js
    assert 'link.target = "_blank";' in site_js
    assert 'link.rel = "noopener noreferrer";' in site_js
    assert 'document.body.classList.toggle("base-route", !isItn);' in site_js
    assert 'localStorage.setItem("isDarkMode", nextTheme);' in site_js
    assert "const replacement = trigger.cloneNode(false);" in site_js
    assert "trigger.replaceWith(replacement);" in site_js
    assert "global" not in config["navigation"]
    pages = [
        page
        for page in _navigable_pages(config)
        if not page.startswith("itn/docs/internals/")
    ]

    assert len(pages) == len(set(pages))
    assert {"/", *(_public_route(page) for page in pages)} == PUBLIC_ROUTES
    assert all(_page_file(page).is_file() for page in pages)


def test_navigable_pages_have_search_and_llm_metadata() -> None:
    config = json.loads((ROOT / "docs.json").read_text(encoding="utf-8"))
    pages = _navigable_pages(config)
    titles = []

    for page in pages:
        content = _page_file(page).read_text(encoding="utf-8")
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
