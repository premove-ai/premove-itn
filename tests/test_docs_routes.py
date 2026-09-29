"""Keep the Mintlify navigation aligned with docs.premove.dev routes."""

import json
from pathlib import Path
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

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
    assert config["favicon"] == "/premove-icon.png"
    assert "website/" in (ROOT / ".mintignore").read_text(encoding="utf-8").splitlines()
    assert config["seo"]["indexing"] == "navigable"
    assert config["seo"]["metatags"]["author"] == "Premove AI"
    assert config["seo"]["metatags"]["og:locale"] == "en_US"
    assert "end-to-end latency" in config["description"]
    assert config["navigation"]["groups"][0]["group"] == "Overview"
    hidden_root_groups = [
        group
        for group in config["navigation"]["groups"]
        if group.get("hidden") and "index" in group["pages"]
    ]
    assert hidden_root_groups == [
        {
            "group": "Premove",
            "hidden": True,
            "searchable": True,
            "pages": ["index"],
        }
    ]
    assert config["navbar"]["links"] == [
        {
            "type": "github",
            "href": "https://github.com/premove-ai",
        },
    ]
    index_content = (ROOT / "index.mdx").read_text(encoding="utf-8")
    assert "mode: center" in index_content.split("---", 2)[1]
    assert 'noindex: "true"' in index_content.split("---", 2)[1]
    assert "I got obsessed with exploring voice agents" in index_content
    assert (
        "Premove is my effort to work on those problems in the open, one by one."
        in index_content
    )
    assert "[**Premove ITN**](/itn/)" in index_content
    assert "—" not in index_content
    itn_index = (ROOT / "itn/index.md").read_text(encoding="utf-8")
    assert "Speech recognition can return spoken-form text" in itn_index
    assert "Read the build story" in itn_index
    assert 'device="auto"' in (ROOT / "itn/docs/deployment.md").read_text(
        encoding="utf-8"
    )
    assert "CUDA is not a validated v0.3.0 platform" in (
        ROOT / "itn/docs/deployment.md"
    ).read_text(encoding="utf-8")
    architecture = (ROOT / "itn/docs/internals/architecture.md").read_text(
        encoding="utf-8"
    )
    assert (
        "https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/contextual.py"
        in architecture
    )
    assert "../../../src/" not in architecture
    structured_prediction = (
        ROOT / "itn/docs/internals/structured-prediction.md"
    ).read_text(encoding="utf-8")
    assert (
        "https://github.com/premove-ai/premove-itn/blob/main/src/premove_itn/structured_loss.py"
        in structured_prediction
    )
    inference_artifact = (ROOT / "itn/docs/internals/inference-artifact.md").read_text(
        encoding="utf-8"
    )
    assert (
        "https://github.com/premove-ai/premove-itn/blob/main/docs/model-card.md"
        in inference_artifact
    )
    assert "../../../docs/" not in inference_artifact
    assert 'Documentation = "https://docs.premove.dev/itn/docs"' in (
        ROOT / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert 'href="https://docs.premove.dev/itn/docs"' in (ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    assert "#footer" in (ROOT / "style.css").read_text(encoding="utf-8")
    style_css = (ROOT / "style.css").read_text(encoding="utf-8")
    assert "font-size: 2.5rem;" in style_css
    assert 'background: url("/premove-icon.png")' in style_css
    assert "filter: invert(1);" in style_css
    assert "#theme-preference-menu-trigger::before" in style_css
    assert 'content: "☾";' in style_css
    assert 'content: "☀";' in style_css
    assert "#theme-preference-menu-item-system" in style_css
    assert "#theme-preference-menu-content" in style_css
    assert ".base-route #search-bar-entry" in style_css
    assert "margin-bottom: 1.5rem !important;" in style_css
    assert '#content > [data-as="p"]:first-child' in style_css
    assert "margin-top: 1.5rem !important;" in style_css
    assert (ROOT / "premove-icon.png").is_file()
    site_js = (ROOT / "site.js").read_text(encoding="utf-8")
    assert '"https://github.com/premove-ai/premove-itn"' in site_js
    assert '"https://github.com/premove-ai"' in site_js
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


def test_landing_page_discovery_points_to_premove_home() -> None:
    canonical = "https://premove.dev/"
    site = ROOT / "website"
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "<title>Premove: Open Source Voice Infra</title>" in html
    assert 'property="og:title" content="Premove: Open Source Voice Infra"' in html
    assert f'<link rel="canonical" href="{canonical}"' in html
    assert "noindex" not in html

    structured_data = json.loads(
        html.split('<script type="application/ld+json">', 1)[1].split("</script>", 1)[0]
    )
    website = next(
        node for node in structured_data["@graph"] if node["@type"] == "WebSite"
    )
    assert website["url"] == canonical

    sitemap = ElementTree.parse(site / "sitemap.xml")
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    assert [node.text for node in sitemap.iter(f"{namespace}loc")] == [canonical]

    robots = (site / "robots.txt").read_text(encoding="utf-8")
    parser = RobotFileParser()
    parser.parse(robots.splitlines())
    assert parser.can_fetch("Googlebot", canonical)
    assert parser.can_fetch("OAI-SearchBot", canonical)
    assert "Sitemap: https://premove.dev/sitemap.xml" in robots

    llms = (site / "llms.txt").read_text(encoding="utf-8")
    assert llms.startswith("# Premove\n")
    assert f"[Premove home]({canonical})" in llms
    assert "https://docs.premove.dev/llms.txt" in llms


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
        if page == "index":
            assert "description" not in metadata
        else:
            assert metadata.get("description"), page
        titles.append(metadata["title"])
        assert not frontmatter[2].lstrip().startswith("# "), page

    assert len(titles) == len(set(titles))
