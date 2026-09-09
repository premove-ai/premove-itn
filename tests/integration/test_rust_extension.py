from premove_itn import _rust


def test_native_build_metadata_identifies_benchmark_artifact() -> None:
    info = _rust.build_info()
    assert info["profile"] in {"debug", "release"}
    assert isinstance(info["debug_assertions"], bool)
    assert info["target"]
    assert info["rustc_version"].startswith("rustc ")
    assert info["crate_version"] == "0.1.0"
    assert "text-processing-rs" in info["text_processing_rs_revision"]


def test_python_imports_production_rust_realizer() -> None:
    assert _rust.realize("TIME", "four thirty") == "04:30"
    assert _rust.realize("MONEY", "two dollars and fifty cents") == "$2.5"
    assert _rust.realize("DECIMAL", "one point oh five") == "1.05"
    assert _rust.realize("MEASUREMENT", "one point five million meters") == "1500000 m"
    assert _rust.realize("ORDINAL", "the eighth") == "8th"
    assert _rust.realize("PHONE", "three two sil nine") == "329"
    assert _rust.realize_options("CARDINAL", "two") == ["2"]
    assert _rust.representations_equivalent("DATE", "4 march 2014", "2014-03-04")
    assert _rust.representations_equivalent("TIME", "04:30 p.m.", "4.30 PM")
    assert _rust.representations_equivalent("ORDINAL", "VIII", "the eighth")
    assert _rust.representations_equivalent("PHONE", "3292-3297", "329-23297")
