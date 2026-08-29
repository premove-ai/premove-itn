from premove_itn import _rust


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
