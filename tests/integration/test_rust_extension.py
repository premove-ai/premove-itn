from premove_itn import _rust


def test_python_imports_production_rust_realizer() -> None:
    assert _rust.realize("TIME", "four thirty") == "04:30"
    assert _rust.realize_options("CARDINAL", "two") == ["2"]
    assert _rust.representations_equivalent("DATE", "4 march 2014", "2014-03-04")
