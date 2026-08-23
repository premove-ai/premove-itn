from premove_itn import _rust


def test_python_imports_production_rust_realizer() -> None:
    assert _rust.realize("TIME", "four thirty") == "04:30"
