from scripts.check_frozen_boundaries import verify_frozen_boundaries


def test_frozen_release_boundaries_are_consistent() -> None:
    verify_frozen_boundaries()
