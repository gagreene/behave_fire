"""Verify directory-derived marks on the actual pytest collection."""


def test_collection_tier_markers_match_directories(request):
    expected = {"unit": [], "integration": ["integration"], "regression": ["regression"]}
    test_root = request.config.rootpath / "tests"
    for item in request.session.items:
        tier = item.path.relative_to(test_root).parts[0]
        marks = [mark.name for mark in item.iter_markers() if mark.name in {"integration", "regression"}]
        assert tier in expected, f"{item.nodeid} is outside the test tiers"
        assert marks == expected[tier], f"{item.nodeid} has tier markers {marks!r}, expected {expected[tier]!r}"
