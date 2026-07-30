from nway_protocol.stage_real_artifact import _plurality, _to_class


def test_label_normalization_and_tie_safe_plurality():
    assert _to_class("'seizure'") == "sz"
    assert _to_class("BIPDs") == "iic"
    assert _to_class(None) is None
    assert _plurality(["lpd", "lpd", "gpd"]) == "lpd"
    assert _plurality(["lpd", "gpd"]) is None
