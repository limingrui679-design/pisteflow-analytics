from pisteflow.sources.mit_ocw import _license_identity, infer_action


def test_infer_mit_ocw_actions() -> None:
    assert infer_action("step_forward_group-220k_512kb.mp4") == "advance"
    assert infer_action("parry_4_repost-220k_512kb.mp4") == "parry_4_riposte"
    assert infer_action("jumpe_lunge-220k_512kb.mp4") == "jump_lunge"


def test_mit_ocw_license_identity_ignores_http_scheme() -> None:
    assert _license_identity("http://creativecommons.org/licenses/by-nc-sa/3.0/") == (
        "creativecommons.org",
        "/licenses/by-nc-sa/3.0",
    )
