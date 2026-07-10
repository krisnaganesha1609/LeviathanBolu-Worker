from tts.personalities import PersonalityRegistry


def test_loads_leviathan_and_bolu_from_yaml():
    reg = PersonalityRegistry("config/personalities.yaml")
    assert set(reg.names()) >= {"LEVIATHAN", "BOLU"}


def test_resolve_leviathan_matches_spec():
    reg = PersonalityRegistry("config/personalities.yaml")
    cfg, matched = reg.resolve("LEVIATHAN")
    assert matched is True
    assert cfg.voice == "am_adam"
    assert cfg.speed == 0.90
    assert cfg.pitch == -1


def test_resolve_bolu_matches_spec():
    reg = PersonalityRegistry("config/personalities.yaml")
    cfg, matched = reg.resolve("BOLU")
    assert matched is True
    assert cfg.voice == "af_bella"
    assert cfg.speed == 1.15
    assert cfg.pitch == 2


def test_resolve_unknown_falls_back_to_default_without_raising():
    reg = PersonalityRegistry("config/personalities.yaml")
    cfg, matched = reg.resolve("NONEXISTENT_PERSONALITY")
    assert matched is False
    assert cfg.voice == reg.resolve(reg.default_name)[0].voice


def test_resolve_is_case_insensitive():
    reg = PersonalityRegistry("config/personalities.yaml")
    cfg_lower, matched = reg.resolve("leviathan")
    assert matched is True
    assert cfg_lower.voice == "am_adam"
