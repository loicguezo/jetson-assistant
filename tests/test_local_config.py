"""Unit tests for settings.local.yaml override and config wizard helpers."""

from unittest.mock import patch

import yaml

from app.config import Config, _config_paths

# ── _config_paths ─────────────────────────────────────────────────


def test_config_paths_returns_two_entries(tmp_path):
    base = tmp_path / "settings.yaml"
    paths = _config_paths(base)
    assert len(paths) == 2
    assert paths[0] == base
    assert paths[1] == tmp_path / "settings.local.yaml"


# ── Config.load with local override ───────────────────────────────


def test_local_config_overrides_base(tmp_path):
    base = tmp_path / "settings.yaml"
    base.write_text("tts:\n  backend: kokoro\n  voice: af_sarah\n")
    local = tmp_path / "settings.local.yaml"
    local.write_text("tts:\n  backend: piper\n  piper_model: de_DE-thorsten-high\n")

    cfg = Config.load(config_path=str(base))
    assert cfg.tts.backend == "piper"
    assert cfg.tts.piper_model == "de_DE-thorsten-high"
    assert cfg.tts.voice == "af_sarah"  # untouched by local


def test_local_config_missing_is_silently_ignored(tmp_path):
    base = tmp_path / "settings.yaml"
    base.write_text("tts:\n  backend: kokoro\n")
    # No settings.local.yaml — should not raise
    cfg = Config.load(config_path=str(base))
    assert cfg.tts.backend == "kokoro"


def test_local_config_partial_section_override(tmp_path):
    base = tmp_path / "settings.yaml"
    base.write_text("stt:\n  language: en\n  model: small.en\n")
    local = tmp_path / "settings.local.yaml"
    local.write_text("stt:\n  language: de\n")

    cfg = Config.load(config_path=str(base))
    assert cfg.stt.language == "de"
    assert cfg.stt.model == "small.en"  # unchanged


def test_env_var_wins_over_local_config(tmp_path):
    base = tmp_path / "settings.yaml"
    base.write_text("tts:\n  backend: kokoro\n")
    local = tmp_path / "settings.local.yaml"
    local.write_text("tts:\n  backend: piper\n")

    with patch.dict("os.environ", {"JA_TTS_BACKEND": "kokoro"}):
        cfg = Config.load(config_path=str(base))
    assert cfg.tts.backend == "kokoro"  # env var wins


# ── write_local_config ─────────────────────────────────────────────


def test_write_local_config_creates_file(tmp_path):
    from manage import write_local_config

    path = tmp_path / "settings.local.yaml"
    write_local_config(path, {"tts": {"backend": "piper"}})

    assert path.exists()
    data = yaml.safe_load(path.read_text())
    assert data["tts"]["backend"] == "piper"


def test_write_local_config_merges_into_existing(tmp_path):
    from manage import write_local_config

    path = tmp_path / "settings.local.yaml"
    path.write_text("tts:\n  backend: piper\n  speed: 1.2\n")

    write_local_config(
        path, {"tts": {"piper_model": "de_DE-thorsten-high"}, "stt": {"language": "de"}}
    )

    data = yaml.safe_load(path.read_text())
    assert data["tts"]["backend"] == "piper"  # preserved
    assert data["tts"]["speed"] == 1.2  # preserved
    assert data["tts"]["piper_model"] == "de_DE-thorsten-high"  # added
    assert data["stt"]["language"] == "de"  # new section


def test_write_local_config_overwrites_existing_key(tmp_path):
    from manage import write_local_config

    path = tmp_path / "settings.local.yaml"
    path.write_text("tts:\n  backend: kokoro\n")

    write_local_config(path, {"tts": {"backend": "piper"}})

    data = yaml.safe_load(path.read_text())
    assert data["tts"]["backend"] == "piper"


def test_write_local_config_creates_parent_dirs(tmp_path):
    from manage import write_local_config

    path = tmp_path / "nested" / "dir" / "settings.local.yaml"
    write_local_config(path, {"tts": {"backend": "piper"}})
    assert path.exists()


def test_write_local_config_includes_header(tmp_path):
    from manage import write_local_config

    path = tmp_path / "settings.local.yaml"
    write_local_config(path, {"tts": {"backend": "piper"}})

    raw = path.read_text()
    assert "do not edit manually" in raw
    assert "jetson-assistant config" in raw


def test_write_local_config_recovers_from_corrupt_file(tmp_path):
    from manage import write_local_config

    path = tmp_path / "settings.local.yaml"
    path.write_text(":::invalid yaml:::\n{[broken")

    # Should not raise — treats corrupt file as empty and overwrites
    write_local_config(path, {"tts": {"backend": "piper"}})

    data = yaml.safe_load(path.read_text())
    assert data["tts"]["backend"] == "piper"


def test_write_local_config_recovers_from_non_dict_yaml(tmp_path):
    from manage import write_local_config

    path = tmp_path / "settings.local.yaml"
    path.write_text("- just\n- a\n- list\n")  # valid YAML but not a dict

    write_local_config(path, {"tts": {"backend": "piper"}})

    data = yaml.safe_load(path.read_text())
    assert data["tts"]["backend"] == "piper"


def test_local_config_corrupt_falls_back_to_base(tmp_path):
    base = tmp_path / "settings.yaml"
    base.write_text("tts:\n  backend: kokoro\n")
    local = tmp_path / "settings.local.yaml"
    local.write_text(":::not valid yaml:::")

    # Should not raise — corrupt local config is ignored, base values used
    cfg = Config.load(config_path=str(base))
    assert cfg.tts.backend == "kokoro"


def test_local_config_deleted_after_load(tmp_path):
    base = tmp_path / "settings.yaml"
    base.write_text("tts:\n  backend: kokoro\n")
    local = tmp_path / "settings.local.yaml"
    local.write_text("tts:\n  backend: piper\n")

    local.unlink()  # delete between runs

    # Second load should fall back to base without error
    cfg = Config.load(config_path=str(base))
    assert cfg.tts.backend == "kokoro"
