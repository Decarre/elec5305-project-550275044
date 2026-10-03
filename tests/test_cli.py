import yaml

from instrument_localization.cli import main


def test_validate_command_and_legacy_dry_run(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"target_instruments": ["drums", "bass"]}))
    assert main(["validate", "--config", str(config)]) == 0
    assert '\"target_instruments\"' in capsys.readouterr().out
    assert main(["--config", str(config), "--dry-run"]) == 0
    assert '\"model_type\"' in capsys.readouterr().out
