from instrument_localization.config import ExperimentConfig, load_config


def test_attention_config_loads():
    config = load_config("configs/attention.yaml")
    assert config.model_type == "attention"
    assert len(config.target_instruments) == 5


def test_even_smoothing_window_is_rejected():
    config = ExperimentConfig(smoothing_frames=4)
    try:
        config.validate()
    except ValueError as error:
        assert "positive odd integer" in str(error)
    else:
        raise AssertionError("expected invalid smoothing window to be rejected")

