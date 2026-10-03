import pytest

torch = pytest.importorskip("torch")

from instrument_localization.models import (  # noqa: E402
    AttentionInstrumentModel,
    BaselineInstrumentModel,
    MaxInstrumentModel,
)


def test_baseline_output_shapes():
    model = BaselineInstrumentModel(n_mels=64, num_instruments=5)
    output = model(torch.randn(2, 64, 40))
    assert output["clip_logits"].shape == (2, 5)
    assert output["frame_logits"].shape == (2, 40, 5)
    assert torch.allclose(
        output["clip_probabilities"], output["frame_probabilities"].mean(dim=1)
    )
    assert torch.allclose(
        torch.sigmoid(output["clip_logits"]), output["clip_probabilities"], atol=1e-6
    )


def test_max_pooling_uses_frame_probabilities():
    model = MaxInstrumentModel(n_mels=64, num_instruments=5)
    output = model(torch.randn(2, 64, 40))
    assert torch.allclose(
        output["clip_probabilities"], output["frame_probabilities"].max(dim=1).values
    )


def test_attention_weights_sum_over_time():
    model = AttentionInstrumentModel(n_mels=64, num_instruments=5)
    output = model(torch.randn(2, 64, 40))
    assert output["clip_logits"].shape == (2, 5)
    assert output["attention_weights"].shape == (2, 40, 5)
    expected = torch.ones(2, 5)
    assert torch.allclose(output["attention_weights"].sum(dim=1), expected, atol=1e-5)
    reconstructed = (output["attention_weights"] * output["frame_probabilities"]).sum(dim=1)
    assert torch.allclose(output["clip_probabilities"], reconstructed, atol=1e-6)
    assert torch.allclose(torch.sigmoid(output["clip_logits"]), reconstructed, atol=1e-6)

