"""Checkpoint provenance and rejection regression tests (requires optional Torch)."""
from __future__ import annotations

import numpy as np
import pytest

from gridguardpinn.andes_runtime import load_checkpoint, sha256_file


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from gridguardpinn.andes_surrogate import AndesMultiMachineSurrogate

    model = AndesMultiMachineSurrogate(
        np.zeros(20, dtype=np.float32),
        np.ones(20, dtype=np.float32),
    )
    path = tmp_path / "model.pt"
    payload = {
        "state_dict": model.state_dict(),
        "training": {"config": {"hidden_width": 96, "hidden_layers": 5}},
        "gate": {"residual_threshold": 0.2},
        "summary": {
            "protocol": "qa-test",
            "experiment_metadata": {"location_encoding": "one_hot"},
        },
    }
    torch.save(payload, path)
    return path, payload


def test_load_with_explicit_encoding(checkpoint):
    path, _ = checkpoint
    model, threshold, _ = load_checkpoint(
        path, expected_sha256=sha256_file(path), expected_protocol="qa-test"
    )
    assert model is not None
    assert threshold == 0.2


def test_unpinned_hash_rejected(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(
            path, expected_sha256="0" * 64, expected_protocol="qa-test"
        )


@pytest.mark.parametrize("encoding", [None, "topology"])
def test_unsupported_or_missing_encoding_rejected(checkpoint, encoding):
    torch = pytest.importorskip("torch")
    path, payload = checkpoint
    if encoding is None:
        payload["summary"]["experiment_metadata"] = {}
    else:
        payload["summary"]["experiment_metadata"]["location_encoding"] = encoding
    torch.save(payload, path)
    with pytest.raises(ValueError, match="one-hot"):
        load_checkpoint(
            path, expected_sha256=sha256_file(path), expected_protocol="qa-test"
        )


def test_wrong_protocol_rejected(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="protocol"):
        load_checkpoint(
            path, expected_sha256=sha256_file(path), expected_protocol="wrong"
        )
