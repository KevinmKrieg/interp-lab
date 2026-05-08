import torch

from interp_lab.nla_toy import TinyNLA
from interp_lab.sae import SparseAutoencoder


def test_sae_encode_decode_shapes():
    model = SparseAutoencoder(d_model=12, n_features=24)
    x = torch.randn(5, 12)
    reconstruction, features = model(x)
    assert reconstruction.shape == x.shape
    assert features.shape == (5, 24)
    assert torch.all(features >= 0)


def test_tiny_nla_shapes_and_descriptions():
    model = TinyNLA(d_model=10, n_labels=9)
    x = torch.randn(6, 10)
    reconstruction, logits = model(x)
    assert reconstruction.shape == x.shape
    assert logits.shape == (6, 9)
    assert len(model.describe(x[0], top_k=2)) == 2
