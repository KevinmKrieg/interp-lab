from interp_lab.models import make_model


def test_toy_model_dispatch():
    config = {"seed": 7, "model": {"backend": "toy", "n_layers": 2, "d_model": 8}}
    model = make_model(config)
    assert model.n_layers == 2
    assert model.d_model == 8
