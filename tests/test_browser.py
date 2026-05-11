import json

from interp_lab.browser import build_browser


def test_browser_embeds_feature_data(tmp_path):
    reports = tmp_path
    (reports / "assets").mkdir()
    (reports / "patching_summary.json").write_text(
        json.dumps(
            {
                "backend": "toy",
                "n_cases": 1,
                "best_patch": {"layer": 1, "position": 2, "recovery": 0.5},
                "cases": [
                    {
                        "clean_prompt": "A B",
                        "corrupt_prompt": "A C",
                        "clean_logit_diff": 1.0,
                        "corrupt_logit_diff": 0.2,
                        "best_layer": 1,
                        "best_position": 2,
                        "best_token": "B",
                        "best_recovery": 0.5,
                    }
                ],
                "skipped_cases": [{"clean_prompt": "A D", "reason": "low margin"}],
            }
        ),
        encoding="utf-8",
    )
    (reports / "sae_feature_cards.json").write_text(
        json.dumps(
            {
                "metrics": {"mse": 0.1, "mean_l0": 4},
                "features": [
                    {
                        "feature_id": 3,
                        "label": "copy-token feature",
                        "max_activation": 2.5,
                        "examples": [{"prompt": "A B A", "token": "A", "position": 2, "activation": 2.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (reports / "nla_toy_summary.json").write_text(json.dumps({"mean_cosine_similarity": 0.7}), encoding="utf-8")

    index = build_browser(reports)
    html = index.read_text(encoding="utf-8")

    assert "SAE Feature Browser" in html
    assert "Aggregate Patching" in html
    assert "Patching Case Table" in html
    assert "copy-token feature" in html
    assert "feature-search" in html
