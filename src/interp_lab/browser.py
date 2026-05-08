from __future__ import annotations

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def build_browser(reports_dir: str | Path) -> Path:
    reports = Path(reports_dir)
    index = reports / "index.html"
    index.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>interp-lab results</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; margin: 32px; color: #1c2430; background: #f7f7f4; }
    main { max-width: 1040px; margin: 0 auto; }
    h1 { font-size: 40px; margin-bottom: 4px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
    .card { background: white; border: 1px solid #d9ded8; border-radius: 8px; padding: 18px; }
    img { max-width: 100%; border: 1px solid #e0e0dc; border-radius: 6px; }
    a { color: #225e72; }
  </style>
</head>
<body>
<main>
  <h1>interp-lab</h1>
  <p>Mechanistic interpretability artifacts from the local toy runs.</p>
  <section class="grid">
    <article class="card"><h2>Patching heatmap</h2><img src="assets/patching_heatmap.png" alt="Activation patching heatmap"></article>
    <article class="card"><h2>Attention pattern</h2><img src="assets/attention_pattern.png" alt="Attention pattern"></article>
    <article class="card"><h2>SAE feature distribution</h2><img src="assets/sae_feature_histogram.png" alt="SAE feature histogram"></article>
    <article class="card"><h2>NLA reconstruction</h2><img src="assets/nla_reconstruction.png" alt="NLA reconstruction comparison"></article>
  </section>
  <p><a href="patching_case_study.md">Patching case study</a> · <a href="sae_feature_cards.md">SAE cards</a> · <a href="nla_toy_card.md">Tiny NLA card</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index


def serve_reports(reports_dir: str | Path, port: int = 8000) -> None:
    reports = Path(reports_dir).resolve()
    build_browser(reports)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(reports), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving interp-lab browser at http://127.0.0.1:{port}")
    server.serve_forever()
