from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _json_script(name: str, payload: dict) -> str:
    return f'<script type="application/json" id="{name}">{json.dumps(payload)}</script>'


def build_browser(reports_dir: str | Path) -> Path:
    reports = Path(reports_dir)
    index = reports / "index.html"
    patching = _load_json(reports / "patching_summary.json", {})
    sae = _load_json(reports / "sae_feature_cards.json", {"metrics": {}, "features": []})
    nla = _load_json(reports / "nla_toy_summary.json", {})
    backend = "GPT-2" if patching.get("backend") == "hf" else "Toy"
    index.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>interp-lab results</title>
  <style>
    :root {{
      --bg: #f6f5f0;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #68727d;
      --line: #d9ddd6;
      --accent: #276b6f;
      --accent-2: #8b4a38;
      --soft: #eaf1ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 22px 48px; }}
    header {{ display: flex; justify-content: space-between; gap: 20px; align-items: end; margin-bottom: 24px; }}
    h1 {{ font-size: 42px; line-height: 1; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 24px; margin: 0 0 14px; }}
    h3 {{ font-size: 17px; margin: 0 0 8px; }}
    p {{ margin: 0; color: var(--muted); }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .badge {{ display: inline-flex; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: var(--panel); color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }}
    .hero {{ grid-column: span 8; }}
    .side {{ grid-column: span 4; }}
    .third {{ grid-column: span 4; }}
    .half {{ grid-column: span 6; }}
    .full {{ grid-column: 1 / -1; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfbf8; min-height: 78px; }}
    .metric strong {{ display: block; font-size: 24px; line-height: 1.1; margin-top: 4px; }}
    .metric span {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    img {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    input {{ flex: 1 1 260px; min-width: 0; }}
    select {{ flex: 0 0 180px; }}
    .feature-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .feature {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 14px; }}
    .feature-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; margin-bottom: 10px; }}
    .feature-id {{ font-weight: 700; }}
    .label {{ color: var(--accent); font-weight: 650; }}
    .example {{ border-top: 1px solid var(--line); padding-top: 8px; margin-top: 8px; }}
    code {{ background: var(--soft); border-radius: 5px; padding: 2px 5px; }}
    .links {{ display: flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; }}
    .empty {{ color: var(--muted); padding: 14px; border: 1px dashed var(--line); border-radius: 8px; }}
    @media (max-width: 840px) {{
      header {{ display: block; }}
      .hero, .side, .half, .third {{ grid-column: 1 / -1; }}
      h1 {{ font-size: 34px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>interp-lab</h1>
      <p>{backend} mechanistic interpretability report browser.</p>
    </div>
    <span class="badge">{backend} backend</span>
  </header>

  <section class="grid">
    <article class="panel hero">
      <h2>Activation Patching</h2>
      <img src="assets/patching_heatmap.png" alt="Activation patching heatmap">
      <div class="links">
        <a href="patching_case_study.md">case study</a>
        <a href="patching_summary.json">summary JSON</a>
      </div>
    </article>
    <aside class="panel side">
      <h2>Patch Metrics</h2>
      <p id="patch-copy"></p>
      <div class="metric-grid">
        <div class="metric"><span>cases</span><strong id="metric-cases">-</strong></div>
        <div class="metric"><span>best recovery</span><strong id="metric-best">-</strong></div>
        <div class="metric"><span>mean recovery</span><strong id="metric-mean">-</strong></div>
        <div class="metric"><span>best site</span><strong id="metric-site">-</strong></div>
      </div>
    </aside>

    <article class="panel half">
      <h2>Attention Pattern</h2>
      <img src="assets/attention_pattern.png" alt="Attention pattern">
    </article>
    <article class="panel half">
      <h2>NLA Bottleneck</h2>
      <img src="assets/nla_reconstruction.png" alt="NLA reconstruction comparison">
      <div class="metric-grid">
        <div class="metric"><span>cosine</span><strong id="metric-nla-cos">-</strong></div>
        <div class="metric"><span>baseline</span><strong id="metric-nla-base">-</strong></div>
      </div>
      <p id="nla-copy" style="margin-top:12px;"></p>
      <div class="links"><a href="nla_toy_card.md">NLA card</a><a href="nla_toy_summary.json">summary JSON</a></div>
    </article>

    <section class="panel full">
      <h2>SAE Feature Browser</h2>
      <div class="toolbar">
        <input id="feature-search" type="search" placeholder="Search feature labels, tokens, prompts, or ids">
        <select id="feature-sort" aria-label="Sort features">
          <option value="activation">Sort by activation</option>
          <option value="id">Sort by feature id</option>
          <option value="label">Sort by label</option>
        </select>
      </div>
      <div class="metric-grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-bottom: 16px;">
        <div class="metric"><span>features shown</span><strong id="metric-feature-count">-</strong></div>
        <div class="metric"><span>SAE MSE</span><strong id="metric-sae-mse">-</strong></div>
        <div class="metric"><span>mean L0</span><strong id="metric-sae-l0">-</strong></div>
      </div>
      <img src="assets/sae_feature_histogram.png" alt="SAE feature activation histogram" style="margin-bottom:16px;">
      <div id="feature-list" class="feature-list"></div>
      <div class="links"><a href="sae_feature_cards.md">feature cards Markdown</a><a href="sae_feature_cards.json">feature JSON</a></div>
    </section>
  </section>
</main>
{_json_script("patching-data", patching)}
{_json_script("sae-data", sae)}
{_json_script("nla-data", nla)}
<script>
  const patching = JSON.parse(document.getElementById("patching-data").textContent);
  const sae = JSON.parse(document.getElementById("sae-data").textContent);
  const nla = JSON.parse(document.getElementById("nla-data").textContent);
  const fmt = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-";

  document.getElementById("metric-cases").textContent = patching.n_cases ?? "1";
  document.getElementById("metric-best").textContent = fmt(patching.best_patch?.recovery);
  document.getElementById("metric-mean").textContent = fmt(patching.mean_best_patch_recovery ?? patching.best_patch?.recovery);
  document.getElementById("metric-site").textContent = patching.best_patch ? `L${{patching.best_patch.layer}} P${{patching.best_patch.position}}` : "-";
  document.getElementById("patch-copy").innerHTML = patching.clean_prompt
    ? `Clean <code>${{patching.clean_prompt}}</code><br>Corrupt <code>${{patching.corrupt_prompt}}</code>`
    : "Run patching to populate this section.";

  document.getElementById("metric-nla-cos").textContent = fmt(nla.mean_cosine_similarity, 3);
  document.getElementById("metric-nla-base").textContent = fmt(nla.random_text_cosine_baseline, 3);
  document.getElementById("nla-copy").innerHTML = nla.example
    ? `Example <code>${{nla.example.prompt}}</code><br>${{(nla.example.description || []).map(x => `<code>${{x}}</code>`).join(" ")}}`
    : "Run the NLA toy experiment to populate this section.";

  const search = document.getElementById("feature-search");
  const sort = document.getElementById("feature-sort");
  const list = document.getElementById("feature-list");
  const features = sae.features || [];
  document.getElementById("metric-feature-count").textContent = features.length;
  document.getElementById("metric-sae-mse").textContent = fmt(sae.metrics?.mse, 3);
  document.getElementById("metric-sae-l0").textContent = fmt(sae.metrics?.mean_l0, 1);

  function featureText(feature) {{
    return [
      feature.feature_id,
      feature.label,
      ...(feature.examples || []).flatMap(example => [example.token, example.prompt, example.position])
    ].join(" ").toLowerCase();
  }}

  function renderFeatures() {{
    const query = search.value.trim().toLowerCase();
    let visible = features.filter(feature => !query || featureText(feature).includes(query));
    if (sort.value === "id") {{
      visible = visible.sort((a, b) => a.feature_id - b.feature_id);
    }} else if (sort.value === "label") {{
      visible = visible.sort((a, b) => String(a.label).localeCompare(String(b.label)));
    }} else {{
      visible = visible.sort((a, b) => b.max_activation - a.max_activation);
    }}
    document.getElementById("metric-feature-count").textContent = visible.length;
    if (!visible.length) {{
      list.innerHTML = '<div class="empty">No features match the current filter.</div>';
      return;
    }}
    list.innerHTML = visible.map(feature => `
      <article class="feature">
        <div class="feature-head">
          <div>
            <div class="feature-id">Feature ${{feature.feature_id}}</div>
            <div class="label">${{feature.label}}</div>
          </div>
          <code>${{fmt(feature.max_activation, 2)}}</code>
        </div>
        ${{(feature.examples || []).map(example => `
          <div class="example">
            <code>${{example.prompt}}</code><br>
            token <strong>${{example.token}}</strong>, position ${{example.position}}, activation ${{fmt(example.activation, 2)}}
          </div>
        `).join("")}}
      </article>
    `).join("");
  }}

  search.addEventListener("input", renderFeatures);
  sort.addEventListener("change", renderFeatures);
  renderFeatures();
</script>
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
