// Filters every stored sample by cluster and probability, and draws the matching ones a page at a time.
(function () {
  const data = window.EXAMPLES;
  const groupWord = window.GROUP_WORD || "cluster";
  const PAGE = 25;
  const ACTIVATION_STOPS = [[0, [255, 255, 255]], [0.15, [254, 243, 199]], [0.4, [253, 186, 116]], [0.7, [249, 115, 22]], [1, [190, 18, 60]]];
  const NEGATIVE_STOPS = [[0, [255, 255, 255]], [1, [147, 197, 253]]];
  const PROBABILITY_STOPS = [[0, [255, 255, 255]], [0.5, [196, 181, 253]], [1, [109, 40, 217]]];
  const state = { cluster: "any", min: 0.5, sort: "prob", colour: "activation", shown: PAGE };
  const clusterById = new Map(data.clusters.map((c) => [String(c.id), c]));
  const peakOf = data.samples.map((s) => Math.max(0, ...s.acts) / data.steps * data.peak);
  const $ = (id) => document.getElementById(id);

  function blend(stops, x) {
    x = Math.min(Math.max(x, 0), 1);
    for (let i = 1; i < stops.length; i++) {
      const [x1, c1] = stops[i];
      const [x0, c0] = stops[i - 1];
      if (x <= x1) {
        const t = (x - x0) / (x1 - x0);
        return c0.map((v, k) => Math.round(v + t * (c1[k] - v)));
      }
    }
    return stops[stops.length - 1][1];
  }

  function textOn(rgb) {
    const luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
    return luminance > 0.55 ? "#111" : "#fff";
  }

  function probabilityFor(hit, cluster) {
    if (cluster === "any") return hit[1].length ? hit[1][0][1] / 100 : 0;
    const found = hit[1].find(([c]) => String(c) === cluster);
    return found ? found[1] / 100 : 0;
  }

  // A hit is always drawn under its own most probable cluster. The box shows whether it passes the filter.
  function topCluster(hit) {
    return hit[1].length ? String(hit[1][0][0]) : null;
  }

  function matching(sample) {
    return sample.hits.filter((hit) => probabilityFor(hit, state.cluster) >= state.min - 1e-9);
  }

  function filtered() {
    const rows = [];
    data.samples.forEach((sample, i) => {
      const hits = matching(sample);
      if (hits.length) rows.push({ i, hits, best: Math.max(...hits.map((h) => probabilityFor(h, state.cluster))) });
    });
    const key = state.sort === "prob" ? (r) => r.best * 10 + peakOf[r.i] / data.peak : (r) => peakOf[r.i];
    return rows.sort((a, b) => key(b) - key(a));
  }

  function tokenBackground(sample, k, hit) {
    if (state.colour === "probability") {
      return hit ? blend(PROBABILITY_STOPS, probabilityFor(hit, state.cluster)) : [255, 255, 255];
    }
    const step = sample.acts[k] / data.steps;
    return step >= 0 ? blend(ACTIVATION_STOPS, step) : blend(NEGATIVE_STOPS, -step);
  }

  function tokenTitle(text, value, hit) {
    const clusters = hit ? "  hit: " + hit[1].map(([c, p]) => `${clusterById.get(String(c)).name} ${(p / 100).toFixed(2)}`).join(", ") : "";
    return `${text}  ${value.toFixed(3)}${clusters}`;
  }

  function sampleCard(row) {
    const sample = data.samples[row.i];
    const hitAt = new Map(sample.hits.map((h) => [h[0], h]));
    const passing = new Set(row.hits.map((h) => h[0]));
    const card = document.createElement("div");
    card.style.cssText = "background:#fff;color:#111;border-radius:6px;padding:10px 12px;margin:10px 0;border:1px solid #ddd";
    const head = document.createElement("div");
    head.style.cssText = "font-size:0.8rem;color:#555;margin-bottom:6px";
    head.textContent = `sample ${sample.id} · peak ${peakOf[row.i].toFixed(3)} · ${row.hits.length} of ${sample.hits.length} hits pass · best probability ${row.best.toFixed(2)}`;
    card.appendChild(head);
    const body = document.createElement("div");
    body.style.cssText = "line-height:2.2;font-size:0.85rem";
    sample.tokens.forEach((wordId, k) => {
      const text = data.vocab[wordId];
      const hit = hitAt.get(k);
      const rgb = tokenBackground(sample, k, hit);
      const span = document.createElement("span");
      span.textContent = text;
      span.title = tokenTitle(text, sample.acts[k] / data.steps * data.peak, hit);
      span.style.cssText = `white-space:pre-wrap;padding:2px 1px;margin-right:2px;border-radius:2px;background:rgb(${rgb});color:${textOn(rgb)}`;
      const cluster = hit ? clusterById.get(topCluster(hit) || "") : null;
      if (cluster) {
        span.style.borderBottom = `3px solid ${cluster.colour}`;
        if (passing.has(k)) span.style.outline = "2px solid #111";
      }
      if (cluster && cluster.href) {
        const link = document.createElement("a");
        link.href = cluster.href;
        link.style.cssText = "color:inherit;text-decoration:none";
        link.appendChild(span);
        body.appendChild(link);
      } else {
        body.appendChild(span);
      }
    });
    card.appendChild(body);
    return card;
  }

  function legend() {
    const stops = state.colour === "probability" ? PROBABILITY_STOPS : ACTIVATION_STOPS;
    const label = state.colour === "probability" ? ["p 0", "p 1"] : ["0", data.peak.toFixed(2)];
    const box = $("ex-legend");
    box.innerHTML = "";
    box.append(label[0]);
    for (let x = 0; x <= 1.0001; x += 0.1) {
      const swatch = document.createElement("span");
      swatch.style.cssText = `display:inline-block;width:14px;height:12px;background:rgb(${blend(stops, x)});border:1px solid #ccc`;
      box.appendChild(swatch);
    }
    box.append(label[1]);
  }

  function render() {
    const rows = filtered();
    $("ex-min-value").textContent = state.min.toFixed(2);
    $("ex-count").textContent = `${rows.length} of ${data.samples.length} samples match`;
    const list = $("ex-list");
    list.innerHTML = "";
    rows.slice(0, state.shown).forEach((row) => list.appendChild(sampleCard(row)));
    $("ex-more").style.display = rows.length > state.shown ? "" : "none";
    legend();
    history.replaceState(null, "", "#" + new URLSearchParams({ cluster: state.cluster, min: state.min, sort: state.sort, colour: state.colour }));
  }

  function readHash() {
    new URLSearchParams(location.hash.slice(1)).forEach((value, key) => {
      if (key === "min") state.min = parseFloat(value) || state.min;
      else if (key in state) state[key] = value;
    });
  }

  function setUp() {
    const select = $("ex-cluster");
    select.add(new Option(`any ${groupWord}`, "any"));
    data.clusters.forEach((c) => select.add(new Option(`${c.name}: ${c.n_hits} hits`, String(c.id))));
    readHash();
    select.value = state.cluster;
    $("ex-min").value = state.min;
    $("ex-sort").value = state.sort;
    $("ex-colour").value = state.colour;
    const onChange = (key, read) => (event) => { state[key] = read(event.target.value); state.shown = PAGE; render(); };
    select.addEventListener("change", onChange("cluster", String));
    $("ex-min").addEventListener("input", onChange("min", parseFloat));
    $("ex-sort").addEventListener("change", onChange("sort", String));
    $("ex-colour").addEventListener("change", onChange("colour", String));
    $("ex-more").addEventListener("click", () => { state.shown += PAGE; render(); });
    render();
  }

  setUp();
})();
