"use strict";
(function () {
  const DATA = JSON.parse(document.getElementById("persistence-data").textContent);
  const SHOWN_CLUSTERS = 100, SHOWN_HISTORY = 200, SHOWN_NAMES = 12;
  const stepAt = { trees: 0, modes: 0 };
  let scope = "trees";

  const byId = id => document.getElementById(id);
  const view = () => DATA[scope];
  const countText = value => value.toLocaleString("en-US");
  const densityRatio = persistence => Math.pow(10, persistence).toFixed(persistence < 1 ? 2 : 0);

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function mergesDoneAt(step) {
    return step === 0 ? 0 : view().step_ends[step - 1];
  }

  const shownName = peak => peak.density_is_capped ? `${peak.name} (copies)` : peak.name;

  function treeNode(tree) {
    if (!tree.href) return tree.name;
    const link = element("a", "", tree.name);
    link.href = tree.href;
    link.target = "_blank";
    link.rel = "noopener";
    return link;
  }

  function nameNodes(peak) {
    if (!peak.trees.length) return [shownName(peak)];
    const nodes = peak.trees.flatMap((tree, i) => (i ? ["+", treeNode(tree)] : [treeNode(tree)]));
    return peak.density_is_capped ? [...nodes, " (copies)"] : nodes;
  }

  function nodesIn(tag, className, ...nodes) {
    const parent = element(tag, className);
    parent.append(...nodes);
    return parent;
  }

  function tileValues(example) {
    const raw = atob(example.tile), values = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const signed = raw.charCodeAt(i) << 24 >> 24;
      values[i] = signed / DATA.tile_levels * example.tile_max;
    }
    return values;
  }

  function colourIndex(value, limit) {
    const share = limit > 0 ? (value / limit + 1) / 2 : 0.5;
    return Math.min(DATA.tile_colors.length - 1, Math.max(0, Math.floor(share * DATA.tile_colors.length)));
  }

  function tileCanvas(values, limit) {
    const canvas = document.createElement("canvas");
    canvas.className = "pers-tile";
    canvas.width = DATA.tile_width;
    canvas.height = DATA.tile_height;
    const context = canvas.getContext("2d"), image = context.createImageData(DATA.tile_width, DATA.tile_height);
    values.forEach((value, i) => {
      const [r, g, b] = DATA.tile_colors[colourIndex(value, limit)];
      image.data.set([r, g, b, 255], 4 * i);
    });
    context.putImageData(image, 0, 0);
    return canvas;
  }

  function exampleRow(example, tile) {
    const row = element("div", "pers-example"), text = element("div", "pers-side");
    text.append(example.before, element("mark", "", example.token));
    const words = element("div", "pers-words");
    words.append(text, element("div", "pers-meta", `rho ${example.rho.toFixed(3)}`));
    row.append(words, tile);
    return row;
  }

  function sideColumn(role, peak, limit) {
    const column = element("div", "pers-column"), examples = element("div", "pers-examples");
    examples.append(...peak.examples.map(example => exampleRow(example, tileCanvas(tileValues(example), limit))));
    column.append(
      nodesIn("h4", "", `${role}: `, ...nameNodes(peak)),
      element("div", "pers-meta", `peak rho ${peak.rho.toFixed(3)}, ${countText(peak.hits)} hits ${view().members_are}`),
      examples,
    );
    return column;
  }

  function sharedTileLimit(peaks, route) {
    const examples = peaks.flatMap(peak => peak.examples).concat(route ? route.stops.map(stop => stop.example) : []);
    return Math.max(0, ...examples.map(example => example.tile_max));
  }

  function stopTitle(stop, position, route) {
    const isMerge = position === route.merge_stop;
    return nodesIn(
      isMerge ? "b" : "div", "pers-stop-title",
      `${position + 1} of ${route.stops.length}, `, treeNode(stop.tree), ...(isMerge ? [", merge point"] : []),
    );
  }

  function routeStop(stop, position, route, limit) {
    const card = element("div", position === route.merge_stop ? "pers-stop pers-stop-merge" : "pers-stop");
    const tile = tileCanvas(tileValues(stop.example), limit);
    card.append(stopTitle(stop, position, route), exampleRow(stop.example, tile));
    return card;
  }

  function routeSummary(route, dying, surviving) {
    const thinned = route.step > 1 ? `showing every ${route.step} points` : "showing every point";
    return `Route from ${dying.name} to ${surviving.name}, ${countText(route.n_points)} points long, ${thinned}.`;
  }

  function routeSection(route, dying, surviving, limit) {
    const section = element("div"), cards = element("div", "pers-route");
    cards.append(...route.stops.map((stop, position) => routeStop(stop, position, route, limit)));
    section.append(element("h4", "", routeSummary(route, dying, surviving)), cards);
    return section;
  }

  function persistenceSentence(merge, dying) {
    if (dying.density_is_capped) return `persistence ${merge.persistence.toFixed(3)}, not a measurement, since ${dying.name} is a copy group at the capped density. `;
    return `persistence ${merge.persistence.toFixed(3)}, so ${dying.name} is ${densityRatio(merge.persistence)} times as dense as where they met. `;
  }

  function mergeCard(merge) {
    const dying = view().peaks[merge.dying], surviving = view().peaks[merge.surviving];
    const card = element("div", "pers-merge"), columns = element("div", "pers-columns");
    const limit = sharedTileLimit([dying, surviving], merge.route);
    columns.append(sideColumn("Merging", dying, limit), sideColumn("Merged into", surviving, limit));
    card.append(
      nodesIn("h3", "", ...nameNodes(dying), " merges into ", ...nameNodes(surviving)),
      element("div", "pers-meta", persistenceSentence(merge, dying) + `They met at rho ${merge.saddle_rho.toFixed(3)}.`),
    );
    if (merge.route) card.append(routeSection(merge.route, dying, surviving, limit));
    card.append(columns);
    return card;
  }

  function clustersAfter(done) {
    const peaks = view().peaks, owner = peaks.map((_, i) => i);
    const find = node => {
      while (owner[node] !== node) { owner[node] = owner[owner[node]]; node = owner[node]; }
      return node;
    };
    for (const merge of view().merges.slice(0, done)) owner[find(merge.dying)] = find(merge.surviving);
    const members = new Map();
    peaks.forEach((_, i) => {
      const head = find(i);
      if (!members.has(head)) members.set(head, []);
      members.get(head).push(i);
    });
    return [...members].map(([head, list]) => ({
      head, list, hits: list.reduce((sum, i) => sum + peaks[i].hits, 0),
    })).sort((a, b) => b.hits - a.hits);
  }

  function namesText(list, head) {
    const peaks = view().peaks;
    const others = list.filter(i => i !== head).sort((a, b) => peaks[b].rho - peaks[a].rho);
    const shown = others.slice(0, SHOWN_NAMES).map(i => peaks[i].name).join(", ");
    return others.length > SHOWN_NAMES ? `${shown}, and ${others.length - SHOWN_NAMES} more` : shown;
  }

  function cell(text, className) {
    return element("td", className, text);
  }

  function hitCell(peak) {
    const td = element("td", "text"), example = peak.examples[0];
    td.append(example.before, element("mark", "", example.token));
    return td;
  }

  function clusterRow(cluster) {
    const peak = view().peaks[cluster.head], row = element("tr");
    row.append(
      nodesIn("td", "", ...nameNodes(peak)), cell(peak.rho.toFixed(3), "num"), cell(countText(cluster.list.length), "num"),
      cell(countText(cluster.hits), "num"), cell(namesText(cluster.list, cluster.head)), hitCell(peak),
    );
    return row;
  }

  function historyRow(merge) {
    const peaks = view().peaks, row = element("tr");
    row.append(
      cell(merge.persistence.toFixed(3), "num"), nodesIn("td", "", ...nameNodes(peaks[merge.dying])),
      nodesIn("td", "", ...nameNodes(peaks[merge.surviving])), cell(merge.saddle_rho.toFixed(3), "num"),
    );
    return row;
  }

  function drawReadout(step, done, nClusters) {
    const merges = view().merges, nSteps = view().step_ends.length;
    const threshold = step === 0 ? "none yet" : merges[done - 1].persistence.toFixed(3);
    byId("readout").textContent =
      `Step ${countText(step)} of ${countText(nSteps)}. Threshold ${threshold}. ` +
      `${countText(done)} of ${countText(merges.length)} merges done. ${countText(nClusters)} clusters left.`;
  }

  function drawStepMerges(step, done) {
    const box = byId("step-merges");
    if (step === 0) {
      box.replaceChildren(element("p", "pers-meta", "No merges yet. Press Next or move the slider."));
      return;
    }
    box.replaceChildren(...view().merges.slice(mergesDoneAt(step - 1), done).map(mergeCard));
  }

  function drawClusters(clusters) {
    const noPeak = view().hits_with_no_peak;
    byId("clusters-note").textContent =
      `Largest first, showing ${countText(Math.min(SHOWN_CLUSTERS, clusters.length))} of ${countText(clusters.length)}. ` +
      (noPeak ? `${countText(noPeak)} hits are in graph pieces that hold no peak, so they are in no cluster here.` : "");
    byId("clusters").replaceChildren(...clusters.slice(0, SHOWN_CLUSTERS).map(clusterRow));
  }

  function drawHistory(done) {
    const latest = view().merges.slice(Math.max(0, done - SHOWN_HISTORY), done).reverse();
    byId("history-note").textContent = `Showing ${countText(latest.length)} of ${countText(done)}.`;
    byId("history").replaceChildren(...latest.map(historyRow));
  }

  function draw() {
    const step = stepAt[scope], done = mergesDoneAt(step), nSteps = view().step_ends.length;
    const clusters = clustersAfter(done), slider = byId("step");
    slider.max = String(nSteps);
    slider.value = String(step);
    byId("step-back").disabled = step === 0;
    byId("step-next").disabled = step === nSteps;
    byId("scope-trees").setAttribute("aria-pressed", String(scope === "trees"));
    byId("scope-modes").setAttribute("aria-pressed", String(scope === "modes"));
    drawReadout(step, done, clusters.length);
    drawStepMerges(step, done);
    drawClusters(clusters);
    drawHistory(done);
  }

  function goTo(step) {
    stepAt[scope] = Math.min(view().step_ends.length, Math.max(0, step));
    draw();
  }

  function useScope(next) {
    scope = next;
    draw();
  }

  byId("step").addEventListener("input", event => goTo(Number(event.target.value)));
  byId("step-back").addEventListener("click", () => goTo(stepAt[scope] - 1));
  byId("step-next").addEventListener("click", () => goTo(stepAt[scope] + 1));
  byId("scope-trees").addEventListener("click", () => useScope("trees"));
  byId("scope-modes").addEventListener("click", () => useScope("modes"));
  draw();
})();
