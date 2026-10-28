function showTab(group, index) {
  document.querySelectorAll('[data-g="' + group + '"]').forEach(
    (element) => element.classList.toggle('on', element.dataset.n === index)
  );
  // a plot drawn while hidden has no width, and redraws on resize
  window.dispatchEvent(new Event('resize'));
}

// The up-close plot has three traces per tree, in tree order: links, points, root.
function showPicks(picker) {
  const trees = Number(picker.dataset.trees);
  const on = [...picker.querySelectorAll('.pick.on')].map((button) => Number(button.dataset.t));
  const visible = [];
  for (let t = 0; t < trees; t++) {
    const shown = on.includes(t);
    visible.push(shown, shown, shown);
  }
  Plotly.restyle(picker.dataset.plot, { visible: visible });
  Plotly.relayout(picker.dataset.plot, { 'xaxis.autorange': true, 'yaxis.autorange': true });
}

function togglePick(button) {
  button.classList.toggle('on');
  showPicks(button.closest('.picker'));
}

function pickAll(button, on) {
  const picker = button.closest('.picker');
  picker.querySelectorAll('.pick').forEach((pick) => pick.classList.toggle('on', on));
  showPicks(picker);
}

// Clicking a point on the map jumps to its cluster's card.
window.addEventListener('load', () => {
  const map = document.getElementById('hit-map');
  if (map && map.on) {
    map.on('plotly_click', (event) => {
      location.hash = 'cluster-' + event.points[0].customdata;
    });
  }
});
