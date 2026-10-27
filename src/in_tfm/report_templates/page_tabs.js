function showTab(group, index) {
  document.querySelectorAll('[data-g="' + group + '"]').forEach(
    (element) => element.classList.toggle('on', element.dataset.n === index)
  );
  // a plot drawn while hidden has no width, and redraws on resize
  window.dispatchEvent(new Event('resize'));
}
