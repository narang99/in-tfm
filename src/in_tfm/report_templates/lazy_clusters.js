// Cluster bodies ship inside <template>, so the browser parses them without building their DOM.
// A big report holds hundreds of clusters of token spans, and building them all is what makes it slow.
// Each body is stamped into the page once its section comes near the viewport.
function stampCluster(section) {
  const template = section.querySelector("template.lazy-body");
  if (!template) return;
  template.parentElement.classList.remove("pending");
  template.replaceWith(template.content);
  nearView.unobserve(section);
}

const nearView = new IntersectionObserver(
  (entries) => entries.filter((e) => e.isIntersecting).forEach((e) => stampCluster(e.target)),
  { rootMargin: "1500px 0px" }
);
document.querySelectorAll("section.cluster").forEach((section) => nearView.observe(section));

// A link to a cluster lands on an unstamped placeholder, and the placeholders above it grow once stamped.
// So stamp everything down to it first. Charts can still resize, so keep following it until the reader moves.
let followedCluster = null;
new ResizeObserver(() => followedCluster?.scrollIntoView()).observe(document.body);
for (const reading of ["wheel", "touchstart", "keydown", "mousedown"]) {
  window.addEventListener(reading, () => { followedCluster = null; }, { passive: true });
}

function showLinkedCluster() {
  const target = location.hash && document.getElementById(location.hash.slice(1));
  if (!target || !target.matches("section.cluster")) return;
  const sections = [...document.querySelectorAll("section.cluster")];
  sections.slice(0, sections.indexOf(target) + 1).forEach(stampCluster);
  followedCluster = target;
  target.scrollIntoView();
}
window.addEventListener("hashchange", showLinkedCluster);
showLinkedCluster();
